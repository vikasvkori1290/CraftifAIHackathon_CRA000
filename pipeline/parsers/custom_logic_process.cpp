#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Persistent state for zone tracking (keyed by track_id) ──────────────────
struct ZoneRecord {
    int streak;
    int frames_in_zone;
    std::string zone_name;
    std::string last_zone_name;
    int peak_streak;
};
static std::map<guint64, ZoneRecord> g_zone_records;

// ── PPE class names (6-class PPE model) ─────────────────────────────────────
// class 0: Gloves, 1: Vest, 2: goggles, 3: helmet, 4: mask, 5: safety_shoe
// We only care about class 1 (Vest) and class 3 (helmet).

// ── Structures for internal logic ───────────────────────────────────────────
struct PersonTrack {
    guint64 track_id;
    float x1, y1, x2, y2;
    float score;
};

struct PPEDet {
    float x1, y1, x2, y2;
    float score;
    int cls; // 1=Vest, 3=helmet
};

struct PPECompliance {
    guint64 track_id;
    float bx1, by1, bx2, by2;
    bool has_hat;
    bool has_vest;
    float hat_score;
    float vest_score;
    bool compliant;
};

struct ZoneStatus {
    guint64 track_id;
    float bx1, by1, bx2, by2;
    float score;
    std::string zone;
    std::string status; // "INTRUSION!", "IN ZONE", "CLEAR"
    int streak;
    float dwell_seconds;
    bool is_entry_event; // streak == alert_frames
};

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[custom_logic] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) in->frame_meta;

    // ── Step 1: Collect person tracks (gie_id=1) and PPE detections (gie_id=2) from NvDsObjectMeta ──
    std::vector<PersonTrack> persons;
    std::vector<PPEDet> ppe_dets;

    if (frame_meta) {
        for (NvDsMetaList *ol = frame_meta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;

            if (obj->unique_component_id == 1) {
                // Person detector (class 0 = person in COCO)
                PersonTrack pt;
                pt.track_id = obj->object_id;
                pt.x1 = obj->rect_params.left;
                pt.y1 = obj->rect_params.top;
                pt.x2 = obj->rect_params.left + obj->rect_params.width;
                pt.y2 = obj->rect_params.top + obj->rect_params.height;
                pt.score = obj->confidence;
                persons.push_back(pt);
            } else if (obj->unique_component_id == 2) {
                // PPE detector: class 1=Vest, 3=helmet
                int cls = obj->class_id;
                if (cls == 1 || cls == 3) {
                    PPEDet pd;
                    pd.x1 = obj->rect_params.left;
                    pd.y1 = obj->rect_params.top;
                    pd.x2 = obj->rect_params.left + obj->rect_params.width;
                    pd.y2 = obj->rect_params.top + obj->rect_params.height;
                    pd.score = obj->confidence;
                    pd.cls = cls;
                    ppe_dets.push_back(pd);
                }
            }
        }
    }

    // ── Step 2: PPE Compliance (Node 9) ─────────────────────────────────────
    // Associate helmet (cls=3) and vest (cls=1) detections with person tracks
    // using body-region overlap with greedy assignment.

    struct Assignment {
        bool has_hat;
        bool has_vest;
        float hat_score;
        float vest_score;
        int hat_idx;
        int vest_idx;
    };

    std::map<guint64, Assignment> assigned;
    for (const PersonTrack &p : persons) {
        Assignment a;
        a.has_hat = false; a.has_vest = false;
        a.hat_score = 0.0f; a.vest_score = 0.0f;
        a.hat_idx = -1; a.vest_idx = -1;
        assigned[p.track_id] = a;
    }

    // Helper: overlap area
    auto overlap_area = [](float ax1, float ay1, float ax2, float ay2,
                           float bx1, float by1, float bx2, float by2) -> float {
        float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
        float iy = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
        return ix * iy;
    };

    // For each item type: build candidates, sort by quality desc, greedily assign
    // item_type: 0 = hard hat (cls=3, label='helmet'), 1 = vest (cls=1, label='Vest')
    for (int item_type = 0; item_type < 2; ++item_type) {
        int target_cls = (item_type == 0) ? 3 : 1; // helmet=3, Vest=1

        struct Candidate {
            float quality;
            float det_score;
            int det_idx;
            guint64 track_id;
        };
        std::vector<Candidate> candidates;

        for (int j = 0; j < (int)ppe_dets.size(); ++j) {
            const PPEDet &det = ppe_dets[j];
            if (det.cls != target_cls) continue;

            float cx = (det.x1 + det.x2) * 0.5f;
            float cy = (det.y1 + det.y2) * 0.5f;

            for (const PersonTrack &t : persons) {
                float bx1 = t.x1, by1 = t.y1, bx2 = t.x2, by2 = t.y2;
                float height = by2 - by1;
                if (height <= 0.0f || bx2 <= bx1) continue;

                float rx1, ry1, rx2, ry2;
                if (item_type == 0) {
                    // hard hat: upper 35% of person box
                    rx1 = bx1; ry1 = by1; rx2 = bx2; ry2 = by1 + 0.35f * height;
                } else {
                    // vest: 20%-80% of person box height
                    rx1 = bx1; ry1 = by1 + 0.20f * height; rx2 = bx2; ry2 = by1 + 0.80f * height;
                }

                // Check if detection center is within the region
                if (cx >= rx1 && cx <= rx2 && cy >= ry1 && cy <= ry2) {
                    float ov = overlap_area(det.x1, det.y1, det.x2, det.y2,
                                            rx1, ry1, rx2, ry2);
                    float det_area = (det.x2 - det.x1) * (det.y2 - det.y1);
                    float quality = ov / std::max(1e-9f, det_area);

                    Candidate c;
                    c.quality = quality;
                    c.det_score = det.score;
                    c.det_idx = j;
                    c.track_id = t.track_id;
                    candidates.push_back(c);
                }
            }
        }

        // Sort descending by (quality, det_score)
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
                if (a.quality != b.quality) return a.quality > b.quality;
                return a.det_score > b.det_score;
            });

        // Greedy assignment
        std::vector<bool> used_det(ppe_dets.size(), false);
        for (const Candidate &c : candidates) {
            if (used_det[c.det_idx]) continue;
            auto it = assigned.find(c.track_id);
            if (it == assigned.end()) continue;

            if (item_type == 0) {
                if (it->second.has_hat) continue;
                it->second.has_hat = true;
                it->second.hat_score = c.det_score;
                it->second.hat_idx = c.det_idx;
            } else {
                if (it->second.has_vest) continue;
                it->second.has_vest = true;
                it->second.vest_score = c.det_score;
                it->second.vest_idx = c.det_idx;
            }
            used_det[c.det_idx] = true;
        }
    }

    // Build compliance results
    std::vector<PPECompliance> compliance;
    for (const PersonTrack &p : persons) {
        auto it = assigned.find(p.track_id);
        PPECompliance pc;
        pc.track_id = p.track_id;
        pc.bx1 = p.x1; pc.by1 = p.y1; pc.bx2 = p.x2; pc.by2 = p.y2;
        if (it != assigned.end()) {
            pc.has_hat = it->second.has_hat;
            pc.has_vest = it->second.has_vest;
            pc.hat_score = it->second.hat_score;
            pc.vest_score = it->second.vest_score;
        } else {
            pc.has_hat = false; pc.has_vest = false;
            pc.hat_score = 0.0f; pc.vest_score = 0.0f;
        }
        pc.compliant = pc.has_hat && pc.has_vest;
        compliance.push_back(pc);
    }

    // ── Step 3: Hazard-zone entry (Node 10) ─────────────────────────────────
    // Default zone: full frame
    const int ALERT_FRAMES = 2;
    const float FPS_FALLBACK = 25.0f;

    // Try to get FPS from frame_meta
    float fps = FPS_FALLBACK;
    // DeepStream doesn't expose source FPS easily in frame_meta; use fallback
    // The source is 12 fps per PIR, but we use the fallback parameter as specified.

    // Zone polygon: full frame [[0,0],[W,0],[W,H],[0,H]]
    std::vector<cv::Point2f> zone_poly;
    zone_poly.push_back(cv::Point2f(0.0f, 0.0f));
    zone_poly.push_back(cv::Point2f((float)W, 0.0f));
    zone_poly.push_back(cv::Point2f((float)W, (float)H));
    zone_poly.push_back(cv::Point2f(0.0f, (float)H));
    const char *zone_name_str = "Restricted Area";

    std::vector<ZoneStatus> statuses;
    std::vector<ZoneStatus> entry_events;

    for (const PersonTrack &t : persons) {
        float foot_x = (t.x1 + t.x2) * 0.5f;
        float foot_y = t.y2;
        cv::Point2f foot(foot_x, foot_y);

        // Point-in-polygon test
        double dist = cv::pointPolygonTest(zone_poly, foot, false);
        bool in_zone = (dist >= 0.0);

        std::string cur_zone = in_zone ? zone_name_str : "";

        // Get or create persistent record
        ZoneRecord &rec = g_zone_records[t.track_id];
        // If newly inserted, all fields are zero-initialized by default map insertion
        // but let's be explicit for clarity (map value-initializes to zero for POD-like)

        if (in_zone) {
            rec.frames_in_zone += 1;
            rec.streak += 1;
            rec.zone_name = zone_name_str;
            rec.last_zone_name = zone_name_str;
        } else {
            rec.streak = 0;
            rec.zone_name.clear();
        }
        rec.peak_streak = std::max(rec.peak_streak, rec.streak);

        bool active = rec.streak >= ALERT_FRAMES;

        ZoneStatus zs;
        zs.track_id = t.track_id;
        zs.bx1 = t.x1; zs.by1 = t.y1; zs.bx2 = t.x2; zs.by2 = t.y2;
        zs.score = t.score;
        zs.zone = rec.zone_name;
        if (active) zs.status = "INTRUSION!";
        else if (in_zone) zs.status = "IN ZONE";
        else zs.status = "CLEAR";
        zs.streak = rec.streak;
        zs.dwell_seconds = (float)rec.frames_in_zone / fps;
        zs.is_entry_event = (rec.streak == ALERT_FRAMES);

        statuses.push_back(zs);

        if (rec.streak == ALERT_FRAMES) {
            entry_events.push_back(zs);
        }
    }

    // ── Step 4: Safety breach alert (Node 11) ───────────────────────────────
    // For each entry event, attach PPE compliance info

    // Build compliance lookup by track_id
    std::map<guint64, const PPECompliance*> compliance_map;
    for (const PPECompliance &pc : compliance) {
        compliance_map[pc.track_id] = &pc;
    }

    // ── Step 5: Draw and Publish ────────────────────────────────────────────
    cv::Mat frame = read_nv12_bgr(in);
    render_meta::Builder sb(W, H);

    // Draw zone polygon outline (translucent)
    {
        std::vector<cv::Point> pts;
        for (const cv::Point2f &p : zone_poly)
            pts.push_back(cv::Point((int)p.x, (int)p.y));
        // Draw zone boundary
        const cv::Scalar zone_color(0, 0, 255); // red
        cv::polylines(frame, pts, true, zone_color, 2, cv::LINE_AA);
    }

    // Colors for compliance status
    const cv::Scalar COLOR_COMPLIANT(0, 200, 0);    // green
    const cv::Scalar COLOR_NONCOMPLIANT(0, 0, 255);  // red
    const cv::Scalar COLOR_INTRUSION(0, 0, 255);     // red
    const cv::Scalar COLOR_INZONE(0, 200, 255);      // orange-yellow
    const cv::Scalar COLOR_CLEAR(0, 200, 0);         // green

    // Draw person boxes with PPE compliance + zone status
    for (size_t i = 0; i < persons.size(); ++i) {
        const PersonTrack &p = persons[i];
        // Find compliance
        const PPECompliance *pc = nullptr;
        auto cit = compliance_map.find(p.track_id);
        if (cit != compliance_map.end()) pc = cit->second;

        bool compliant = pc ? pc->compliant : false;
        cv::Scalar box_color = compliant ? COLOR_COMPLIANT : COLOR_NONCOMPLIANT;

        // Find zone status
        std::string zone_status_str;
        float dwell_s = 0.0f;
        for (const ZoneStatus &zs : statuses) {
            if (zs.track_id == p.track_id) {
                zone_status_str = zs.status;
                dwell_s = zs.dwell_seconds;
                if (zs.status == "INTRUSION!") box_color = COLOR_INTRUSION;
                break;
            }
        }

        int ix1 = (int)p.x1, iy1 = (int)p.y1, ix2 = (int)p.x2, iy2 = (int)p.y2;
        cv::rectangle(frame, cv::Point(ix1, iy1), cv::Point(ix2, iy2), box_color, 2);

        // Label: person <conf>%
        char label_buf[128];
        snprintf(label_buf, sizeof(label_buf), "person %d%%", (int)(p.score * 100.0f));
        cv::putText(frame, label_buf, cv::Point(ix1, std::max(iy1 - 6, 12)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv::LINE_AA);

        // PPE status icons: small text below the box showing hat/vest status
        {
            int yoff = iy2 + 14;
            if (pc) {
                char ppe_buf[64];
                snprintf(ppe_buf, sizeof(ppe_buf), "%s %s",
                         pc->has_hat ? "hat:Y" : "hat:N",
                         pc->has_vest ? "vest:Y" : "vest:N");
                cv::putText(frame, ppe_buf, cv::Point(ix1, std::min(yoff, H - 4)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.38,
                            compliant ? COLOR_COMPLIANT : COLOR_NONCOMPLIANT,
                            1, cv::LINE_AA);
            }
        }

        // Publish person box with compliance info
        {
            std::string lbl = compliant ? "compliant" : "non-compliant";
            sb.box((double)p.x1, (double)p.y1,
                   (double)(p.x2 - p.x1), (double)(p.y2 - p.y1),
                   lbl.c_str(), (int)p.track_id, (double)p.score);
        }
    }

    // Draw PPE detection boxes (helmet / vest)
    for (const PPEDet &pd : ppe_dets) {
        int ix1 = (int)pd.x1, iy1 = (int)pd.y1, ix2 = (int)pd.x2, iy2 = (int)pd.y2;
        const char *ppe_name = (pd.cls == 3) ? "helmet" : "Vest";
        cv::Scalar c = (pd.cls == 3) ? cv::Scalar(255, 180, 0) : cv::Scalar(0, 255, 255);
        cv::rectangle(frame, cv::Point(ix1, iy1), cv::Point(ix2, iy2), c, 1);
        char lbuf[64];
        snprintf(lbuf, sizeof(lbuf), "%s %d%%", ppe_name, (int)(pd.score * 100.0f));
        cv::putText(frame, lbuf, cv::Point(ix1, std::max(iy1 - 4, 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv::LINE_AA);
    }

    // Publish zone statuses and safety breach incidents
    for (const ZoneStatus &zs : statuses) {
        sb.field_s("zone_status", zs.status.c_str());
        sb.field_d("dwell_seconds", (double)zs.dwell_seconds);
    }

    // Publish safety breach incidents for entry events
    for (const ZoneStatus &ev : entry_events) {
        const PPECompliance *pc = nullptr;
        auto cit = compliance_map.find(ev.track_id);
        if (cit != compliance_map.end()) pc = cit->second;

        // Publish as a polygon (the zone) with incident info
        std::vector<double> zone_xy;
        for (const cv::Point2f &zp : zone_poly) {
            zone_xy.push_back((double)zp.x);
            zone_xy.push_back((double)zp.y);
        }
        sb.polygon(zone_xy, "hazard_zone");

        sb.field_s("event", "safety_breach");
        sb.field_s("reason", "hazard_zone_entry");
        sb.field_i("breach_track_id", (int)ev.track_id);
        sb.field_s("breach_zone", ev.zone.c_str());
        sb.field_d("breach_dwell_s", (double)ev.dwell_seconds);
        sb.field_d("breach_score", (double)ev.score);
        if (pc) {
            sb.field_s("breach_ppe_compliant", pc->compliant ? "true" : "false");
            sb.field_s("breach_has_hat", pc->has_hat ? "true" : "false");
            sb.field_s("breach_has_vest", pc->has_vest ? "true" : "false");
        }
    }

    // Publish aggregate counts
    {
        int active_count = 0;
        for (const ZoneStatus &zs : statuses) {
            if (zs.status == "INTRUSION!") active_count++;
        }
        sb.field_i("active_intruder_count", active_count);

        int distinct_count = 0;
        for (auto &kv : g_zone_records) {
            if (kv.second.peak_streak >= ALERT_FRAMES) distinct_count++;
        }
        sb.field_i("distinct_intruder_count", distinct_count);
    }

    write_nv12(frame, in);
    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);