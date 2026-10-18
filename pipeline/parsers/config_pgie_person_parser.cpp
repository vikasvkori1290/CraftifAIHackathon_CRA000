// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigPgiePerson(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigPgiePerson);

extern "C" bool ParseConfigPgiePerson(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[PgiePerson] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // YOLO11s output0 engine shape: [1, 84, 8400]
    // After batch-dim strip: inferDims numDims=2, d[0]=84, d[1]=8400
    // Layout is [C, N] where C = 4 + numClasses, N = number of proposals.
    // We need to transpose to [N, C] logically while decoding.

    const unsigned int dimC = layer0.inferDims.d[0]; // 84
    const unsigned int dimN = layer0.inferDims.d[1]; // 8400 (varies with input res)

    // Sanity: dimC should equal 4 + numClasses
    if (dimC != 4u + numClasses) {
        // Try swapped layout: d[0]=N, d[1]=C
        if (layer0.inferDims.d[1] == 4u + numClasses) {
            // Already [N, C] — fall through with swapped interpretation
            const unsigned int numProposals = layer0.inferDims.d[0];
            const unsigned int rowSize = layer0.inferDims.d[1]; // 4 + numClasses

            for (unsigned int i = 0; i < numProposals; ++i) {
                const float* row = data + i * rowSize;
                float cx = row[0];
                float cy = row[1];
                float bw = row[2];
                float bh = row[3];

                // Find best class
                float maxScore = 0.0f;
                unsigned int bestClass = 0;
                for (unsigned int c = 0; c < numClasses; ++c) {
                    float s = row[4 + c];
                    if (s > maxScore) {
                        maxScore = s;
                        bestClass = c;
                    }
                }

                float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                    ? detectionParams.perClassPreclusterThreshold[bestClass] : 0.3f;
                if (maxScore < thr) continue;

                float left   = std::min<float>(std::max<float>(cx - bw * 0.5f, 0.0f), netW);
                float top    = std::min<float>(std::max<float>(cy - bh * 0.5f, 0.0f), netH);
                float right  = std::min<float>(std::max<float>(cx + bw * 0.5f, 0.0f), netW);
                float bottom = std::min<float>(std::max<float>(cy + bh * 0.5f, 0.0f), netH);

                float w = right - left;
                float h = bottom - top;
                if (w <= 1.0f || h <= 1.0f) continue;

                NvDsInferObjectDetectionInfo obj;
                obj.classId = bestClass;
                obj.left = left;
                obj.top = top;
                obj.width = w;
                obj.height = h;
                obj.detectionConfidence = maxScore;
                objectList.push_back(obj);
            }
            return true;
        }
        fprintf(stderr, "[PgiePerson] unexpected dims: d[0]=%u d[1]=%u numClasses=%u\n",
                layer0.inferDims.d[0], layer0.inferDims.d[1], numClasses);
        return false;
    }

    // Normal path: layout is [C, N] (= [84, 8400])
    // data[c * dimN + n] gives element at (c, n)
    // For proposal n:
    //   cx = data[0*dimN + n], cy = data[1*dimN + n]
    //   bw = data[2*dimN + n], bh = data[3*dimN + n]
    //   class_score[c] = data[(4+c)*dimN + n]

    for (unsigned int n = 0; n < dimN; ++n) {
        // Find best class score
        float maxScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float s = data[(4u + c) * dimN + n];
            if (s > maxScore) {
                maxScore = s;
                bestClass = c;
            }
        }

        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
            ? detectionParams.perClassPreclusterThreshold[bestClass] : 0.3f;
        if (maxScore < thr) continue;

        float cx = data[0u * dimN + n];
        float cy = data[1u * dimN + n];
        float bw = data[2u * dimN + n];
        float bh = data[3u * dimN + n];

        float left   = std::min<float>(std::max<float>(cx - bw * 0.5f, 0.0f), netW);
        float top    = std::min<float>(std::max<float>(cy - bh * 0.5f, 0.0f), netH);
        float right  = std::min<float>(std::max<float>(cx + bw * 0.5f, 0.0f), netW);
        float bottom = std::min<float>(std::max<float>(cy + bh * 0.5f, 0.0f), netH);

        float w = right - left;
        float h = bottom - top;
        if (w <= 1.0f || h <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = w;
        obj.height = h;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}