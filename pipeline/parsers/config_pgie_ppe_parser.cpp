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

extern "C" bool ParseConfigPgiePpe(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigPgiePpe);

extern "C" bool ParseConfigPgiePpe(
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
        fprintf(stderr, "[PPE] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape (with batch stripped): [10, 8400]
    // layer0.inferDims: numDims=2, d[0]=10, d[1]=8400
    // Format: 10 x N where 10 = 4 (cx,cy,w,h) + 6 classes
    // We need to transpose to [N, 10] logically.

    const unsigned int featureDim = layer0.inferDims.d[0]; // 10 = 4 + numClasses
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    // Only emit classes 1 (Vest) and 3 (helmet)
    const unsigned int allowedClasses[] = {1, 3};
    const unsigned int numAllowed = 2;

    for (unsigned int i = 0; i < numProposals; ++i) {
        // Data is laid out as [featureDim][numProposals], column-major per proposal
        // Row r, col i => data[r * numProposals + i]
        float cx = data[0 * numProposals + i];
        float cy = data[1 * numProposals + i];
        float bw = data[2 * numProposals + i];
        float bh = data[3 * numProposals + i];

        // Find best class among all classes
        float maxScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses && c + 4 < featureDim; ++c) {
            float score = data[(4 + c) * numProposals + i];
            if (score > maxScore) {
                maxScore = score;
                bestClass = c;
            }
        }

        // Filter: only keep allowed classes (1=Vest, 3=helmet)
        bool isAllowed = false;
        for (unsigned int a = 0; a < numAllowed; ++a) {
            if (bestClass == allowedClasses[a]) {
                isAllowed = true;
                break;
            }
        }
        if (!isAllowed) continue;

        // Threshold — scores are post-sigmoid probabilities, threshold directly
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.3f;
        if (maxScore < thr) continue;

        // Convert centre-xywh to top-left xywh in network-input coords
        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;

        // Clamp to network input dimensions
        left  = std::max<float>(0.0f, std::min<float>(left, netW));
        top   = std::max<float>(0.0f, std::min<float>(top, netH));
        float right  = std::max<float>(0.0f, std::min<float>(cx + bw * 0.5f, netW));
        float bottom = std::max<float>(0.0f, std::min<float>(cy + bh * 0.5f, netH));

        float finalW = right - left;
        float finalH = bottom - top;

        // Drop degenerate boxes
        if (finalW <= 1.0f || finalH <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = finalW;
        obj.height = finalH;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}