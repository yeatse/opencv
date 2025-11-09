// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_OP_METAL_HPP__
#define __OPENCV_DNN_OP_METAL_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

#include <unordered_map>

namespace cv { namespace dnn {

constexpr bool haveMetal() {
#ifdef HAVE_METAL
    return true;
#else
    return false;
#endif
}

#ifdef HAVE_METAL

class MetalBackendNode;
class MetalBackendWrapper;

class MetalNet
{
public:
    MetalNet();
    ~MetalNet();

    void addOutput(const std::string& name);

    bool isInitialized();
    void init(Target targetId);
    void createGraph(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers, bool isAsync);

    std::vector<void*> setInputs(const std::vector<cv::Mat>& inputs,
                                  const std::vector<std::string>& names);

    void addBlobs(const std::vector<cv::Ptr<BackendWrapper>>& ptrs);

    void reset();

    // Graph building helper methods (used by layer initMetal())
    void* addReLU(void* inputTensor, const std::string& name);
    void* addAddition(void* tensor1, void* tensor2, const std::string& name);
    void* addConv2D(void* inputTensor, void* weightsTensor, void* biasTensor,
                     const std::vector<int>& strides, const std::vector<int>& paddings,
                     const std::vector<int>& dilations, int groups, const std::string& name);
    void* getTensor(const std::string& name);
    void addTensor(const std::string& name, void* tensor);

    // Opaque pointer to Objective-C implementation (MPSGraphNetImpl)
    void* impl;

    // Metal resources (managed by impl)
    std::unordered_map<std::string, cv::Ptr<MetalBackendWrapper>> allBlobs;

    bool hasNetOwner;
    bool isInit;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};

class MetalBackendNode : public BackendNode
{
public:
    MetalBackendNode(void* tensor);  // MPSGraphTensor* (internal)

    std::string name;
    void* tensor;           // MPSGraphTensor* (opaque to C++, implementation detail)
    Ptr<MetalNet> net;      // Reference to parent graph
};

class MetalBackendWrapper : public BackendWrapper
{
public:
    MetalBackendWrapper(int targetId, Mat& m);
    ~MetalBackendWrapper();

    virtual void copyToHost() CV_OVERRIDE;
    virtual void setHostDirty() CV_OVERRIDE;

    std::string name;
    Mat* host;                  // CPU memory
    void* metalBuffer;          // id<MTLBuffer> (opaque)
    void* tensorData;           // MPSGraphTensorData* (opaque, internal)
    size_t size;
    std::vector<int32_t> dimensions;

    // Made public for forwardMetal() function access
    void allocateMetalBuffer();
    void syncToDevice();
    void syncToHost();
};

#endif  // HAVE_METAL

void forwardMetal(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers,
                  Ptr<BackendNode>& node, bool isAsync);

}}  // namespace cv::dnn

#endif  // __OPENCV_DNN_OP_METAL_HPP__
