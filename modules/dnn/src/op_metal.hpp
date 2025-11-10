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

// Graph builder class for Metal backend
// Encapsulates graph building operations similar to ml::GraphBuilder in WebNN
class MetalGraphBuilder
{
public:
    MetalGraphBuilder(void* graphImpl);  // MPSGraphNetImpl* (opaque)

    // Operation builders (return MPSGraphTensor* as void*)
    void* Relu(void* inputTensor, const std::string& name);
    void* Add(void* tensor1, void* tensor2, const std::string& name);
    void* Mul(void* tensor1, void* tensor2, const std::string& name);
    void* Max(void* tensor1, void* tensor2, const std::string& name);
    void* Min(void* tensor1, void* tensor2, const std::string& name);
    void* Div(void* tensor1, void* tensor2, const std::string& name);
    void* Conv2d(void* inputTensor, void* weightsTensor, void* biasTensor,
                 const std::vector<int>& strides, const std::vector<int>& paddings,
                 const std::vector<int>& dilations, int groups, const std::string& name);

    // Tensor management
    void* GetTensor(const std::string& name);
    void AddTensor(const std::string& name, void* tensor);

private:
    void* impl;  // MPSGraphNetImpl* (opaque pointer)
};

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

    // Graph builder for creating operations (similar to WebNN's ml::GraphBuilder)
    // Access via getBuilder() to ensure it's initialized
    MetalGraphBuilder& getBuilder();

    // Device management
    void* getDevice() const;  // Returns id<MTLDevice> as void*

    // Opaque pointer to Objective-C implementation (MPSGraphNetImpl)
    void* impl;

    // Metal resources (managed by impl)
    std::unordered_map<std::string, cv::Ptr<MetalBackendWrapper>> allBlobs;

    bool hasNetOwner;
    bool isInit;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

private:
    MetalGraphBuilder* builderPtr;  // Pointer to builder, initialized when impl is created
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

    // Device management - share device from MetalNet
    void setDevice(void* device);  // device is id<MTLDevice>
    void* getDevice() const { return metalDevice; }

    std::string name;
    Mat* host;                  // CPU memory (may point to hostClone if original was non-continuous)
    void* metalBuffer;          // id<MTLBuffer> (opaque)
    void* tensorData;           // MPSGraphTensorData* (opaque, internal)
    void* metalDevice;          // id<MTLDevice> (opaque) - shared from MetalNet
    size_t size;
    std::vector<int32_t> dimensions;

    // Made public for forwardMetal() function access
    void allocateMetalBuffer();
    void syncToDevice();
    void syncToHost();

private:
    Mat hostClone;              // Cloned Mat if original was non-continuous
};

#endif  // HAVE_METAL

void forwardMetal(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers,
                  Ptr<BackendNode>& node, bool isAsync);

}}  // namespace cv::dnn

#endif  // __OPENCV_DNN_OP_METAL_HPP__
