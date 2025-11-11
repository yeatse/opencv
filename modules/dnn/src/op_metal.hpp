// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_OP_METAL_HPP__
#define __OPENCV_DNN_OP_METAL_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

#include <unordered_map>

#ifdef HAVE_METAL
#include "metal/graph_builder.hpp"
#endif

namespace cv { namespace dnn {

constexpr bool haveMetal() {
#ifdef HAVE_METAL
    return true;
#else
    return false;
#endif
}

#ifdef HAVE_METAL

// Opaque pointer types for Metal backend objects
// These hide Objective-C types from C++ headers
typedef void* MTLDevicePtr;          // Opaque pointer to id<MTLDevice>
typedef void* MTLBufferPtr;          // Opaque pointer to id<MTLBuffer>
typedef void* MPSGraphTensorDataPtr; // Opaque pointer to MPSGraphTensorData*

class MetalBackendNode;
class MetalBackendWrapper;

class MetalNet
{
public:
    MetalNet();
    ~MetalNet();

    void addOutput(const std::string& name);
    void setUnconnectedNodes(Ptr<MetalBackendNode>& node);

    bool isInitialized();
    void init(Target targetId);
    void createGraph(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers, bool isAsync);

    std::vector<metal::MPSGraphTensorPtr> setInputs(const std::vector<cv::Mat>& inputs,
                                                     const std::vector<std::string>& names);

    void addBlobs(const std::vector<cv::Ptr<BackendWrapper>>& ptrs);

    void reset();

    // Graph builder for creating operations (similar to WebNN's ml::GraphBuilder)
    // Access via getBuilder() to ensure it's initialized
    metal::MetalGraphBuilder& getBuilder();

    // Device management
    MTLDevicePtr getDevice() const;

    // Opaque pointer to Objective-C implementation (MPSGraphNetImpl)
    metal::MPSGraphNetImplPtr impl;

    // Metal resources (managed by impl)
    std::unordered_map<std::string, cv::Ptr<MetalBackendWrapper>> allBlobs;

    bool hasNetOwner;
    bool isInit;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

private:
    metal::MetalGraphBuilder* builderPtr;  // Pointer to builder, initialized when impl is created
};

class MetalBackendNode : public BackendNode
{
public:
    MetalBackendNode(metal::MPSGraphTensorPtr tensor);

    std::string name;
    metal::MPSGraphTensorPtr tensor;
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
    void setDevice(MTLDevicePtr device);
    MTLDevicePtr getDevice() const { return metalDevice; }

    std::string name;
    Mat* host;                  // CPU memory (may point to hostClone if original was non-continuous)
    MTLBufferPtr metalBuffer;
    MPSGraphTensorDataPtr tensorData;
    MTLDevicePtr metalDevice;   // Shared from MetalNet
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
