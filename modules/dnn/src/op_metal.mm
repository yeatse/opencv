// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "op_metal.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Objective-C declarations must be at global scope
// Internal implementation class (uses MPSGraph)
@interface MPSGraphNetImpl : NSObject

@property (nonatomic, strong) MPSGraph* graph;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) MPSGraphExecutable* executable;

@property (nonatomic, strong) NSMutableDictionary<NSString*, MPSGraphTensorData*>* allBlobs;
@property (nonatomic, strong) NSMutableArray<NSString*>* inputNames;
@property (nonatomic, strong) NSMutableArray<NSString*>* outputNames;
@property (nonatomic, strong) NSMutableDictionary<NSString*, MPSGraphTensor*>* namedTensors;

@property (nonatomic, assign) BOOL isInitialized;
@property (nonatomic, assign) BOOL isCompiled;

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)addOutput:(NSString*)name;

@end

@implementation MPSGraphNetImpl

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        _commandQueue = [device newCommandQueue];
        _graph = [[MPSGraph alloc] init];
        _allBlobs = [NSMutableDictionary new];
        _inputNames = [NSMutableArray new];
        _outputNames = [NSMutableArray new];
        _namedTensors = [NSMutableDictionary new];
        _isInitialized = NO;
        _isCompiled = NO;
    }
    return self;
}

- (void)addOutput:(NSString*)name {
    [_outputNames addObject:name];
}

@end

// C++ implementation
namespace cv { namespace dnn {

// MetalNet implementation
MetalNet::MetalNet() : impl(nullptr), hasNetOwner(false), isInit(false) {
}

MetalNet::~MetalNet() {
    reset();
}

void MetalNet::init(Target targetId) {
    @autoreleasepool {
        if (!impl) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                CV_Error(Error::StsError, "Metal device not available");
            }
            MPSGraphNetImpl* netImpl = [[MPSGraphNetImpl alloc] initWithDevice:device];
            impl = (__bridge_retained void*)netImpl;
        }
        isInit = true;
    }
}

void MetalNet::createGraph(Target targetId) {
    // Phase 0: Stub implementation
    // Graph building will be implemented in Phase 1
}

void MetalNet::addOutput(const std::string& name) {
    outputNames.push_back(name);
    @autoreleasepool {
        if (impl) {
            MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
            [netImpl addOutput:[NSString stringWithUTF8String:name.c_str()]];
        }
    }
}

bool MetalNet::isInitialized() {
    return isInit;
}

std::vector<void*> MetalNet::setInputs(const std::vector<cv::Mat>& inputs,
                                        const std::vector<std::string>& names) {
    // Phase 0: Stub implementation
    // Input handling will be implemented in Phase 1
    std::vector<void*> result;
    return result;
}

void MetalNet::forward(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers, bool isAsync) {
    // Phase 0: Stub implementation
    // Forward pass will be implemented in Phase 1
    CV_Error(Error::StsNotImplemented,
             "Metal backend forward pass not implemented yet. "
             "Layers should fall back to CPU implementation.");
}

void MetalNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper>>& ptrs) {
    for (const auto& ptr : ptrs) {
        Ptr<MetalBackendWrapper> wrapper = ptr.dynamicCast<MetalBackendWrapper>();
        if (!wrapper.empty()) {
            allBlobs[wrapper->name] = wrapper;
        }
    }
}

void MetalNet::reset() {
    if (impl) {
        @autoreleasepool {
            // Transfer ownership and release
            (void)(__bridge_transfer MPSGraphNetImpl*)impl;
            impl = nullptr;
        }
    }
    allBlobs.clear();
    inputNames.clear();
    outputNames.clear();
    isInit = false;
}

// MetalBackendNode implementation
MetalBackendNode::MetalBackendNode(void* tensor_) : BackendNode(DNN_BACKEND_METAL) {
    tensor = tensor_;
}

// MetalBackendWrapper implementation
MetalBackendWrapper::MetalBackendWrapper(int targetId, Mat& m)
    : BackendWrapper(DNN_BACKEND_METAL, targetId), host(&m), metalBuffer(nullptr),
      tensorData(nullptr), size(m.total() * m.elemSize())
{
    // Get dimensions
    dimensions.resize(m.dims);
    for (int i = 0; i < m.dims; i++) {
        dimensions[i] = static_cast<int32_t>(m.size[i]);
    }
}

MetalBackendWrapper::~MetalBackendWrapper() {
    @autoreleasepool {
        if (metalBuffer) {
            // Transfer ownership and release
            (void)(__bridge_transfer id<MTLBuffer>)metalBuffer;
            metalBuffer = nullptr;
        }
        if (tensorData) {
            // Transfer ownership and release
            (void)(__bridge_transfer MPSGraphTensorData*)tensorData;
            tensorData = nullptr;
        }
    }
}

void MetalBackendWrapper::copyToHost() {
    // Phase 0: Stub implementation
    // Data transfer will be implemented in Phase 1
}

void MetalBackendWrapper::setHostDirty() {
    // Phase 0: Stub implementation
    // Dirty tracking will be implemented in Phase 1
}

void MetalBackendWrapper::allocateMetalBuffer() {
    // Phase 0: Stub implementation
    // Buffer allocation will be implemented in Phase 1
}

void MetalBackendWrapper::syncToDevice() {
    // Phase 0: Stub implementation
    // Data sync will be implemented in Phase 1
}

void MetalBackendWrapper::syncToHost() {
    // Phase 0: Stub implementation
    // Data sync will be implemented in Phase 1
}

// Forward function
void forwardMetal(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers,
                  Ptr<BackendNode>& node, bool isAsync) {
    CV_Assert(!node.empty());
    Ptr<MetalBackendNode> metalNode = node.dynamicCast<MetalBackendNode>();
    CV_Assert(!metalNode.empty());
    CV_Assert(!metalNode->net.empty());

    metalNode->net->forward(outBlobsWrappers, isAsync);
}

}}  // namespace cv::dnn

#else  // HAVE_METAL

namespace cv { namespace dnn {

// Stub implementation when Metal is not available
void forwardMetal(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers,
                  Ptr<BackendNode>& node, bool isAsync) {
    CV_Error(Error::StsNotImplemented, "Metal backend is not enabled in this OpenCV build");
}

}}  // namespace cv::dnn

#endif  // HAVE_METAL
