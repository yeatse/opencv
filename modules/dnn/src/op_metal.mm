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
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl) {
            CV_Error(Error::StsError, "Metal network not initialized");
        }

        // Mark as initialized - graph operations are added by layer initMetal() methods
        netImpl.isInitialized = YES;
    }
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
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl) {
            CV_Error(Error::StsError, "Metal network not initialized");
        }

        std::vector<void*> result;

        // Store input names
        inputNames.clear();
        for (const auto& name : names) {
            inputNames.push_back(name);
        }

        // Create placeholder tensors for inputs
        for (size_t i = 0; i < inputs.size() && i < names.size(); i++) {
            const cv::Mat& mat = inputs[i];
            const std::string& name = names[i];

            // Convert Mat dimensions to NSArray
            NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
            for (int d = 0; d < mat.dims; d++) {
                [shape addObject:@(mat.size[d])];
            }

            // Create placeholder tensor
            MPSGraphTensor* tensor = [netImpl.graph placeholderWithShape:shape
                                                                dataType:MPSDataTypeFloat32
                                                                    name:[NSString stringWithUTF8String:name.c_str()]];

            // Store in named tensors dictionary
            netImpl.namedTensors[[NSString stringWithUTF8String:name.c_str()]] = tensor;

            // Add to input names list
            [netImpl.inputNames addObject:[NSString stringWithUTF8String:name.c_str()]];

            // Return the tensor pointer
            result.push_back((__bridge void*)tensor);
        }

        return result;
    }
}

void MetalNet::forward(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers, bool isAsync) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !netImpl.isInitialized) {
            CV_Error(Error::StsError, "Metal network not initialized or no graph built");
        }

        // Prepare input feeds
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary new];

        // Feed input data
        for (const auto& blobEntry : allBlobs) {
            MetalBackendWrapper* wrapper = blobEntry.second.get();
            if (!wrapper) continue;

            // Sync data to device
            wrapper->syncToDevice();

            // Create MPSGraphTensorData from Metal buffer
            NSString* blobName = [NSString stringWithUTF8String:blobEntry.first.c_str()];
            MPSGraphTensor* tensor = netImpl.namedTensors[blobName];

            if (tensor && wrapper->metalBuffer) {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)wrapper->metalBuffer;

                // Convert dimensions to shape
                NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
                for (int32_t dim : wrapper->dimensions) {
                    [shape addObject:@(dim)];
                }

                MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:buffer
                                shape:shape
                             dataType:MPSDataTypeFloat32];

                feeds[tensor] = tensorData;
            }
        }

        // Compile graph if needed
        if (!netImpl.isCompiled && netImpl.outputNames.count > 0) {
            NSMutableArray<MPSGraphTensor*>* targetTensors = [NSMutableArray new];
            for (NSString* outName in netImpl.outputNames) {
                MPSGraphTensor* outTensor = netImpl.namedTensors[outName];
                if (outTensor) {
                    [targetTensors addObject:outTensor];
                }
            }

            if (targetTensors.count > 0) {
                MPSGraphCompilationDescriptor* compDesc = [MPSGraphCompilationDescriptor new];
                netImpl.executable = [netImpl.graph compileWithDevice:netImpl.device
                                                                feeds:feeds
                                                        targetTensors:targetTensors
                                                     targetOperations:nil
                                                compilationDescriptor:compDesc];
                netImpl.isCompiled = YES;
            }
        }

        // Execute graph
        if (netImpl.executable) {
            NSMutableArray<MPSGraphTensor*>* targetTensors = [NSMutableArray new];
            for (NSString* outName in netImpl.outputNames) {
                MPSGraphTensor* outTensor = netImpl.namedTensors[outName];
                if (outTensor) {
                    [targetTensors addObject:outTensor];
                }
            }

            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
                [netImpl.executable runWithMTLCommandQueue:netImpl.commandQueue
                                                      feeds:feeds
                                              targetTensors:targetTensors
                                           targetOperations:nil];

            // Copy results back to output wrappers
            for (size_t i = 0; i < outBlobsWrappers.size() && i < netImpl.outputNames.count; i++) {
                Ptr<MetalBackendWrapper> wrapper = outBlobsWrappers[i].dynamicCast<MetalBackendWrapper>();
                if (wrapper.empty()) continue;

                NSString* outName = netImpl.outputNames[i];
                MPSGraphTensor* outTensor = netImpl.namedTensors[outName];
                MPSGraphTensorData* resultData = results[outTensor];

                if (resultData && resultData.mpsndarray.buffer) {
                    // Store result buffer reference for later copying
                    id<MTLBuffer> resultBuffer = resultData.mpsndarray.buffer;

                    // Ensure output wrapper has Metal buffer
                    if (!wrapper->metalBuffer) {
                        wrapper->allocateMetalBuffer();
                    }

                    // Copy result data to output buffer
                    id<MTLBuffer> outBuffer = (__bridge id<MTLBuffer>)wrapper->metalBuffer;
                    size_t copySize = MIN(resultBuffer.length, outBuffer.length);
                    memcpy(outBuffer.contents, resultBuffer.contents, copySize);

                    // Sync back to host
                    wrapper->syncToHost();
                }
            }
        }
    }
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

// Graph building helper methods
void* MetalNet::getTensor(const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl) return nullptr;

        NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
        MPSGraphTensor* tensor = netImpl.namedTensors[nsName];
        return (__bridge void*)tensor;
    }
}

void MetalNet::addTensor(const std::string& name, void* tensor) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor) return;

        NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
        MPSGraphTensor* mpsTensor = (__bridge MPSGraphTensor*)tensor;
        netImpl.namedTensors[nsName] = mpsTensor;
    }
}

void* MetalNet::addReLU(void* inputTensor, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !inputTensor) return nullptr;

        MPSGraphTensor* input = (__bridge MPSGraphTensor*)inputTensor;
        MPSGraphTensor* output = [netImpl.graph reLUWithTensor:input
                                                          name:[NSString stringWithUTF8String:name.c_str()]];

        // Store named tensor
        addTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalNet::addAddition(void* tensor1, void* tensor2, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor1 || !tensor2) return nullptr;

        MPSGraphTensor* t1 = (__bridge MPSGraphTensor*)tensor1;
        MPSGraphTensor* t2 = (__bridge MPSGraphTensor*)tensor2;
        MPSGraphTensor* output = [netImpl.graph additionWithPrimaryTensor:t1
                                                          secondaryTensor:t2
                                                                    name:[NSString stringWithUTF8String:name.c_str()]];

        // Store named tensor
        addTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalNet::addConv2D(void* inputTensor, void* weightsTensor, void* biasTensor,
                           const std::vector<int>& strides, const std::vector<int>& paddings,
                           const std::vector<int>& dilations, int groups, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !inputTensor || !weightsTensor) return nullptr;

        MPSGraphTensor* input = (__bridge MPSGraphTensor*)inputTensor;
        MPSGraphTensor* weights = (__bridge MPSGraphTensor*)weightsTensor;

        // Create convolution descriptor
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:strides.size() > 1 ? strides[1] : 1
                                                                                                   strideInY:strides.size() > 0 ? strides[0] : 1
                                                                                             dilationRateInX:dilations.size() > 1 ? dilations[1] : 1
                                                                                             dilationRateInY:dilations.size() > 0 ? dilations[0] : 1
                                                                                                      groups:groups
                                                                                                paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                  dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                               weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        // Set explicit padding
        if (paddings.size() >= 4) {
            desc.paddingLeft = paddings[1];
            desc.paddingRight = paddings[3];
            desc.paddingTop = paddings[0];
            desc.paddingBottom = paddings[2];
        }

        MPSGraphTensor* output = [netImpl.graph convolution2DWithSourceTensor:input
                                                               weightsTensor:weights
                                                                  descriptor:desc
                                                                        name:[NSString stringWithUTF8String:name.c_str()]];

        // Add bias if present
        if (biasTensor) {
            MPSGraphTensor* bias = (__bridge MPSGraphTensor*)biasTensor;
            output = [netImpl.graph additionWithPrimaryTensor:output
                                              secondaryTensor:bias
                                                         name:[NSString stringWithFormat:@"%s_bias", name.c_str()]];
        }

        // Store named tensor
        addTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
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
    syncToHost();
}

void MetalBackendWrapper::setHostDirty() {
    // Phase 0: Stub implementation
    // Dirty tracking will be implemented in Phase 1
}

void MetalBackendWrapper::allocateMetalBuffer() {
    @autoreleasepool {
        if (!metalBuffer && size > 0) {
            // Get Metal device from somewhere - for now, create default device
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                CV_Error(Error::StsError, "Metal device not available");
            }

            // Allocate Metal buffer
            id<MTLBuffer> buffer = [device newBufferWithLength:size
                                                       options:MTLResourceStorageModeShared];
            if (!buffer) {
                CV_Error(Error::StsError, "Failed to allocate Metal buffer");
            }

            metalBuffer = (__bridge_retained void*)buffer;
        }
    }
}

void MetalBackendWrapper::syncToDevice() {
    @autoreleasepool {
        if (!host || !host->data) return;

        // Allocate Metal buffer if needed
        if (!metalBuffer) {
            allocateMetalBuffer();
        }

        // Copy data from host Mat to Metal buffer
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metalBuffer;
        memcpy(buffer.contents, host->data, size);

        // No explicit sync needed for MTLResourceStorageModeShared
    }
}

void MetalBackendWrapper::syncToHost() {
    @autoreleasepool {
        if (!host || !metalBuffer) return;

        // Copy data from Metal buffer to host Mat
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metalBuffer;
        memcpy(host->data, buffer.contents, size);
    }
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
