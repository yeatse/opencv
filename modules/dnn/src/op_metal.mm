// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "op_metal.hpp"
#include "net_impl.hpp"

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

// Helper function to convert OpenCV Mat type to MPSDataType
static MPSDataType getMPSDataType(int matType) {
    switch (CV_MAT_DEPTH(matType)) {
        case CV_32F:
            return MPSDataTypeFloat32;
        case CV_16F:
            return MPSDataTypeFloat16;
        case CV_8U:
            return MPSDataTypeUInt8;
        case CV_8S:
            return MPSDataTypeInt8;
        case CV_16S:
            return MPSDataTypeInt16;
        case CV_32S:
            return MPSDataTypeInt32;
        default:
            CV_Error(Error::StsNotImplemented,
                     cv::format("Unsupported Mat type for Metal backend: %s",
                               typeToString(matType).c_str()));
            return MPSDataTypeFloat32; // Unreachable, but keeps compiler happy
    }
}

// MetalGraphBuilder implementation
MetalGraphBuilder::MetalGraphBuilder(void* graphImpl) : impl(graphImpl) {
}

void* MetalGraphBuilder::Relu(void* inputTensor, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !inputTensor) return nullptr;

        MPSGraphTensor* input = (__bridge MPSGraphTensor*)inputTensor;
        MPSGraphTensor* output = [netImpl.graph reLUWithTensor:input
                                                          name:[NSString stringWithUTF8String:name.c_str()]];

        // Store named tensor
        AddTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalGraphBuilder::Add(void* tensor1, void* tensor2, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor1 || !tensor2) return nullptr;

        MPSGraphTensor* t1 = (__bridge MPSGraphTensor*)tensor1;
        MPSGraphTensor* t2 = (__bridge MPSGraphTensor*)tensor2;
        MPSGraphTensor* output = [netImpl.graph additionWithPrimaryTensor:t1
                                                          secondaryTensor:t2
                                                                    name:[NSString stringWithUTF8String:name.c_str()]];

        // Store named tensor
        AddTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalGraphBuilder::Mul(void* tensor1, void* tensor2, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor1 || !tensor2) return nullptr;

        MPSGraphTensor* t1 = (__bridge MPSGraphTensor*)tensor1;
        MPSGraphTensor* t2 = (__bridge MPSGraphTensor*)tensor2;
        MPSGraphTensor* output = [netImpl.graph multiplicationWithPrimaryTensor:t1
                                                                secondaryTensor:t2
                                                                          name:[NSString stringWithUTF8String:name.c_str()]];

        // Store named tensor
        AddTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalGraphBuilder::Conv2d(void* inputTensor, void* weightsTensor, void* biasTensor,
                                  const std::vector<int>& strides, const std::vector<int>& paddings,
                                  const std::vector<int>& dilations, int groups, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !inputTensor || !weightsTensor) return nullptr;

        MPSGraphTensor* input = (__bridge MPSGraphTensor*)inputTensor;
        MPSGraphTensor* weights = (__bridge MPSGraphTensor*)weightsTensor;

        // Create convolution descriptor
        MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:strides[1]
                                                                                                    strideInY:strides[0]
                                                                                              dilationRateInX:dilations[1]
                                                                                              dilationRateInY:dilations[0]
                                                                                                       groups:groups
                                                                                                 paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                   dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingLeft = paddings[1];
        desc.paddingRight = paddings[1];
        desc.paddingTop = paddings[0];
        desc.paddingBottom = paddings[0];

        MPSGraphTensor* output = [netImpl.graph convolution2DWithSourceTensor:input
                                                                weightsTensor:weights
                                                                   descriptor:desc
                                                                         name:[NSString stringWithUTF8String:name.c_str()]];

        // Add bias if provided
        if (biasTensor) {
            MPSGraphTensor* bias = (__bridge MPSGraphTensor*)biasTensor;
            output = [netImpl.graph additionWithPrimaryTensor:output
                                              secondaryTensor:bias
                                                         name:[NSString stringWithUTF8String:(name + "_bias").c_str()]];
        }

        // Store named tensor
        AddTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalGraphBuilder::GetTensor(const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl) return nullptr;

        NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
        MPSGraphTensor* tensor = netImpl.namedTensors[nsName];
        return (__bridge void*)tensor;
    }
}

void MetalGraphBuilder::AddTensor(const std::string& name, void* tensor) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor) return;

        NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
        MPSGraphTensor* mpsTensor = (__bridge MPSGraphTensor*)tensor;
        netImpl.namedTensors[nsName] = mpsTensor;
    }
}

// MetalNet implementation
MetalNet::MetalNet() : impl(nullptr), hasNetOwner(false), isInit(false), builderPtr(nullptr) {
}

MetalNet::~MetalNet() {
    reset();
    if (builderPtr) {
        delete builderPtr;
        builderPtr = nullptr;
    }
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

            // Initialize the graph builder
            if (!builderPtr) {
                builderPtr = new MetalGraphBuilder(impl);
            }
        }
        isInit = true;
    }
}

MetalGraphBuilder& MetalNet::getBuilder() {
    if (!builderPtr) {
        CV_Error(Error::StsError, "Metal graph builder not initialized. Call init() first.");
    }
    return *builderPtr;
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

        // Ensure inputs and names match to catch wiring bugs
        CV_Assert(inputs.size() == names.size());

        std::vector<void*> result;

        // Clear input names (both C++ and Objective-C arrays)
        inputNames.clear();
        [netImpl.inputNames removeAllObjects];

        // Create placeholder tensors for inputs
        for (size_t i = 0; i < inputs.size(); i++) {
            const cv::Mat& mat = inputs[i];
            const std::string& name = names[i];

            // Store input names
            inputNames.push_back(name);

            // Convert Mat dimensions to NSArray
            NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
            for (int d = 0; d < mat.dims; d++) {
                [shape addObject:@(mat.size[d])];
            }

            // Derive MPSDataType from Mat type instead of hard-coding float32
            MPSDataType dataType = getMPSDataType(mat.type());

            // Create placeholder tensor
            MPSGraphTensor* tensor = [netImpl.graph placeholderWithShape:shape
                                                                dataType:dataType
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

        // Feed input data - only feed placeholders (input tensors), not outputs
        for (NSString* inputName in netImpl.inputNames) {
            std::string inputNameStr = [inputName UTF8String];
            auto it = allBlobs.find(inputNameStr);
            if (it == allBlobs.end()) {
                CV_LOG_WARNING(NULL, cv::format("Metal: Input '%s' not found in allBlobs", inputNameStr.c_str()));
                continue;
            }

            MetalBackendWrapper* wrapper = it->second.get();
            if (!wrapper) continue;

            // Sync data to device
            wrapper->syncToDevice();

            // Get the placeholder tensor
            MPSGraphTensor* tensor = netImpl.namedTensors[inputName];

            if (tensor && wrapper->metalBuffer && wrapper->host) {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)wrapper->metalBuffer;

                // Convert dimensions to shape
                NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
                for (int32_t dim : wrapper->dimensions) {
                    [shape addObject:@(dim)];
                }

                // Derive MPSDataType from wrapper's host Mat type
                MPSDataType dataType = getMPSDataType(wrapper->host->type());

                MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:buffer
                                shape:shape
                             dataType:dataType];

                feeds[tensor] = tensorData;
            }
        }

        // Compile graph if needed (using direct execution without precompilation for now)
        NSMutableArray<MPSGraphTensor*>* targetTensors = [NSMutableArray new];
        for (NSString* outName in netImpl.outputNames) {
            MPSGraphTensor* outTensor = netImpl.namedTensors[outName];
            if (outTensor) {
                [targetTensors addObject:outTensor];
            }
        }

        // Execute graph using runWithFeeds (managed execution)
        // This API manages device/queue internally and syncs results to CPU
        // Note: runWithFeeds is synchronous and blocks until completion
        if (targetTensors.count > 0) {
            MPSGraphTensorDataDictionary* results =
                [netImpl.graph runWithFeeds:feeds
                              targetTensors:targetTensors
                           targetOperations:nil];

            // Copy results back to output wrappers
            // Look up each wrapper's tensor by name instead of using global index
            for (const auto& wrapper : outBlobsWrappers) {
                Ptr<MetalBackendWrapper> metalWrapper = wrapper.dynamicCast<MetalBackendWrapper>();
                if (metalWrapper.empty()) continue;

                // Look up tensor by wrapper's name (not global index)
                NSString* outName = [NSString stringWithUTF8String:metalWrapper->name.c_str()];
                MPSGraphTensor* outTensor = netImpl.namedTensors[outName];
                if (!outTensor) {
                    CV_LOG_WARNING(NULL, cv::format("Metal: Output tensor '%s' not found", metalWrapper->name.c_str()));
                    continue;
                }

                MPSGraphTensorData* resultData = results[outTensor];
                if (resultData) {
                    // Get the data directly from MPSGraphTensorData's underlying buffer
                    MPSNDArray* resultArray = [resultData mpsndarray];

                    // Read data directly into host memory instead of going through Metal buffer
                    if (metalWrapper->host && metalWrapper->host->data) {
                        // Validate that host memory is contiguous to prevent corruption
                        CV_Assert(metalWrapper->host->isContinuous());
                        [resultArray readBytes:metalWrapper->host->data strideBytes:nil];
                    } else {
                        // Fallback: use Metal buffer as intermediate
                        if (!metalWrapper->metalBuffer) {
                            metalWrapper->allocateMetalBuffer();
                        }
                        id<MTLBuffer> outBuffer = (__bridge id<MTLBuffer>)metalWrapper->metalBuffer;
                        [resultArray readBytes:outBuffer.contents strideBytes:nil];
                        metalWrapper->syncToHost();
                    }
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

void* MetalNet::getDevice() const {
    @autoreleasepool {
        if (!impl) return nullptr;
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        return (__bridge void*)netImpl.device;
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
      tensorData(nullptr), metalDevice(nullptr), size(m.total() * m.elemSize())
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

void MetalBackendWrapper::setDevice(void* device) {
    metalDevice = device;
}

void MetalBackendWrapper::allocateMetalBuffer() {
    @autoreleasepool {
        if (!metalBuffer && size > 0) {
            // Use shared device from MetalNet if available, otherwise create default
            id<MTLDevice> device = nil;
            if (metalDevice) {
                device = (__bridge id<MTLDevice>)metalDevice;
            } else {
                device = MTLCreateSystemDefaultDevice();
            }

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

        // Validate that host memory is contiguous to prevent corruption
        CV_Assert(host->isContinuous());

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

        // Validate that host memory is contiguous to prevent corruption
        CV_Assert(host->isContinuous());

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

// Helper to add outputs when layer doesn't support Metal
void Net::Impl::addMetalOutputs(LayerData& ld)
{
    Ptr<MetalBackendNode> metalNode = ld.backendNodes[DNN_BACKEND_METAL].dynamicCast<MetalBackendNode>();
    if (!metalNode.empty() && !metalNode->net.empty())
    {
        metalNode->net->addOutput(ld.name);
    }
}

// Initialize Metal backend for the network
void Net::Impl::initMetalBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    CV_Assert_N(preferableBackend == DNN_BACKEND_METAL, haveMetal());

    Ptr<MetalNet> net;

    // First pass: Set wrapper names
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;
        if (ld.id == 0)
        {
            CV_Assert((netInputLayer->outNames.empty() && ld.outputBlobsWrappers.size() == 1) ||
                      (netInputLayer->outNames.size() == ld.outputBlobsWrappers.size()));
            for (size_t i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                Ptr<MetalBackendWrapper> wrapper = ld.outputBlobsWrappers[i].dynamicCast<MetalBackendWrapper>();
                if (wrapper.empty()) continue;
                std::string outputName = netInputLayer->outNames.empty() ? ld.name : netInputLayer->outNames[i];
                // Only append index suffix if we don't have named inputs
                outputName = (netInputLayer->outNames.empty() && ld.outputBlobsWrappers.size() > 1) ? (outputName + "." + std::to_string(i)) : outputName;
                wrapper->name = outputName;
            }
        }
        else
        {
            for (size_t i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                Ptr<MetalBackendWrapper> wrapper = ld.outputBlobsWrappers[i].dynamicCast<MetalBackendWrapper>();
                if (wrapper.empty()) continue;
                std::string outputName = ld.outputBlobsWrappers.size() > 1 ? (ld.name + "." + std::to_string(i)) : ld.name;
                wrapper->name = outputName;
            }
        }
    }

    // Second pass: Build Metal graphs
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;

        if (ld.id == 0 && ld.skip)
            continue;

        bool fused = ld.skip;
        Ptr<Layer> layer = ld.layerInstance;
        if (!fused && !layer->supportBackend(preferableBackend))
        {
            CV_LOG_WARNING(NULL, "Layer " + ld.type + " name " + ld.name + " is unsupported by Metal backend, falling back to CPU");

            addMetalOutputs(ld);
            net = Ptr<MetalNet>();
            layer->preferableTarget = DNN_TARGET_CPU;

            // Mark input nodes as unconnected
            for (size_t i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
                Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
                if (!inpNode.empty()) {
                    Ptr<MetalBackendNode> metalNode = inpNode.dynamicCast<MetalBackendNode>();
                    if (!metalNode.empty() && !metalNode->net.empty()) {
                        metalNode->net->addOutput(metalNode->name);
                    }
                }
            }
            continue;
        }
        ld.skip = true; // Initially skip all Metal supported layers

        // Collect input nodes
        std::vector<Ptr<BackendNode>> inputNodes;
        for (size_t i = 0; i < ld.inputBlobsId.size(); ++i)
        {
            if (inputNodes.size() == ld.inputBlobsId.size()) break;

            LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
            Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
            if (!inpNode.empty())
            {
                Ptr<MetalBackendNode> metalInpNode = inpNode.dynamicCast<MetalBackendNode>();
                if (!metalInpNode.empty() && !metalInpNode->net.empty())
                {
                    if (metalInpNode->net == net && !fused) {
                        inputNodes.push_back(inpNode);
                        continue;
                    }
                }
            }

            // Create new network if needed
            if (net.empty()) {
                net = Ptr<MetalNet>(new MetalNet());
                net->init(static_cast<Target>(preferableTarget));
            }

            if (!fused) {
                std::vector<std::string> inputNames;
                std::vector<cv::Mat> inputs;

                // Find this layer in the consumers
                auto curr_pos = inpLd.consumers.begin();
                auto compare = [&ld] (const LayerPin& lp) { return lp.lid == ld.id; };
                auto cons = curr_pos;
                while ((cons = std::find_if(curr_pos, inpLd.consumers.end(), compare)) != inpLd.consumers.end())
                {
                    int cons_inp = cons->oid;
                    Ptr<MetalBackendWrapper> inpWrapper = inpLd.outputBlobsWrappers[cons_inp].dynamicCast<MetalBackendWrapper>();
                    if (!inpWrapper.empty())
                    {
                        auto iter = std::find(inputNames.begin(), inputNames.end(), inpWrapper->name);
                        if (iter == inputNames.end()) {
                            inputNames.push_back(inpWrapper->name);
                            inputs.push_back(inpLd.outputBlobs[cons_inp]);
                        }
                    }
                    curr_pos = cons + 1;
                }

                auto inps = net->setInputs(inputs, inputNames);
                for (size_t i = 0; i < inps.size(); i++) {
                    MetalBackendNode* node = new MetalBackendNode(inps[i]);
                    node->net = net;
                    node->name = inputNames[i];  // Assign name so addOutput() can resolve it
                    inputNodes.emplace_back(Ptr<BackendNode>(node));
                }

                // Share device from network to input wrappers
                void* sharedDevice = net->getDevice();
                if (sharedDevice) {
                    for (size_t i = 0; i < inpLd.outputBlobsWrappers.size(); ++i) {
                        Ptr<MetalBackendWrapper> metalWrapper = inpLd.outputBlobsWrappers[i].dynamicCast<MetalBackendWrapper>();
                        if (!metalWrapper.empty()) {
                            metalWrapper->setDevice(sharedDevice);
                        }
                    }
                }

                // Register input wrappers in allBlobs so they can be found during forward pass
                net->addBlobs(inpLd.outputBlobsWrappers);
            }
        }

        Ptr<BackendNode> node;
        if (!net.empty())
        {
            if (fused)
            {
                bool inPlace = ld.inputBlobsId.size() == 1 && ld.outputBlobs.size() == 1 &&
                               ld.inputBlobs[0]->data == ld.outputBlobs[0].data;
                CV_Assert(inPlace);
                node = layers[ld.inputBlobsId[0].lid].backendNodes[preferableBackend];
                ld.inputBlobsWrappers = layers[ld.inputBlobsId[0].lid].inputBlobsWrappers;
            }
        }
        else {
            net = Ptr<MetalNet>(new MetalNet());
            net->init(static_cast<Target>(preferableTarget));
        }

        if (!fused)
        {
            CV_Assert(ld.inputBlobsId.size() == inputNodes.size());
            if (inputNodes.size())
            {
                // Call layer's initMetal to add operations to the graph
                try
                {
                    node = layer->initMetal(ld.inputBlobsWrappers, inputNodes);
                }
                catch (const cv::Exception&)
                {
                    CV_LOG_WARNING(NULL, "Layer " + ld.name + " initMetal failed, falling back to CPU");
                    layer->preferableTarget = DNN_TARGET_CPU;
                    addMetalOutputs(ld);
                    net = Ptr<MetalNet>();
                    continue;
                }
            }
        }

        if (!node.empty())
        {
            ld.backendNodes[DNN_BACKEND_METAL] = node;
            Ptr<MetalBackendNode> metalNode = node.dynamicCast<MetalBackendNode>();
            CV_Assert(!metalNode.empty());
            metalNode->name = ld.name;
            metalNode->net = net;

            // Add to Metal network's blob management (both inputs and outputs like WebNN)
            net->addBlobs(ld.inputBlobsWrappers);
            net->addBlobs(ld.outputBlobsWrappers);

            // Share device from network to all wrappers
            void* sharedDevice = net->getDevice();
            if (sharedDevice) {
                for (auto& wrapper : ld.outputBlobsWrappers) {
                    Ptr<MetalBackendWrapper> metalWrapper = wrapper.dynamicCast<MetalBackendWrapper>();
                    if (!metalWrapper.empty()) {
                        metalWrapper->setDevice(sharedDevice);
                    }
                }
            }

            // Mark outputs for graph execution (like WebNN does)
            addMetalOutputs(ld);

            // Enable Metal execution for this layer (clear skip flag)
            ld.skip = false;
        }
    }

    // Finalize all Metal networks
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;
        Ptr<BackendNode> node = ld.backendNodes[DNN_BACKEND_METAL];
        if (!node.empty())
        {
            Ptr<MetalBackendNode> metalNode = node.dynamicCast<MetalBackendNode>();
            if (!metalNode.empty() && !metalNode->net.empty())
            {
                metalNode->net->createGraph(static_cast<Target>(preferableTarget));
            }
        }
    }
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
