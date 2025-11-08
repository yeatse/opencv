# MPSGraph Backend Implementation Plan for OpenCV DNN

**Project:** GPU Acceleration on Apple Devices using MPSGraph
**Target Platforms:** macOS, iOS, visionOS, tvOS
**Reference Implementation:** WebNN Backend (`modules/dnn/src/op_webnn.*`)
**Created:** 2025-11-07

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Components Design](#3-core-components-design)
4. [Layer Support Matrix](#4-layer-support-matrix)
5. [Implementation Phases](#5-implementation-phases)
6. [Build System Integration](#6-build-system-integration)
7. [Memory Management Strategy](#7-memory-management-strategy)
8. [Graph Building Pipeline](#8-graph-building-pipeline)
9. [Testing Strategy](#9-testing-strategy)
10. [Performance Optimization](#10-performance-optimization)
11. [Migration from WebNN](#11-migration-from-webnn)
12. [Risk Assessment](#12-risk-assessment)
13. [Timeline & Milestones](#13-timeline--milestones)

---

## 1. EXECUTIVE SUMMARY

### Objective

Add **MPSGraph** backend to OpenCV DNN module to enable GPU-accelerated neural network inference on Apple devices (macOS, iOS, visionOS, tvOS) using Metal Performance Shaders Graph API.

### Why MPSGraph?

**Current State:**
- OpenCV DNN on Apple platforms: CPU-only or disabled OpenCL backend
- `CV_OCL4DNN = 0` on Apple platforms (see `modules/dnn/CMakeLists.txt`)
- No native GPU acceleration for Apple Neural Engine or Metal

**MPSGraph Advantages:**
- ✅ **Native Apple Support** - First-party framework, well-optimized
- ✅ **Cross-Device** - macOS, iOS, visionOS, tvOS with single codebase
- ✅ **Hardware Access** - GPU (Metal), Neural Engine, CPU
- ✅ **Graph Optimization** - Automatic kernel fusion, memory planning
- ✅ **Production Ready** - Used by Core ML, stable API since iOS 14/macOS 11
- ✅ **Framework Integration** - Seamless with existing Metal pipelines

**Target Performance:**
- 3-5x faster than CPU baseline on macOS (M-series)
- 5-10x faster than CPU on iOS devices
- Comparable or better than OpenCL on non-Apple platforms

### Deliverables

1. **Core Backend Implementation**
   - `modules/dnn/src/op_mpsgraph.hpp/mm` (Objective-C++)
   - `MPSGraphBackendNode`, `MPSGraphBackendWrapper`, `MPSGraphNet` classes

2. **Layer Support** (Phase 1: 15+ layers)
   - Convolution, Pooling, Activation, BatchNorm, Concat, etc.
   - Each layer implements `initMPSGraph()` method

3. **Build System**
   - CMake detection for Metal framework
   - Platform-specific compilation (Apple only)
   - Integration with existing OpenCV build

4. **Documentation & Tests**
   - Usage examples
   - Performance benchmarks
   - Unit tests for all supported layers

---

## 2. ARCHITECTURE OVERVIEW

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  OpenCV DNN with MPSGraph Backend                    │
└─────────────────────────────────────────────────────────────────────┘

    User Application (C++/Objective-C++)
            │
            │ cv::dnn::Net API
            ▼
    ┌───────────────────┐
    │   Net::Impl       │
    │                   │
    │ • layers map      │
    │ • backend mgmt    │
    └────────┬──────────┘
             │
             │ setPreferableBackend(DNN_BACKEND_MPSGRAPH)
             │ setPreferableTarget(DNN_TARGET_CPU/OPENCL)
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │   MPSGraph Backend (op_mpsgraph.mm)            │
    │                                                │
    │  ┌──────────────────────────────────┐         │
    │  │    MPSGraphNet                    │         │
    │  │  • MPSGraph* graph                │         │
    │  │  • id<MTLDevice> device           │         │
    │  │  • id<MTLCommandQueue> queue      │         │
    │  │  • MPSGraphExecutable* executable │         │
    │  └──────────────────────────────────┘         │
    │                                                │
    │  ┌──────────────────────────────────┐         │
    │  │  MPSGraphBackendNode              │         │
    │  │  • MPSGraphTensor* tensor         │         │
    │  │  • Ptr<MPSGraphNet> net           │         │
    │  └──────────────────────────────────┘         │
    │                                                │
    │  ┌──────────────────────────────────┐         │
    │  │  MPSGraphBackendWrapper           │         │
    │  │  • cv::Mat* hostMat               │         │
    │  │  • id<MTLBuffer> metalBuffer      │         │
    │  │  • MPSGraphTensorData* tensorData │         │
    │  └──────────────────────────────────┘         │
    └───────────────┬────────────────────────────────┘
                    │
                    │ MPSGraph Objective-C API
                    ▼
    ┌────────────────────────────────────────────────┐
    │        MPSGraph Framework                      │
    │                                                │
    │  • Graph building (operators)                  │
    │  • Compilation & optimization                  │
    │  • Execution on Metal                          │
    └───────────────┬────────────────────────────────┘
                    │
                    ▼
           Metal (GPU/Neural Engine/CPU)
```

### Design Principles

**1. Graph-Based Execution** (Like WebNN, Unlike CUDA/OpenCL)
- Build complete computational graph during initialization
- Single execution call per inference
- Runtime optimization by MPSGraph

**2. Hybrid Execution**
- Supported layers run on MPSGraph (GPU/Neural Engine)
- Unsupported layers fall back to OpenCV CPU
- Multiple MPSGraph instances if graph is split

**3. Zero-Copy Integration**
- Direct `cv::Mat` memory usage where possible
- Minimize CPU ↔ GPU transfers
- Shared memory buffers on unified memory systems (Apple Silicon)

**4. Apple-Native Development**
- Objective-C++ implementation (`.mm` files)
- Metal framework integration
- Follow Apple API conventions

---

## 3. CORE COMPONENTS DESIGN

### 3.1 File Structure

```
modules/dnn/src/
├── op_mpsgraph.hpp           # Public interface (C++ compatible header)
└── op_mpsgraph.mm            # Implementation (Objective-C++)

modules/dnn/src/layers/
├── convolution_layer.cpp     # Add initMPSGraph() method
├── pooling_layer.cpp         # Add initMPSGraph() method
├── ...                       # Add to all supported layers

cmake/
└── OpenCVDetectMPSGraph.cmake  # Build detection

platforms/apple/
└── mpsgraph_utils.mm         # Apple-specific utilities (optional)
```

### 3.2 MPSGraphNet Class

**File:** `modules/dnn/src/op_mpsgraph.mm`

```objc
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
- (NSArray<MPSGraphTensor*>*)setInputs:(NSArray<cv::Mat>*)inputs
                                 names:(NSArray<NSString*>*)names;
- (void)compileWithTarget:(int)targetId;
- (void)forward:(const std::vector<cv::Ptr<cv::dnn::BackendWrapper>>&)outBlobsWrappers
        isAsync:(BOOL)isAsync;

@end
```

**C++ Wrapper:**

```cpp
namespace cv { namespace dnn {

class MPSGraphNet {
public:
    MPSGraphNet();
    ~MPSGraphNet();

    void init(Target targetId);
    void createGraph(Target targetId);
    void addOutput(const std::string& name);

    std::vector<void*> setInputs(const std::vector<cv::Mat>& inputs,
                                  const std::vector<std::string>& names);

    void forward(const std::vector<Ptr<BackendWrapper>>& outBlobsWrappers,
                 bool isAsync);

    bool isInitialized() const;
    void reset();

    // Opaque pointer to Objective-C implementation
    void* impl;  // MPSGraphNetImpl*

    // Metal resources (managed by impl)
    std::unordered_map<std::string, cv::Ptr<MPSGraphBackendWrapper>> allBlobs;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};

}}  // namespace cv::dnn
```

### 3.3 MPSGraphBackendNode Class

```cpp
namespace cv { namespace dnn {

class MPSGraphBackendNode : public BackendNode {
public:
    MPSGraphBackendNode(void* tensor);  // MPSGraphTensor*

    std::string name;
    void* tensor;           // MPSGraphTensor* (opaque to C++)
    Ptr<MPSGraphNet> net;   // Reference to parent graph
};

}}  // namespace cv::dnn
```

### 3.4 MPSGraphBackendWrapper Class

```cpp
namespace cv { namespace dnn {

class MPSGraphBackendWrapper : public BackendWrapper {
public:
    MPSGraphBackendWrapper(int targetId, cv::Mat& m);
    ~MPSGraphBackendWrapper();

    virtual void copyToHost() CV_OVERRIDE;
    virtual void setHostDirty() CV_OVERRIDE;

    std::string name;
    cv::Mat* host;                  // CPU memory
    void* metalBuffer;              // id<MTLBuffer> (opaque)
    void* tensorData;               // MPSGraphTensorData* (opaque)
    size_t size;
    std::vector<int32_t> dimensions;

private:
    void allocateMetalBuffer();
    void syncToDevice();
    void syncToHost();
};

}}  // namespace cv::dnn
```

### 3.5 Layer Interface Extension

Each supported layer adds:

```cpp
// In convolution_layer.cpp
class ConvolutionLayerImpl : public ConvolutionLayer {
public:
    // ... existing methods ...

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        if (backendId == DNN_BACKEND_MPSGRAPH) {
            // Check if this specific conv config is supported
            return true;
        }
        return ConvolutionLayer::supportBackend(backendId);
    }

    virtual Ptr<BackendNode> initMPSGraph(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendNode>>& nodes) CV_OVERRIDE;
};
```

---

## 4. LAYER SUPPORT MATRIX

### Phase 1: Core Layers (Priority 1)

| OpenCV Layer | MPSGraph API | Implementation Complexity | Notes |
|--------------|--------------|---------------------------|-------|
| **Convolution** | `convolution2DWithSourceTensor:weightsTensor:descriptor:name:` | Medium | NCHW↔NHWC layout handling |
| **ReLU** | `reLUWithTensor:name:` | Low | Direct mapping |
| **Pooling** (Max/Avg) | `maxPooling2D/averagePooling2D` | Low | Descriptor for params |
| **BatchNorm** | `normalize` or custom graph | Medium | May need manual implementation |
| **Concat** | `concatTensors:dimension:name:` | Low | Axis mapping |
| **Eltwise** (Add/Mul) | `additionWithPrimaryTensor:secondaryTensor:` | Low | Element-wise ops |
| **InnerProduct** | `matrixMultiplicationPrimaryTensor:secondaryTensor:` | Low | Reshape + MatMul |
| **Softmax** | `softMaxWithTensor:axis:name:` | Low | Direct mapping |
| **Reshape** | `reshapeTensor:withShape:name:` | Low | Shape manipulation |
| **Permute** | `transposeTensor:permutation:name:` | Low | Axis permutation |

**Estimated Effort:** 2-3 weeks for Phase 1 layers

### Phase 2: Advanced Layers (Priority 2)

| OpenCV Layer | MPSGraph API | Implementation Complexity | Notes |
|--------------|--------------|---------------------------|-------|
| **Deconvolution** | `convolutionTranspose2D` | Medium | Transpose convolution |
| **DepthwiseConv** | `depthwiseConvolution2D` | Medium | Groups support |
| **PReLU** | Custom (min + mul) | Low | Parameterized ReLU |
| **Scale** | `multiplicationWithPrimaryTensor:secondaryTensor:` | Low | Scale + shift |
| **Slice** | `sliceTensor:dimension:start:length:name:` | Low | Tensor slicing |
| **Flatten** | `reshapeTensor:withShape:` | Low | Reshape variant |
| **LRN** | Custom normalization | Medium | Local response normalization |
| **Sigmoid/Tanh** | `sigmoidWithTensor:` / `tanhWithTensor:` | Low | Activation functions |

**Estimated Effort:** 2 weeks for Phase 2 layers

### Phase 3: Detection & Specialized (Priority 3)

| OpenCV Layer | MPSGraph API | Implementation Complexity | Notes |
|--------------|--------------|---------------------------|-------|
| **NMS** | `nonMaximumSuppressionWithBoxesTensor:...` | Medium | Built-in NMS (iOS 17+) |
| **ROIPooling** | Custom grid sampling | High | Manual implementation |
| **Proposal** | Custom graph | High | Region proposal network |
| **PriorBox** | Constant + arithmetic | Medium | Anchor generation |
| **DetectionOutput** | NMS + decode | High | SSD post-processing |
| **Region** (YOLO) | Custom graph + NMS | High | YOLO decoding |

**Estimated Effort:** 3-4 weeks for Phase 3 layers

### Unsupported Layers (CPU Fallback)

- **LSTM/GRU** - Complex; use MPSGraph RNN ops (future)
- **Custom layers** - User-defined operations
- **Quantized layers** - Needs quantization API integration

### Implementation Priority

```
Priority 1 (Week 1-3):  Core layers for basic CNNs (ResNet, VGG, MobileNet)
Priority 2 (Week 4-5):  Advanced layers for modern architectures
Priority 3 (Week 6-9):  Detection models (YOLO, SSD, Faster R-CNN)
Priority 4 (Week 10+):  Quantization, RNN, edge cases
```

---

## 5. IMPLEMENTATION PHASES

### Phase 0: Setup & Infrastructure (Week 1)

**Tasks:**
1. Create file structure
   - `op_mpsgraph.hpp` (C++ header)
   - `op_mpsgraph.mm` (Objective-C++ implementation)
   - `OpenCVDetectMPSGraph.cmake` (build detection)

2. Add backend enumeration
   ```cpp
   // In dnn.hpp
   enum Backend {
       DNN_BACKEND_DEFAULT = 0,
       DNN_BACKEND_OPENCV = 1,
       // ...
       DNN_BACKEND_WEBNN = 7,
       DNN_BACKEND_MPSGRAPH = 8,  // NEW
   };
   ```

3. Implement basic infrastructure
   - `MPSGraphNet` class (empty graph)
   - `MPSGraphBackendNode` class
   - `MPSGraphBackendWrapper` class
   - `initMPSGraphBackend()` in `Net::Impl`

4. Build system integration
   - Detect Metal framework availability
   - Compile `.mm` files on Apple platforms only
   - Link MetalPerformanceShadersGraph framework

**Deliverable:** Compilable but non-functional backend

**Validation:**
```cpp
Net net = readNetFromONNX("model.onnx");
net.setPreferableBackend(DNN_BACKEND_MPSGRAPH);  // Should not crash
// All layers fall back to CPU (no initMPSGraph yet)
```

---

### Phase 1: Core Layer Implementation (Week 2-3)

**Tasks:**

1. **Convolution Layer**
   - Implement `ConvolutionLayerImpl::initMPSGraph()`
   - Handle NCHW → NHWC layout conversion
   - Map stride, padding, dilation to `MPSGraphConvolution2DOpDescriptor`
   - Weight layout handling (OIHW)

2. **Pooling Layer**
   - Max pooling: `maxPooling2DWithSourceTensor:descriptor:name:`
   - Average pooling: `averagePooling2DWithSourceTensor:descriptor:name:`
   - Descriptor configuration

3. **Activation Layers**
   - ReLU: `reLUWithTensor:name:`
   - ReLU6: `clipByValueWithTensor:minValueTensor:maxValueTensor:name:`
   - Sigmoid: `sigmoidWithTensor:name:`
   - Tanh: `tanhWithTensor:name:`

4. **BatchNorm Layer**
   - Option 1: Fusion into previous conv (if adjacent)
   - Option 2: Manual implementation with `reductionMean/Variance`

5. **Element-wise Layers**
   - Add: `additionWithPrimaryTensor:secondaryTensor:name:`
   - Mul: `multiplicationWithPrimaryTensor:secondaryTensor:name:`
   - Broadcasting support

6. **Graph Building Logic**
   - Implement `Net::Impl::initMPSGraphBackend()`
   - Layer-by-layer graph construction
   - Input/output tensor management
   - Named tensor tracking

7. **Execution Pipeline**
   - Implement `MPSGraphNet::forward()`
   - Input feeding: `cv::Mat` → `MPSGraphTensorData`
   - Graph execution: `runWithMTLCommandQueue:feeds:targetTensors:`
   - Output retrieval: `MPSGraphTensorData` → `cv::Mat`

**Deliverable:** Working inference for simple models (MobileNetV2, ResNet18)

**Validation:**
```cpp
Net net = readNetFromONNX("mobilenet_v2.onnx");
net.setPreferableBackend(DNN_BACKEND_MPSGRAPH);
net.setPreferableTarget(DNN_TARGET_OPENCL);  // GPU
Mat blob = blobFromImage(img, 1.0/255, Size(224,224));
net.setInput(blob);
Mat output = net.forward();  // Should work on GPU
```

---

### Phase 2: Advanced Layers (Week 4-5)

**Tasks:**

1. **InnerProduct/Fully Connected**
   - `matrixMultiplicationPrimaryTensor:secondaryTensor:name:`
   - Reshape input if needed
   - Bias handling

2. **Concat/Split**
   - `concatTensors:dimension:name:`
   - Axis mapping (OpenCV vs MPSGraph)

3. **Reshape/Flatten/Permute**
   - `reshapeTensor:withShape:name:`
   - `transposeTensor:permutation:name:`
   - Shape calculation

4. **Softmax**
   - `softMaxWithTensor:axis:name:`

5. **Deconvolution**
   - `convolutionTranspose2DWithSourceTensor:weightsTensor:outputShape:descriptor:name:`

6. **PReLU/ELU/SELU**
   - Custom graph composition
   - Parameterized activations

**Deliverable:** Support for ResNet50, EfficientNet, Vision Transformers

**Validation:** Accuracy tests on ImageNet models

---

### Phase 3: Detection Layers (Week 6-9)

**Tasks:**

1. **NMS Implementation**
   - Use built-in `nonMaximumSuppressionWithBoxesTensor:...` (iOS 17+)
   - Fallback implementation for older OS

2. **Region/PriorBox**
   - Constant tensor generation
   - Grid creation with arithmetic ops

3. **DetectionOutput (SSD)**
   - Box decoding graph
   - NMS integration
   - Per-class processing

4. **Region (YOLO)**
   - YOLO-specific decoding
   - Sigmoid + exponential
   - NMS post-processing

**Deliverable:** Working YOLOv5, SSD, Faster R-CNN models

**Validation:** COCO mAP benchmarks

---

### Phase 4: Optimization & Polish (Week 10-12)

**Tasks:**

1. **Graph Compilation**
   - Use `MPSGraphExecutable` for repeated inference
   - Serialization/deserialization
   - Compilation cache

2. **Memory Optimization**
   - Shared memory on Apple Silicon
   - Buffer pooling
   - Minimize CPU↔GPU transfers

3. **Layout Optimization**
   - Single layout throughout pipeline (prefer NHWC on Metal)
   - Minimize transpose operations

4. **Quantization Support**
   - INT8 quantization with `quantizeTensor:scaleTensor:...`
   - Dequantization fusion
   - Per-channel quantization

5. **Performance Tuning**
   - Profile with Metal System Trace
   - Kernel fusion opportunities
   - Async execution

**Deliverable:** Production-ready backend with optimizations

**Validation:** Performance benchmarks vs CPU/OpenCL

---

## 6. BUILD SYSTEM INTEGRATION

### 6.1 CMake Detection

**File:** `cmake/OpenCVDetectMPSGraph.cmake`

```cmake
# Detect MPSGraph support on Apple platforms
if(APPLE)
    if(WITH_MPSGRAPH)
        # Check for Metal framework
        find_library(METAL_FRAMEWORK Metal)
        find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)

        if(METAL_FRAMEWORK AND MPSGRAPH_FRAMEWORK)
            # Check minimum OS version
            # MPSGraph requires iOS 14+, macOS 11+, tvOS 14+, visionOS 1+

            # Test compilation
            try_compile(VALID_MPSGRAPH
                "${OpenCV_BINARY_DIR}"
                SOURCES "${OpenCV_SOURCE_DIR}/cmake/checks/mpsgraph.mm"
                CMAKE_FLAGS
                    "-DLINK_LIBRARIES:STRING=${METAL_FRAMEWORK};${MPSGRAPH_FRAMEWORK}"
                OUTPUT_VARIABLE TRY_OUT
            )

            if(VALID_MPSGRAPH)
                set(HAVE_MPSGRAPH ON)
                message(STATUS "MPSGraph: YES")
                message(STATUS "  Metal Framework: ${METAL_FRAMEWORK}")
                message(STATUS "  MPSGraph Framework: ${MPSGRAPH_FRAMEWORK}")
            else()
                message(WARNING "MPSGraph compilation test failed")
                message(STATUS "${TRY_OUT}")
            endif()
        else()
            message(STATUS "MPSGraph: NO (frameworks not found)")
        endif()
    else()
        message(STATUS "MPSGraph: DISABLED (WITH_MPSGRAPH=OFF)")
    endif()
else()
    message(STATUS "MPSGraph: NO (Apple platforms only)")
endif()
```

**File:** `cmake/checks/mpsgraph.mm`

```objc
@import Metal;
@import MetalPerformanceShadersGraph;

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 1;

        MPSGraph* graph = [[MPSGraph alloc] init];
        if (!graph) return 1;

        MPSGraphTensor* a = [graph placeholderWithShape:@[@2,@3]
                                              dataType:MPSDataTypeFloat32
                                                  name:@"a"];
        MPSGraphTensor* b = [graph placeholderWithShape:@[@2,@3]
                                              dataType:MPSDataTypeFloat32
                                                  name:@"b"];
        MPSGraphTensor* c = [graph additionWithPrimaryTensor:a
                                             secondaryTensor:b
                                                        name:@"c"];

        return (a && b && c) ? 0 : 1;
    }
}
```

### 6.2 Module CMakeLists.txt Updates

**File:** `modules/dnn/CMakeLists.txt`

```cmake
# After existing backend configuration

# MPSGraph backend (Apple platforms)
if(HAVE_MPSGRAPH)
    list(APPEND dnn_srcs
        "${CMAKE_CURRENT_LIST_DIR}/src/op_mpsgraph.mm"
    )

    ocv_list_add_prefix(dnn_srcs "${CMAKE_CURRENT_LIST_DIR}/src/")

    # Add Metal frameworks
    ocv_target_link_libraries(${the_module}
        PRIVATE
            ${METAL_FRAMEWORK}
            ${MPSGRAPH_FRAMEWORK}
    )

    # Set Objective-C++ standard
    set_source_files_properties(
        "${CMAKE_CURRENT_LIST_DIR}/src/op_mpsgraph.mm"
        PROPERTIES
            COMPILE_FLAGS "-std=c++11 -fobjc-arc"
    )

    # Define macro for conditional compilation
    ocv_add_definitions(-DHAVE_MPSGRAPH)

    message(STATUS "DNN: MPSGraph backend enabled")
endif()
```

### 6.3 Platform-Specific Configuration

**File:** `platforms/apple/CMakeLists.txt` (if needed)

```cmake
# Apple-specific MPSGraph configuration
if(HAVE_MPSGRAPH)
    # Set deployment target
    if(IOS)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
    elseif(APPLE AND NOT IOS)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
    endif()

    # Enable ARC (Automatic Reference Counting)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")
endif()
```

### 6.4 Conditional Compilation

```cpp
// In op_mpsgraph.hpp
namespace cv { namespace dnn {

constexpr bool haveMPSGraph() {
#ifdef HAVE_MPSGRAPH
    return true;
#else
    return false;
#endif
}

}}  // namespace cv::dnn

// In op_mpsgraph.mm
#ifdef HAVE_MPSGRAPH

// Full implementation

#else

// Stub implementation
void forwardMPSGraph(...) {
    CV_Error(Error::StsNotImplemented,
             "MPSGraph is not enabled in this OpenCV build");
}

#endif  // HAVE_MPSGRAPH
```

---

## 7. MEMORY MANAGEMENT STRATEGY

### 7.1 Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Memory Management Architecture                   │
└─────────────────────────────────────────────────────────────────┘

CPU Memory (cv::Mat)           Metal Memory (MTLBuffer)
     │                                 │
     │                                 │
     ▼                                 ▼
┌──────────────┐              ┌──────────────────┐
│ LayerData    │              │ MPSGraphBackend  │
│              │              │ Wrapper          │
│ outputBlobs  │──────────────│                  │
│   [Mat]      │   wraps      │ • metalBuffer    │
│              │              │ • tensorData     │
└──────────────┘              │ • hostDirty      │
                              │ • deviceDirty    │
                              └──────────────────┘
```

### 7.2 Unified Memory Optimization (Apple Silicon)

On Apple Silicon (M1/M2/M3), CPU and GPU share unified memory:

```objc
// In MPSGraphBackendWrapper::allocateMetalBuffer()

id<MTLBuffer> buffer;

#if TARGET_OS_OSX && (defined(__arm64__) || defined(__aarch64__))
    // Apple Silicon - use shared memory (zero-copy)
    buffer = [device newBufferWithBytesNoCopy:host->data
                                       length:size
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
#else
    // Intel Mac or iOS - need explicit copy
    buffer = [device newBufferWithLength:size
                                 options:MTLResourceStorageModeShared];
    memcpy(buffer.contents, host->data, size);
#endif
```

### 7.3 Lazy Transfer Strategy

```cpp
class MPSGraphBackendWrapper : public BackendWrapper {
private:
    bool hostDirty;      // CPU data modified
    bool deviceDirty;    // GPU data modified

public:
    void copyToHost() CV_OVERRIDE {
        if (!deviceDirty) return;  // GPU data not modified

        // D2H transfer
        @autoreleasepool {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)metalBuffer;
            memcpy(host->data, buf.contents, size);
            deviceDirty = false;
        }
    }

    void setHostDirty() CV_OVERRIDE {
        hostDirty = true;
        deviceDirty = false;
    }

    void syncToDevice() {
        if (!hostDirty) return;  // CPU data not modified

        // H2D transfer
        @autoreleasepool {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)metalBuffer;
            memcpy(buf.contents, host->data, size);
            hostDirty = false;
        }
    }
};
```

### 7.4 Buffer Pooling

```objc
// In MPSGraphNet
@interface MPSGraphNetImpl : NSObject {
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* bufferPool;
}

- (id<MTLBuffer>)acquireBufferOfSize:(NSUInteger)size {
    NSNumber* key = @(size);
    id<MTLBuffer> buffer = bufferPool[key];

    if (!buffer) {
        buffer = [device newBufferWithLength:size
                                     options:MTLResourceStorageModeShared];
        bufferPool[key] = buffer;
    }

    return buffer;
}

@end
```

---

## 8. GRAPH BUILDING PIPELINE

### 8.1 Initialization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│       Net::Impl::initMPSGraphBackend()                          │
│       (Called during first forward())                           │
└─────────────────────────────────────────────────────────────────┘

Step 1: Create Metal Resources
───────────────────────────────

    device = MTLCreateSystemDefaultDevice()
    commandQueue = [device newCommandQueue]

    for each MPSGraphNet instance:
        graph = [[MPSGraph alloc] init]


Step 2: Assign Names to Blobs
──────────────────────────────

    for each layer in network:
        for each output blob:
            wrapper->name = layer.name + "." + index


Step 3: Build Graph Layer by Layer
───────────────────────────────────

    Ptr<MPSGraphNet> currentNet;

    for each layer in topological order:
        │
        ├─ Check if layer supports MPSGraph
        │  if (!layer->supportBackend(DNN_BACKEND_MPSGRAPH))
        │      ├─ Finalize current graph (if exists)
        │      ├─ Fall back to CPU for this layer
        │      └─ Create new MPSGraphNet for next MPSGraph layers
        │
        ├─ Create new MPSGraphNet if needed
        │  if (currentNet.empty() || needNewGraph)
        │      currentNet = Ptr<MPSGraphNet>(new MPSGraphNet())
        │
        ├─ Get input tensors from previous layers
        │  inputNodes = get_input_nodes(layer.inputBlobsId)
        │
        ├─ Call layer-specific MPSGraph initialization
        │  node = layer->initMPSGraph(inputBlobsWrappers, inputNodes)
        │  │
        │  │  Example for Convolution:
        │  │
        │  │  MPSGraphTensor* input = inputNodes[0]->tensor;
        │  │  MPSGraph* graph = net->graph;
        │  │
        │  │  // Create weight tensor
        │  │  MPSGraphTensor* weights = [graph constantWithData:...
        │  │                                              shape:...
        │  │                                           dataType:MPSDataTypeFloat32];
        │  │
        │  │  // Configure descriptor
        │  │  MPSGraphConvolution2DOpDescriptor* desc = ...;
        │  │  desc.strideInX = stride_w;
        │  │  desc.strideInY = stride_h;
        │  │  desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
        │  │
        │  │  // Create convolution operation
        │  │  MPSGraphTensor* output =
        │  │      [graph convolution2DWithSourceTensor:input
        │  │                             weightsTensor:weights
        │  │                                descriptor:desc
        │  │                                      name:@"conv"];
        │  │
        │  │  return Ptr<MPSGraphBackendNode>(
        │  │      new MPSGraphBackendNode(output));
        │
        ├─ Store node in layer's backend nodes
        │  layer.backendNodes[DNN_BACKEND_MPSGRAPH] = node
        │
        └─ Track outputs for this graph
           if (layer is output layer || has CPU consumers)
               currentNet->addOutput(node->name)


Step 4: Compile Graphs
──────────────────────

    for each MPSGraphNet instance:
        if (!net->isInitialized())
            net->compileWithTarget(target)
            │
            └─► Create MPSGraphExecutable
                (Optimized and cached for reuse)
```

### 8.2 Layout Handling

**Challenge:** OpenCV uses NCHW, MPSGraph prefers NHWC on Metal

**Strategy 1: Convert at boundaries** (Recommended for Phase 1)
```objc
// Input: NCHW → NHWC
MPSGraphTensor* nchw = [graph placeholderWithShape:@[@1,@3,@224,@224] ...];
MPSGraphTensor* nhwc = [graph transposeTensor:nchw
                                 permutation:@[@0,@2,@3,@1]  // NCHW→NHWC
                                        name:@"to_nhwc"];

// ... all operations in NHWC ...

// Output: NHWC → NCHW
MPSGraphTensor* output_nchw = [graph transposeTensor:output_nhwc
                                         permutation:@[@0,@3,@1,@2]  // NHWC→NCHW
                                                name:@"to_nchw"];
```

**Strategy 2: Keep NCHW** (Better compatibility)
```objc
// Use NCHW throughout (supported by MPSGraph)
descriptor.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
```

---

## 9. TESTING STRATEGY

### 9.1 Unit Tests

**File:** `modules/dnn/test/test_mpsgraph.cpp`

```cpp
TEST(DNN_MPSGraph, BasicInference) {
    Net net = readNetFromONNX("mobilenet_v2.onnx");
    net.setPreferableBackend(DNN_BACKEND_MPSGRAPH);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    Mat input(224, 224, CV_8UC3);
    randn(input, 127, 50);

    Mat blob = blobFromImage(input, 1.0/255, Size(224,224));
    net.setInput(blob);

    Mat output = net.forward();

    ASSERT_FALSE(output.empty());
    ASSERT_EQ(output.size[0], 1);
    ASSERT_EQ(output.size[1], 1000);
}

TEST(DNN_MPSGraph, AccuracyVsCPU) {
    Net netGPU = readNetFromONNX("resnet50.onnx");
    netGPU.setPreferableBackend(DNN_BACKEND_MPSGRAPH);

    Net netCPU = readNetFromONNX("resnet50.onnx");
    netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat input = imread("test_image.jpg");
    Mat blob = blobFromImage(input, 1.0/255, Size(224,224));

    netGPU.setInput(blob);
    Mat outputGPU = netGPU.forward();

    netCPU.setInput(blob);
    Mat outputCPU = netCPU.forward();

    double maxDiff = cv::norm(outputGPU, outputCPU, NORM_INF);
    EXPECT_LT(maxDiff, 1e-3);  // Numerical accuracy threshold
}
```

### 9.2 Layer-Specific Tests

```cpp
TEST(DNN_MPSGraph, ConvolutionLayer) {
    // Test various convolution configurations
    // - Different kernel sizes
    // - Different strides
    // - Different padding modes
    // - Groups (depthwise)
    // - Dilation
}

TEST(DNN_MPSGraph, PoolingLayer) {
    // Max pooling
    // Average pooling
    // Different kernel sizes
}

// ... tests for each layer type
```

### 9.3 Model Zoo Tests

```cpp
TEST(DNN_MPSGraph, ResNet50_ImageNet) {
    // Full ImageNet inference
    // Compare accuracy with ground truth
}

TEST(DNN_MPSGraph, MobileNetV2_Performance) {
    // Measure FPS
    // Compare with CPU baseline
}

TEST(DNN_MPSGraph, YOLOv5_COCO) {
    // Detection accuracy (mAP)
}
```

### 9.4 Performance Benchmarks

**File:** `modules/dnn/perf/perf_mpsgraph.cpp`

```cpp
PERF_TEST(DNN_MPSGraph, ResNet50_Throughput) {
    Net net = readNetFromONNX("resnet50.onnx");
    net.setPreferableBackend(DNN_BACKEND_MPSGRAPH);

    Mat input(224, 224, CV_8UC3);
    Mat blob = blobFromImage(input, 1.0/255, Size(224,224));

    TEST_CYCLE() {
        net.setInput(blob);
        Mat output = net.forward();
    }

    SANITY_CHECK_NOTHING();
}
```

### 9.5 Continuous Integration

```yaml
# .github/workflows/mpsgraph_ci.yml
name: MPSGraph Backend CI

on: [push, pull_request]

jobs:
  test-macos:
    runs-on: macos-14  # macOS Sonoma (M-series)
    steps:
      - uses: actions/checkout@v3

      - name: Build OpenCV with MPSGraph
        run: |
          mkdir build && cd build
          cmake -DWITH_MPSGRAPH=ON \
                -DBUILD_TESTS=ON \
                -DBUILD_PERF_TESTS=ON \
                ..
          make -j$(sysctl -n hw.ncpu)

      - name: Run Unit Tests
        run: |
          cd build
          ./bin/opencv_test_dnn --gtest_filter="*MPSGraph*"

      - name: Run Performance Tests
        run: |
          cd build
          ./bin/opencv_perf_dnn --gtest_filter="*MPSGraph*"
```

---

## 10. PERFORMANCE OPTIMIZATION

### 10.1 Graph Compilation

```objc
// In MPSGraphNet::compileWithTarget()

- (void)compileWithTarget:(int)targetId {
    // Build feed dictionary with shapes
    NSMutableDictionary* feedShapes = [NSMutableDictionary new];
    for (NSString* name in inputNames) {
        MPSGraphTensor* tensor = namedTensors[name];
        feedShapes[tensor] = tensor.shape;
    }

    // Create compilation descriptor
    MPSGraphCompilationDescriptor* compDesc = [MPSGraphCompilationDescriptor new];

    // Optimization options
    compDesc.optimizationLevel = MPSGraphOptimizationLevel1;

    // Target-specific settings
    if (targetId == DNN_TARGET_OPENCL) {
        // GPU preferred
        compDesc.optimizationProfile = MPSGraphOptimizationProfilePerformance;
    } else {
        // CPU or balanced
        compDesc.optimizationProfile = MPSGraphOptimizationProfilePowerEfficiency;
    }

    // Compile
    self.executable = [graph compileWithDevice:device
                                         feeds:feedShapes
                                 targetTensors:outputTensors
                              targetOperations:nil
                         compilationDescriptor:compDesc];

    self.isCompiled = YES;
}
```

### 10.2 Batch Size Optimization

```cpp
// Detect optimal batch size for hardware
int getOptimalBatchSize(id<MTLDevice> device) {
    // Apple Silicon: larger batches
    if ([device supportsFamily:MTLGPUFamilyApple7]) {
        return 8;
    }
    // Intel GPU: smaller batches
    else {
        return 4;
    }
}
```

### 10.3 Memory Footprint Reduction

```objc
// Release intermediate buffers after graph compilation
- (void)releaseIntermediateBuffers {
    // Clear temporary tensors
    [namedTensors removeAllObjects];

    // Keep only input/output buffers
    NSMutableDictionary* essentialBlobs = [NSMutableDictionary new];
    for (NSString* name in inputNames) {
        essentialBlobs[name] = allBlobs[name];
    }
    for (NSString* name in outputNames) {
        essentialBlobs[name] = allBlobs[name];
    }
    allBlobs = essentialBlobs;
}
```

### 10.4 Profiling Integration

```objc
// Enable Metal profiling
- (void)enableProfiling {
    MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor* captureDescriptor = [MTLCaptureDescriptor new];
    captureDescriptor.captureObject = device;
    captureDescriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    captureDescriptor.outputURL = [NSURL fileURLWithPath:@"mpsgraph_trace.gputrace"];

    NSError* error;
    [captureManager startCaptureWithDescriptor:captureDescriptor error:&error];
}
```

---

## 11. MIGRATION FROM WEBNN

### 11.1 Similarities (Reusable Patterns)

| Aspect | WebNN | MPSGraph | Reuse Strategy |
|--------|-------|----------|----------------|
| **Architecture** | Graph-based | Graph-based | Copy overall structure |
| **Initialization** | `initWebnnBackend()` | `initMPSGraphBackend()` | Adapt function names |
| **Node Type** | `WebnnBackendNode` | `MPSGraphBackendNode` | Copy class structure |
| **Wrapper Type** | `WebnnBackendWrapper` | `MPSGraphBackendWrapper` | Copy class structure |
| **Net Type** | `WebnnNet` | `MPSGraphNet` | Copy class structure |
| **Layer Interface** | `initWebnn()` | `initMPSGraph()` | Same pattern |

### 11.2 Key Differences

| Aspect | WebNN | MPSGraph | Migration Notes |
|--------|-------|----------|-----------------|
| **Language** | C++ | Objective-C++ | `.mm` files required |
| **Graph API** | `ml::GraphBuilder` | `MPSGraph` | Different API syntax |
| **Tensor Type** | `ml::Operand` | `MPSGraphTensor*` | Opaque pointer in C++ |
| **Data Type** | `ml::Input/Output` | `MPSGraphTensorData*` | Different wrapping |
| **Compilation** | `builder.Build()` | `graph compile...` | Different methods |
| **Execution** | `graph.Compute()` | `executable run...` or `graph run...` | Different API |

### 11.3 Code Migration Checklist

**From `op_webnn.hpp/cpp` to `op_mpsgraph.hpp/mm`:**

1. ✅ Copy file structure
2. ✅ Rename classes (Webnn → MPSGraph)
3. ✅ Change file extension (`.cpp` → `.mm` for implementation)
4. ✅ Replace WebNN API calls with MPSGraph Objective-C API
5. ✅ Update memory management (WebNN → Metal buffers)
6. ✅ Adapt graph building logic
7. ✅ Update execution flow
8. ✅ Add `@autoreleasepool` where needed
9. ✅ Handle Objective-C/C++ bridging

**Example: WebNN → MPSGraph**

```cpp
// WebNN (op_webnn.cpp)
ml::Operand BuildConstant(const ml::GraphBuilder& builder,
                          const std::vector<int32_t>& dimensions,
                          const void* value, size_t size,
                          ml::OperandType type) {
    ml::OperandDescriptor desc;
    desc.type = type;
    desc.dimensions = dimensions.data();
    desc.dimensionsCount = dimensions.size();
    ml::ArrayBufferView resource;
    resource.buffer = const_cast<void*>(value);
    resource.byteLength = size;
    return builder.Constant(&desc, &resource);
}
```

```objc
// MPSGraph (op_mpsgraph.mm)
MPSGraphTensor* BuildConstant(MPSGraph* graph,
                              NSArray<NSNumber*>* shape,
                              const void* value,
                              size_t size,
                              MPSDataType dataType) {
    @autoreleasepool {
        NSData* data = [NSData dataWithBytes:value length:size];
        return [graph constantWithData:data
                                 shape:shape
                              dataType:dataType];
    }
}
```

---

## 12. RISK ASSESSMENT

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **API Compatibility** | Medium | High | Test on multiple OS versions (iOS 14+, macOS 11+) |
| **Layout Complexity** | High | Medium | Start with single layout (NCHW), optimize later |
| **Unsupported Layers** | High | Medium | Implement hybrid execution (MPSGraph + CPU) |
| **Memory Leaks** | Medium | High | Use ARC, careful bridging, thorough testing |
| **Performance Regression** | Low | High | Benchmark against CPU baseline |
| **Build Complexity** | Medium | Low | Thorough CMake testing on all platforms |

### 12.2 Platform Risks

| Platform | Risk | Mitigation |
|----------|------|------------|
| **macOS Intel** | Lower performance vs Apple Silicon | Provide fallback to OpenCL if needed |
| **iOS < 14** | MPSGraph unavailable | Graceful degradation to CPU |
| **visionOS** | Limited testing hardware | Use simulator + beta testing |
| **Catalyst** | Potential compatibility issues | Separate testing for Catalyst builds |

### 12.3 Compatibility Risks

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **OpenCV API Changes** | Medium | Follow OpenCV development, update as needed |
| **Metal API Evolution** | Low | Apple maintains backward compatibility |
| **Third-party Models** | Medium | Extensive model zoo testing |
| **Custom Layers** | Low | Document custom layer implementation |

---

## 13. TIMELINE & MILESTONES

### 13.1 Gantt Chart

```
Week  1   2   3   4   5   6   7   8   9  10  11  12
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Phase 0: Setup & Infrastructure
      [████]

Phase 1: Core Layers
          [████████]

Phase 2: Advanced Layers
                  [████]

Phase 3: Detection
                      [████████]

Phase 4: Optimization
                              [████████]

Testing (Continuous)
      [════════════════════════════════════]

Documentation
                                      [████]
```

### 13.2 Milestones

**M1: Infrastructure Complete (Week 1)**
- ✅ File structure created
- ✅ Build system working
- ✅ Backend enumeration added
- ✅ Empty graph compiles

**M2: Basic Inference Working (Week 3)**
- ✅ Convolution, ReLU, Pooling implemented
- ✅ MobileNetV2 runs on MPSGraph
- ✅ Accuracy matches CPU

**M3: Advanced Models Working (Week 5)**
- ✅ All Phase 2 layers implemented
- ✅ ResNet50, EfficientNet working
- ✅ Performance tests passing

**M4: Detection Support (Week 9)**
- ✅ NMS implementation
- ✅ YOLOv5 working
- ✅ SSD working

**M5: Production Ready (Week 12)**
- ✅ All optimizations applied
- ✅ Documentation complete
- ✅ All tests passing
- ✅ Ready for PR

### 13.3 Resource Requirements

**Personnel:**
- 1 Senior Engineer (C++/Objective-C++/Metal): Full-time
- 1 ML Engineer (Model testing): Part-time (30%)
- 1 QA Engineer (Testing): Part-time (50% weeks 6-12)

**Hardware:**
- MacBook Pro M1/M2/M3 (development)
- iPhone 14+ (iOS testing)
- Intel Mac (compatibility testing)
- Apple Vision Pro (visionOS testing, optional)

**Software:**
- Xcode 14+
- CMake 3.20+
- Python 3.8+ (for testing scripts)
- Metal Debugger / Instruments

---

## 14. SUCCESS CRITERIA

### 14.1 Functional Requirements

- ✅ All Phase 1 layers working correctly
- ✅ At least 80% of OpenCV DNN layer types supported
- ✅ Hybrid execution (MPSGraph + CPU fallback)
- ✅ Works on macOS 11+, iOS 14+, visionOS 1+

### 14.2 Performance Requirements

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Speedup vs CPU** | ≥ 3x on Apple Silicon | ResNet50 inference |
| **Memory Overhead** | < 20% | Peak memory usage |
| **Compilation Time** | < 5s for typical models | First forward() |
| **FPS (MobileNetV2)** | ≥ 60 FPS on iPhone 14 | 224×224 input |
| **Accuracy** | < 1e-3 error vs CPU | All test models |

### 14.3 Code Quality Requirements

- ✅ All code follows OpenCV coding style
- ✅ 100% of public APIs documented
- ✅ ≥ 80% test coverage for new code
- ✅ No memory leaks (verified with Instruments)
- ✅ Passes all CI/CD checks

---

## 15. NEXT STEPS

### Immediate Actions (Week 1)

1. **Setup Development Environment**
   - Clone OpenCV repository
   - Install Xcode and Metal tools
   - Setup CMake build

2. **Create File Structure**
   - Create `op_mpsgraph.hpp`
   - Create `op_mpsgraph.mm`
   - Create `OpenCVDetectMPSGraph.cmake`

3. **Initial Implementation**
   - Implement `MPSGraphNet` skeleton
   - Implement `MPSGraphBackendNode`
   - Implement `MPSGraphBackendWrapper`
   - Add backend enumeration

4. **Build System**
   - Add CMake detection
   - Test compilation on macOS
   - Verify framework linking

### Code Review Checkpoints

- **Week 1:** Infrastructure review
- **Week 3:** Core layers review
- **Week 6:** Detection layers review
- **Week 10:** Performance review
- **Week 12:** Final review before PR

### Documentation Plan

1. **API Documentation** - Inline doxygen comments
2. **User Guide** - How to use MPSGraph backend
3. **Developer Guide** - How to add new layers
4. **Performance Guide** - Optimization tips
5. **Migration Guide** - From CPU/OpenCL to MPSGraph

---

## 16. REFERENCES

### Apple Documentation
- [MPSGraph Framework](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [MPSGraph Class Reference](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph)
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [WWDC 2024: MPSGraph Optimization](https://developer.apple.com/videos/play/wwdc2024/10218/)

### OpenCV References
- [DNN Module Documentation](https://docs.opencv.org/4.x/d6/d87/group__dnnLayerList.html)
- [WebNN Backend Implementation](modules/dnn/src/op_webnn.*)
- [CUDA Backend Implementation](modules/dnn/src/cuda4dnn/)
- [Build System Documentation](cmake/README.md)

### Related Work
- Core ML conversion tools
- PyTorch → Core ML pipeline
- TensorFlow Lite Metal delegate

---

## APPENDIX A: Layer Implementation Template

### Template: initMPSGraph() for New Layer

```cpp
// In layer_name.cpp

virtual Ptr<BackendNode> initMPSGraph(
    const std::vector<Ptr<BackendWrapper>>& inputs,
    const std::vector<Ptr<BackendNode>>& nodes) CV_OVERRIDE
{
    CV_Assert(!nodes.empty());

    // Get input tensor from previous layer
    Ptr<MPSGraphBackendNode> inputNode = nodes[0].dynamicCast<MPSGraphBackendNode>();
    CV_Assert(!inputNode.empty());

    @autoreleasepool {
        MPSGraphTensor* inputTensor = (__bridge MPSGraphTensor*)inputNode->tensor;
        MPSGraph* graph = (__bridge MPSGraph*)inputNode->net->getGraph();

        // TODO: Implement layer-specific logic
        // 1. Extract layer parameters
        // 2. Create MPSGraph operations
        // 3. Return new MPSGraphBackendNode

        MPSGraphTensor* outputTensor = /* ... */;

        void* outputTensorPtr = (__bridge_retained void*)outputTensor;
        Ptr<MPSGraphBackendNode> outputNode =
            Ptr<MPSGraphBackendNode>(new MPSGraphBackendNode(outputTensorPtr));
        outputNode->net = inputNode->net;

        return outputNode;
    }
}
```

---

## APPENDIX B: Build Command Examples

### macOS Build (Apple Silicon)

```bash
mkdir build && cd build

cmake -DWITH_MPSGRAPH=ON \
      -DBUILD_opencv_dnn=ON \
      -DBUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
      ..

make -j$(sysctl -n hw.ncpu)
```

### iOS Build

```bash
python platforms/ios/build_framework.py \
    --with_mpsgraph \
    --out ./ios_build

# Framework will be in ios_build/opencv2.framework
```

### Xcode Project Generation

```bash
cmake -G Xcode \
      -DWITH_MPSGRAPH=ON \
      -DBUILD_opencv_dnn=ON \
      ..

open OpenCV.xcodeproj
```

---

**END OF TECHNICAL PLAN**

*This plan is a living document and should be updated as implementation progresses.*
