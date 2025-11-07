# OpenCV DNN Module - Quick Reference Guide

## Core File Locations

**Public Headers**: `/home/user/opencv/modules/dnn/include/opencv2/dnn/`
- `dnn.hpp` - Main Net/Layer API
- `layer.hpp` - Layer base classes  
- `dict.hpp` - Parameter storage
- `all_layers.hpp` - Layer declarations
- `layer.details.hpp` - Registration macros

**Core Implementation**: `/home/user/opencv/modules/dnn/src/`
- `net.cpp` (300 lines) - Net wrapper
- `net_impl.hpp` (295 lines) - Net internals structure
- `net_impl.cpp` (1000+ lines) - Main implementation
- `layer.cpp` (100 lines) - Layer base
- `layer_factory.cpp` (110 lines) - Factory pattern
- `dnn_read.cpp` (104 lines) - Model format detection

**Layer Implementations**: `/home/user/opencv/modules/dnn/src/layers/`
- 50+ layer types (convolution_layer.cpp, fully_connected_layer.cpp, etc.)
- cpu_kernels/ - SIMD optimized kernels

**Backend Implementations**:
- `cuda/` & `cuda4dnn/` - CUDA backend (5000+ lines)
- `ocl4dnn/` - OpenCL backend (2000+ lines)
- `vkcom/` - Vulkan backend (3000+ lines)
- `webnn/` - WebNN backend

**Model Importers**: `/home/user/opencv/modules/dnn/src/`
- `caffe/caffe_importer.cpp` - Caffe models
- `tensorflow/tf_importer.cpp` - TensorFlow models
- `onnx/onnx_importer.cpp` - ONNX models
- `darknet/darknet_importer.cpp` - Darknet/YOLO
- `torch/torch_importer.cpp` - Torch7 models
- `tflite/tflite_importer.cpp` - TensorFlow Lite

**Graph Optimization**:
- `graph_simplifier.cpp` (500+ lines) - Subgraph matching
- `net_impl_fuse.cpp` (500+ lines) - Layer fusion
- `layer_internals.hpp` - LayerPin, LayerData, DataLayer

## Key Classes (with Line Numbers)

```
Net (dnn.hpp:474)
├─ Purpose: Neural network graph
├─ Key Methods:
│  ├─ addLayer() - Add layer to graph
│  ├─ connect() - Connect layers
│  ├─ forward() - Run inference
│  ├─ setPreferableBackend/Target() - Set execution backend
│  ├─ readNetFromXxx() - Load models
│  └─ getParam/setParam() - Access weights
└─ Implementation: Net::Impl in net_impl.hpp

Layer (dnn.hpp:220)
├─ Purpose: Base class for all layers
├─ Key Methods:
│  ├─ forward() - Compute output
│  ├─ finalize() - Setup with actual shapes
│  ├─ getMemoryShapes() - Infer output shapes
│  ├─ initCUDA/Halide/VkCom/etc() - Backend init
│  └─ tryFuse() - Layer fusion support
└─ Properties: name, type, blobs, preferableTarget

LayerParams (dnn.hpp:145)
├─ Purpose: Layer initialization parameters
├─ Inherits: Dict (key-value storage)
├─ Properties:
│  ├─ blobs - Learned parameters (weights, biases)
│  ├─ name - Layer instance name
│  └─ type - Layer type identifier
└─ Used by: All layer constructors

LayerPin (layer_internals.hpp:12)
├─ Purpose: Identifies layer output
├─ Fields: lid (layer_id), oid (output_index)
└─ Used for: Graph connectivity

LayerData (layer_internals.hpp:43)
├─ Purpose: Internal layer representation
├─ Stores: Metadata, I/O blobs, backend nodes
├─ Properties:
│  ├─ outputBlobs - CPU Mat storage
│  ├─ outputBlobsWrappers - Backend wrappers
│  ├─ consumers - Downstream layers
│  └─ backendNodes - Per-backend computation
└─ Used by: Net::Impl for layer management

BackendNode (dnn.hpp:158)
├─ Purpose: Backend-specific computation
├─ Derived: HalideBackendNode, CUDABackendNode, etc.
└─ Created by: layer.initCUDA() / initHalide() / etc.

BackendWrapper (dnn.hpp:171)
├─ Purpose: Wrap Mat for backend memory
├─ Methods: copyToHost(), setHostDirty()
├─ Derived: OpenCLBackendWrapper, CUDABackendWrapper, etc.
└─ Used for: GPU ↔ CPU transfers
```

## Backends & Targets

```
Backends (dnn.hpp:70-88):
  DNN_BACKEND_OPENCV      ← CPU (default)
  DNN_BACKEND_CUDA        ← NVIDIA GPU
  DNN_BACKEND_HALIDE      ← Halide compiler
  DNN_BACKEND_VKCOM       ← Vulkan
  DNN_BACKEND_WEBNN       ← Browser/WebNN
  DNN_BACKEND_TIMVX       ← TIM-VX
  DNN_BACKEND_CANN        ← Huawei CANN
  DNN_BACKEND_INFERENCE_ENGINE ← Intel OpenVINO

Targets (dnn.hpp:95-108):
  DNN_TARGET_CPU          ← CPU 32-bit float
  DNN_TARGET_CPU_FP16     ← ARM 16-bit float
  DNN_TARGET_OPENCL       ← GPU via OpenCL
  DNN_TARGET_OPENCL_FP16  ← GPU FP16
  DNN_TARGET_CUDA         ← NVIDIA GPU 32-bit
  DNN_TARGET_CUDA_FP16    ← NVIDIA GPU 16-bit
  DNN_TARGET_VULKAN       ← Vulkan API
  DNN_TARGET_MYRIAD       ← Intel Movidius
  DNN_TARGET_HDDL         ← Intel HDDL
  DNN_TARGET_NPU          ← NPU accelerator
  DNN_TARGET_FPGA         ← FPGA
```

## Layer Types (65+ available)

**Geometry**: Reshape, Flatten, Expand, Slice, Split, Concat, Permute, Crop, Padding, Resize, Interp, CropAndResize

**Convolution**: Convolution (1D/2D/3D, grouped), Deconvolution, Pooling, ROIPooling, MaxUnpooling

**Connected**: InnerProduct (FC), MatMul, Gemm

**Normalization**: BatchNorm, LayerNorm, InstanceNorm, GroupNorm, LRN, MVN

**Activation**: ReLU, ReLU6, PReLU, ELU, Sigmoid, TanH, Swish, Mish, Gelu, Selu, Celu, HardSwish, HardSigmoid, Softmax, LogSoftmax

**Math**: Eltwise (Add/Sub/Mul/Div/Max/Min), NaryEltwise, Power, Exp, Log, Sqrt, Abs, Ceil, Floor, Round, Reciprocal, Sign, Shrink, Trig functions (Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh), Erf, Compare, Reduce

**Recurrent**: LSTM, GRU

**Detection**: DetectionOutput, Region, Proposal, NormalizeBBox, PriorBox

**Special**: Const, Gather, GatherElements, Scatter, ScatterND, Tile, TopK, Arg, Attention, Correlation, CumSum, Einsum, FlowWarp, Accum, ShuffleChannel, DepthToSpace, SpaceToDepth, DataAugmentation, Reorg, BlankLayer

**INT8 Quantized**: ConvolutionInt8, InnerProductInt8, PoolingInt8, EltwiseInt8, BatchNormInt8, ActivationInt8, etc.

## Model Formats

```
Format          | Files           | Importer
────────────────|─────────────────|──────────────────────────
Caffe           | .caffemodel     | caffe/caffe_importer.cpp
                | .prototxt       |
────────────────|─────────────────|──────────────────────────
TensorFlow      | .pb, .pbtxt     | tensorflow/tf_importer.cpp
────────────────|─────────────────|──────────────────────────
TensorFlow Lite | .tflite         | tflite/tflite_importer.cpp
────────────────|─────────────────|──────────────────────────
ONNX            | .onnx           | onnx/onnx_importer.cpp
────────────────|─────────────────|──────────────────────────
Darknet/YOLO    | .cfg, .weights  | darknet/darknet_importer.cpp
────────────────|─────────────────|──────────────────────────
Torch7          | .t7, .net       | torch/torch_importer.cpp
────────────────|─────────────────|──────────────────────────
OpenVINO IR     | .xml, .bin      | Model Optimizer
```

## API Usage Examples

```cpp
// Load model (auto-detect format)
cv::dnn::Net net = cv::dnn::readNet("model.onnx");

// Or specific format
net = cv::dnn::readNetFromCaffe("network.prototxt", "network.caffemodel");
net = cv::dnn::readNetFromTensorflow("frozen_graph.pb", "config.pbtxt");

// Set backend
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

// Prepare input
cv::Mat input = cv::imread("image.jpg");
cv::Mat blob = cv::dnn::blobFromImage(input, 1.0, cv::Size(224, 224));

// Set input
net.setInput(blob, "input_name", 1.0, cv::Scalar(104, 117, 123));

// Run inference
cv::Mat output = net.forward("output_name");

// Get multiple outputs
std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
std::vector<cv::Mat> outputs;
net.forward(outputs, outNames);

// Access layer parameters
cv::Ptr<cv::dnn::Layer> layer = net.getLayer(net.getLayerId("conv1"));
std::vector<cv::Mat> weights = layer->blobs;

// Performance profiling
std::vector<double> timings;
int64 totalTime = net.getPerfProfile(timings);
```

## Graph Structure

```
Net::Impl manages:
├─ layers (MapIdToLayerData)         - All layers in network
├─ layerNameToId (map)               - Name → ID mapping
├─ outputNameToId (map)              - Output → ID mapping
├─ netInputLayer (DataLayer)         - Pseudo-layer for inputs
├─ blobManager (BlobManager)         - Memory lifecycle management
├─ backendWrappers (map)             - GPU memory wrappers
├─ preferableBackend                 - Execution backend ID
├─ preferableTarget                  - Target device ID
├─ hasDynamicShapes                  - Dynamic shape support
└─ fusion, useWinograd               - Optimization flags

Per-Layer (LayerData):
├─ id, name, type                    - Identification
├─ dtype                             - Data type
├─ params                            - LayerParams
├─ inputBlobsId, consumers           - Connectivity
├─ outputBlobs, outputBlobsWrappers  - I/O storage
├─ internalBlobsWrappers             - Temp storage
├─ layerInstance                     - Layer object
├─ backendNodes                      - Backend-specific
└─ skip, flag                        - Optimization flags
```

## Execution Flow

```
readNet(file)
    ↓
[Model Import] ← Detects format, calls appropriate importer
    ↓
setInput() ← Set input data
    ↓
forward() or forward(outputs, names)
    ↓
[Shape Inference] ← getLayersShapesRecursively()
    ↓
[Memory Allocation] ← allocateLayers()
    ↓
[Backend Init] ← initBackend() (CUDA, Halide, Vulkan, etc.)
    ↓
[Layer Fusion] ← fuseLayers() (optional optimization)
    ↓
[Topological Traversal] ← forwardToLayer()
    │
    ├─ forwardLayer(layer0)
    │   ├─ Get input blobs
    │   ├─ GPU H2D (if needed)
    │   ├─ layer.forward()
    │   ├─ GPU D2H (if needed)
    │   └─ Manage memory
    │
    ├─ forwardLayer(layer1)
    │   └─ (same)
    │
    └─ ... repeat for all layers
    ↓
[Output Extraction] ← getBlob() or getBlobAsync()
    ↓
User receives Mat with results
```

## Memory Management

```
Blob Lifecycle:
1. User Input ← setInput(cv::Mat)
2. Layer Allocation ← allocateLayers() creates outputBlobs
3. Backend Wrapping ← BackendWrapper handles CPU↔GPU
4. Reference Counting ← BlobManager tracks consumers
5. Deallocation ← When last consumer processed
6. Output ← getBlob() transfers D2H and returns Mat

Memory Types:
├─ CPU: cv::Mat (OpenCV native)
├─ OpenCL: UMat (GPU via OpenCL)
├─ CUDA: Device pointers (CUDA arrays)
├─ Vulkan: VkBuffer (Vulkan device memory)
└─ Halide: halide_buffer_t (Halide runtime)

Workspace:
└─ CUDA: cuda4dnn::csl::Workspace for temp GPU memory (reused)
```

## Custom Layer Pattern

```cpp
// 1. Define class
class MyLayer : public cv::dnn::Layer {
    MyLayer(const cv::dnn::LayerParams& p) {
        setParamsFrom(p);
        // Parse custom parameters
    }
    
    static cv::Ptr<cv::dnn::Layer> create(
        const cv::dnn::LayerParams& params) {
        return cv::makePtr<MyLayer>(params);
    }
    
    void forward(cv::InputArrayOfArrays inputs,
                cv::OutputArrayOfArrays outputs,
                cv::OutputArrayOfArrays internals) override {
        // CPU implementation
        std::vector<cv::Mat> ins, outs;
        inputs.getMatVector(ins);
        outputs.getMatVector(outs);
        // Process: outs[i] = f(ins[i])
    }
    
    // Optional: CUDA support
    bool supportBackend(int backendId) override {
        return backendId == cv::dnn::DNN_BACKEND_OPENCV;
    }
};

// 2. Register (before using in model)
cv::dnn::LayerFactory::registerLayer("MyLayer", MyLayer::create);
// Or: CV_DNN_REGISTER_LAYER_CLASS(MyLayer, MyLayer);

// 3. Use in model (custom .cfg or manual construction)
// In importers:
net.addLayer("my_layer_0", "MyLayer", params);
```

## Layer Registration

```
Registration Flow:
1. initializeLayerFactory() called once at startup
2. All built-in layers registered via CV_DNN_REGISTER_LAYER_CLASS()
3. User can add custom layers via LayerFactory::registerLayer()

Factory Lookup:
1. LayerFactory::createLayerInstance("Conv", params)
2. Map lookup: std::map<String, std::vector<Constructor>>
3. Call top constructor in stack (LIFO)
4. Return Ptr<Layer> to new instance

Thread Safety:
├─ Mutex-protected (getLayerFactoryMutex())
├─ Singleton pattern (getLayerFactoryImpl_())
└─ Safe for multi-threaded registration
```

## Performance Optimization Techniques

```
Graph Level:
├─ Layer Fusion: Conv+ReLU → single operation
│  └─ Reduces memory bandwidth, kernel calls
├─ Graph Simplification: Pattern matching to remove redundant ops
│  └─ Subgraph matching with commutative op handling
├─ Memory Reuse: BlobManager deallocates early
│  └─ Reduces peak memory usage
└─ Lazy Evaluation: Only compute needed layers
   └─ Skip unneeded computation

Backend Level:
├─ Winograd: F(6x6) algorithm for 3x3 convolutions
│  └─ Fewer FLOPs with more memory
├─ Kernel Fusion: GPU kernels combined
│  └─ cuDNN, cuBLAS optimized paths
├─ Memory Pooling: Workspace reuse
│  └─ CUDA: cuda4dnn::csl::Workspace
└─ Async Transfers: GPU D2H background transfers
   └─ Overlap computation with memory transfer

Quantization:
├─ INT8 execution
├─ Per-channel scaling
└─ Reduced model size, lower latency, higher throughput
```

## Data Layout Support

```
Layout Modes (dnn.hpp:114-123):
├─ DNN_LAYOUT_NCHW    ← OpenCV default (4D)
├─ DNN_LAYOUT_NHWC    ← TensorFlow style (4D)
├─ DNN_LAYOUT_NCDHW   ← 5D NCHW (3D conv)
├─ DNN_LAYOUT_NDHWC   ← 5D NHWC
├─ DNN_LAYOUT_ND      ← N-dimensional generic
├─ DNN_LAYOUT_PLANAR  ← Special TFLite format
└─ DNN_LAYOUT_UNKNOWN ← Auto-detect

Shape Examples:
NCHW: [batch, channels, height, width]   = [1, 3, 224, 224]
NHWC: [batch, height, width, channels]   = [1, 224, 224, 3]
NCDHW: [batch, channels, depth, height, width] = [1, 3, 10, 224, 224]
```

## Testing & Benchmarks

Location: `/home/user/opencv/modules/dnn/test/` and `/perf/`

Key test files:
- `test_layers.cpp` - Layer functionality tests
- `test_tf_importer.cpp` - TensorFlow import tests
- `test_darknet_importer.cpp` - Darknet/YOLO tests
- `test_onnx_conformance.cpp` - ONNX operator coverage
- `test_backends.cpp` - Backend-specific tests
- `test_int8_layers.cpp` - Quantization tests

Performance benchmarks:
- `perf_precomp.hpp` - Benchmark infrastructure
- Layer-specific performance tests

