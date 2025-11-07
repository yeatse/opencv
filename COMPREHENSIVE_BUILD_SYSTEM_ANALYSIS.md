# OpenCV Build System - Comprehensive Analysis

**Focus Areas:** General Architecture, Apple Platforms (macOS/iOS/visionOS), DNN Module

**Generated:** 2025-11-07

---

## Table of Contents
1. [General Build System Architecture](#1-general-build-system-architecture)
2. [Apple Platform Build System](#2-apple-platform-build-system)
3. [DNN Module Build System](#3-dnn-module-build-system)
4. [Build Configuration](#4-build-configuration)
5. [Build Flow Diagrams](#5-build-flow-diagrams)
6. [File Reference Summary](#file-reference-summary)

---

## Executive Summary

OpenCV's build system is a sophisticated multi-platform CMake-based architecture featuring:

- **Two-Pass CMake Configuration** - Dependency resolution before target creation
- **Modular Plugin Architecture** - 23 core modules with dynamic dependency resolution
- **Multi-Backend Support** - CPU, CUDA, OpenCL, Vulkan, WebNN, OpenVINO
- **Platform Abstraction** - Unified build system for Windows, Linux, macOS, iOS, Android, embedded
- **SIMD Dispatch** - Runtime CPU feature detection with multiple optimization levels
- **Apple Framework Support** - Native framework bundles for macOS/iOS with universal binaries

**Key Statistics:**
- 60+ CMake configuration files
- 23 core modules with dependency graph
- 9 inference backends in DNN module
- 5+ Apple platform targets (macOS, iOS device/simulator, Catalyst, visionOS)
- 10+ SIMD optimization variants (SSE, AVX, NEON, RVV, etc.)

---

## 1. GENERAL BUILD SYSTEM ARCHITECTURE

### 1.1 Root CMakeLists.txt Structure

**File:** `/home/user/opencv/CMakeLists.txt` (2,150+ lines)

#### Key Sections:

**A. Header & CMake Policies (Lines 1-93)**
```
- Prevents in-source builds (lines 8-14)
- Loads minimum version requirements from cmake/OpenCVMinDepVersions.cmake (line 16)
- Configures CMake policies for:
  - CMP0042: MacOS @rpath support
  - CMP0051: Organization of source files
  - CMP0054: if() argument interpretation
  - CMP0067: C++ standard language support
  - CMP0146: CMake FindCUDA preference
```

**B. CMake Hooks System (Lines 96-108)**
- Allows external customization through hooks directory
- Registered via `ocv_cmake_reset_hooks()` and `ocv_cmake_hook()` macros
- Hook points: `CMAKE_INIT`, `PRE_CMAKE_BOOTSTRAP`, `POST_CMAKE_BOOTSTRAP`

**C. Project Bootstrap (Lines 111-142)**
```cmake
# Default build type setup
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build")

# Position independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ${ENABLE_PIC})

# Project declaration triggers platform detection
project(OpenCV CXX C)
```

**D. Platform System File Loading (Lines 144-149)**
```cmake
include("cmake/platforms/OpenCV-${CMAKE_SYSTEM_NAME}.cmake" OPTIONAL)
```
- Currently Darwin.cmake exists but is empty (files use iOS toolchain instead)
- Platform-specific configuration deferred to toolchain files

**E. Installation Paths (Lines 151-162)**
- Detects if prefix is initialized (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
- Default paths:
  - Windows: `${CMAKE_BINARY_DIR}/install`
  - Unix: `/usr/local`
  - Cross-compile: `${CMAKE_BINARY_DIR}/install`

### 1.2 CMake Core Files Organization

**Directory:** `/home/user/opencv/cmake/` (60+ files)

#### Critical Build Control Files:

**A. OpenCVModule.cmake (1,448 lines)**
- **Purpose:** Core module lifecycle management
- **Key Functions:**
  - `ocv_add_module(name [dependencies])` - Module registration (2-pass)
  - `ocv_glob_module_sources([EXCLUDE_CUDA] [EXCLUDE_OPENCL])` - Source discovery
  - `ocv_create_module([extra_deps])` - Module library creation
  - `ocv_define_module(name)` - Complete module definition in one call

**B. Dependency Resolution (Lines 500-695 in OpenCVModule.cmake)**
```
Function: __ocv_resolve_dependencies()
- Tracks 4 module states:
  1. OPENCV_MODULES_BUILD: modules to build
  2. OPENCV_MODULES_DISABLED_USER: explicitly disabled
  3. OPENCV_MODULES_DISABLED_AUTO: disabled due to dependencies
  4. OPENCV_MODULES_DISABLED_FORCE: cannot build in config

- Whitelist feature: BUILD_LIST parameter limits which modules build
- Propagates dependencies across module graph
- Detects circular dependencies and reports errors
- Sorts modules by dependency graph
```

**C. OpenCVUtils.cmake (71,402 bytes, ~1,500+ lines)**
- Generic CMake utility functions
- Common patterns for:
  - File globbing
  - Compiler detection
  - Variable updates
  - Message processing

**D. OpenCVCompilerOptimizations.cmake (41,987 bytes, ~1,000+ lines)**
- **Purpose:** SIMD dispatch and CPU feature detection
- **Key Variables:**
  - `CPU_ALL_OPTIMIZATIONS`: SSE/AVX/NEON/RVV/LASX/VSX variants
  - `CPU_BASELINE`: Default CPU features compiled unconditionally
  - `CPU_DISPATCH`: Optional dispatched implementations
  - `CPU_BASELINE_FINAL`, `CPU_DISPATCH_FINAL`: Resolved lists

**E. OpenCVCompilerOptions.cmake (25,883 bytes)**
- Sets compiler-specific flags
- Handles MSVC, GCC, Clang variations
- Defines `OPENCV_CXX_FLAGS`, `OPENCV_C_FLAGS`
- Processes optimization levels

**F. OpenCVFindProtobuf.cmake**
```cmake
# Lines 9: BUILD_PROTOBUF option controls:
# - ON (default): Build from 3rdparty/protobuf
# - OFF: Use system Protobuf via find_package

# Version handling for Protocol Buffers >= 22.0
# - Requires C++17+ (abseil-cpp dependency)
# - Falls back to system version if C++ standard too old
```

**G. Dependency Detection Files:**
- `OpenCVDetectCXXCompiler.cmake` - Compiler & architecture detection
- `OpenCVDetectCUDA.cmake` - CUDA toolkit detection
- `OpenCVDetectPython.cmake` - Python 2/3 detection for bindings
- `OpenCVDetectTBB.cmake` - Intel TBB detection
- `OpenCVDetectOpenCL.cmake` - OpenCL runtime detection
- `OpenCVDetectVulkan.cmake` - Vulkan API detection

### 1.3 Two-Pass CMake Configuration

**Mechanism Overview:**

The module registration system uses two distinct CMake passes to resolve dependencies:

**PASS 1: Module Discovery & Dependency Tracking (Lines 354-366 in OpenCVModule.cmake)**
```cmake
set(OPENCV_INITIAL_PASS ON)  # Enable first-pass mode
ocv_cmake_hook(PRE_MODULES_SCAN)
_add_modules_1(__main_paths __main_names)  # Scan main modules
_add_modules_1(__extra_paths __extra_names)  # Scan extra modules
__ocv_resolve_dependencies()  # Build dependency graph
set(OPENCV_INITIAL_PASS OFF)  # Disable first-pass
```

**Code Flow in ocv_add_module() (Lines 124-223):**
```cmake
if(OPENCV_INITIAL_PASS)
  # PASS 1: Collect module metadata
  option(BUILD_${the_module} "Include ${the_module} module" ${BUILD_${the_module}_INIT})
  set(OPENCV_MODULE_${the_module}_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}")
  ocv_add_dependencies(${the_module} ${ARGN})
  
  # Register module in global lists
  if(BUILD_${the_module})
    set(OPENCV_MODULES_BUILD ${OPENCV_MODULES_BUILD} "${the_module}")
  endif()
  
  return()  # Don't process further in pass 1
else()
  # PASS 2: Actual project/target creation
  project(${the_module})
  add_definitions(-D_USE_MATH_DEFINES)
endif()
```

**PASS 2: Module Creation & Target Building (Lines 395-398)**
```cmake
_add_modules_2(${OPENCV_MODULES_BUILD})  # Create all targets
# This calls add_subdirectory() for each module again, 
# but now with OPENCV_INITIAL_PASS=OFF
```

**Dependency Resolution Algorithm (Lines 500-695):**
```
1. Initialize module states based on BUILD_* options
2. Apply BUILD_LIST whitelist if specified
3. Propagate dependencies:
   - REQ_DEPS (required): Must have for module
   - OPT_DEPS (optional): Include if available
   - PRIVATE_REQ_DEPS: Internal only
   - PRIVATE_OPT_DEPS: Internal optional
4. Disable modules with unresolved required dependencies
5. Flatten and sort dependency graph
6. For world build: replace inter-module deps with opencv_world reference
```

**Key Advantage:** 
- Allows circular include detection before creating targets
- Enables conditional module disabling based on configuration
- Supports both header-only and library modules

### 1.4 Installation System

**File:** `/home/user/opencv/cmake/OpenCVInstallLayout.cmake`

**Installation Hierarchy:**
```
CMAKE_INSTALL_PREFIX/
├── bin/              # Executables, DLLs
├── lib/              # Libraries (.so, .a, .dylib)
├── lib/cmake/        # CMake config files
├── lib/pkgconfig/    # pkg-config .pc files
├── include/opencv2/  # Public headers
└── share/            # Docs, samples, data
```

**Component-based Installation:**
- `libs` - Runtime libraries
- `dev` - Headers and development files
- `docs` - Documentation
- `python` - Python bindings
- `samples` - Example code

---

## 2. APPLE PLATFORM BUILD SYSTEM

### 2.1 iOS Toolchain Architecture

**Location:** `/home/user/opencv/platforms/ios/cmake/Toolchains/`

#### Toolchain Files:

**A. common-ios-toolchain.cmake (8,266 bytes, 224 lines)**

**Platform Detection (Lines 64-106):**
```cmake
# Detects iOS vs visionOS architecture
if((IPHONEOS OR IPHONESIMULATOR) AND NOT DEFINED IOS_ARCH)
  message(FATAL_ERROR "iOS toolchain requires IOS_ARCH option")
endif()

if((IOS_ARCH MATCHES "^arm64") OR (VISIONOS_ARCH MATCHES "^arm64"))
  set(AARCH64 1)
elseif(IOS_ARCH MATCHES "^armv")
  set(ARM 1)
elseif((IOS_ARCH MATCHES "^x86_64") OR (VISIONOS_ARCH MATCHES "^x86_64"))
  set(X86_64 1)
endif()

# Set CMAKE_OSX_SYSROOT based on platform
if(IPHONEOS)
  set(CMAKE_OSX_SYSROOT "iphoneos")
elseif(IPHONESIMULATOR)
  set(CMAKE_OSX_SYSROOT "iphonesimulator")
elseif(VISIONOS)
  set(CMAKE_OSX_SYSROOT "xros")
elseif(VISIONSIMULATOR)
  set(CMAKE_OSX_SYSROOT "xrsimulator")
elseif(MAC_CATALYST)
  set(CMAKE_OSX_SYSROOT "macosx")
endif()
```

**Deployment Target Configuration (Lines 115-131):**
```cmake
# Environment variable: IPHONEOS_DEPLOYMENT_TARGET or XROS_DEPLOYMENT_TARGET
if(NOT DEFINED IPHONEOS_DEPLOYMENT_TARGET)
  if(NOT DEFINED ENV{IPHONEOS_DEPLOYMENT_TARGET})
    message(FATAL_ERROR "IPHONEOS_DEPLOYMENT_TARGET is not specified")
  endif()
  set(IPHONEOS_DEPLOYMENT_TARGET "$ENV{IPHONEOS_DEPLOYMENT_TARGET}")
endif()
```

**Xcode Build Wrapper (Lines 133-167):**
- For CMake < 3.25.0: Applies workaround for issue #13912 & #23156
- Wraps `xcodebuild` to inject compiler flags
- Generated from template: `xcodebuild_wrapper.in`
- Sets:
  - `CODE_SIGN_IDENTITY=''`
  - `CODE_SIGNING_REQUIRED=NO`
  - `ARCHS=${ARCH}` or uses universal architecture
  - `-sdk ${CMAKE_OSX_SYSROOT}`

**Compiler Configuration (Lines 193-221):**
```cmake
# Skip platform compiler checks (cross-compile)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_C_COMPILER_WORKS TRUE)

# Restrict library/include/package search paths
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Set data pointer sizes
if(AARCH64 OR X86_64)
  set(CMAKE_C_SIZEOF_DATA_PTR 8)
  set(CMAKE_CXX_SIZEOF_DATA_PTR 8)
else()
  set(CMAKE_C_SIZEOF_DATA_PTR 4)
  set(CMAKE_CXX_SIZEOF_DATA_PTR 4)
endif()
```

**Toolchain Config Persistence (Lines 1-62):**
- Saves cmake variables to `toolchain.config.cmake` for try_compile
- Prevents re-detection of platform during compiler checks
- Mechanism: detect → write config → restore for try_compile

**B. Platform-Specific Toolchain Wrappers:**

**Toolchain-iPhoneOS_Xcode.cmake (1-5 lines)**
```cmake
set(IPHONEOS TRUE)
include(${CMAKE_CURRENT_LIST_DIR}/common-ios-toolchain.cmake)
```

**Toolchain-iPhoneSimulator_Xcode.cmake (1-5 lines)**
```cmake
set(IPHONESIMULATOR TRUE)
include(${CMAKE_CURRENT_LIST_DIR}/common-ios-toolchain.cmake)
```

**Toolchain-XROS_Xcode.cmake & Toolchain-XRSimulator_Xcode.cmake**
- Similar pattern for visionOS

**Toolchain-Catalyst_Xcode.cmake**
```cmake
set(MAC_CATALYST TRUE)
include(${CMAKE_CURRENT_LIST_DIR}/common-ios-toolchain.cmake)
```

**C. iOS Platform CMake Module:**

**File:** `/home/user/opencv/platforms/ios/cmake/Modules/Platform/iOS.cmake` (8,212 bytes)
- Sets up iOS-specific compiler flags
- Configures deployment targets
- Defines framework-specific properties
- Handles bitcode embedding for iOS 7+

### 2.2 macOS Framework Build

**File:** `/home/user/opencv/platforms/osx/build_framework.py`

**Purpose:** Build universal OpenCV.framework for macOS

**Key Builder Methods:**

**A. Architecture Support:**
```python
# Default: x86_64
# Universal Binary: x86_64,arm64 (Big Sur+)
self.macos_archs = ["x86_64"]  # or ["x86_64", "arm64"]
```

**B. Build Process:**
```python
def _build(self, outdir):
    # 1. For each architecture:
    #    - Create build-{arch} subdirectory
    #    - Run xcodebuild with specific ARCHS
    # 2. Merge generated libraries into universal binary
    # 3. Create framework bundle structure
    
    for arch in self.macos_archs:
        main_build_dir = self.getBuildDir(main_working_dir, arch)
        cmake_flags = self._get_cmake_flags(arch)
        self.buildOne(arch, 'OSX', main_build_dir, cmake_flags)
```

**C. CMake Flags for Universal Builds:**
```python
cmake_flags = [
    "-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64",  # Universal binary
    "-DAPPLE_FRAMEWORK=ON",
    "-DBUILD_SHARED_LIBS=ON",  # or OFF for static
]
```

### 2.3 iOS Framework Build

**File:** `/home/user/opencv/platforms/ios/build_framework.py` (32,123 bytes)

**Build Targets:**
```python
# Each target group: (architectures, platform)
alltargets = [
    (["armv7"], "iPhoneOS"),      # 32-bit device
    (["armv7s"], "iPhoneOS"),     # A6/A6X chip
    (["arm64"], "iPhoneOS"),      # 64-bit device
    (["i386"], "iPhoneSimulator"), # 32-bit simulator (legacy)
    (["x86_64"], "iPhoneSimulator"), # 64-bit simulator
]
```

**Dynamic vs Static Frameworks:**
```python
if not self.dynamic:
    # Static framework: Merge all .a files into single binary
    self.mergeLibs(main_build_dir)
else:
    # Dynamic framework: Use Xcode to build dylib
    # Requires iOS 8+ for App Store distribution
```

**Bitcode Support (Lines 115-117):**
```python
if xcode_ver >= 7 and target[1] == 'iPhoneOS' and self.embed_bitcode:
    cmake_flags.append("-DCMAKE_C_FLAGS=-fembed-bitcode")
    cmake_flags.append("-DCMAKE_CXX_FLAGS=-fembed-bitcode")
```

**Catalyst Build (Lines 118-142):**
```python
if xcode_ver >= 7 and target[1] == 'Catalyst':
    sdk_path = check_output(["xcodebuild", "-version", "-sdk", "macosx", "Path"])
    c_flags = [
        "-target x86_64-apple-ios14.0-macabi",  # Catalyst target
        "-isysroot %s" % sdk_path,
        "-iframework %s/System/iOSSupport/System/Library/Frameworks" % sdk_path,
        "-isystem %s/System/iOSSupport/usr/include" % sdk_path,
    ]
    cmake_flags.append("-DIOS=1")  # Use iOS codebase
    cmake_flags.append("-DMAC_CATALYST=1")
```

### 2.4 Info.plist Generation

**File:** `/home/user/opencv/cmake/OpenCVGenInfoPlist.cmake` (28 lines)

**Configuration (Lines 1-27):**
```cmake
set(OPENCV_APPLE_BUNDLE_NAME "OpenCV")
set(OPENCV_APPLE_BUNDLE_ID "org.opencv")

if(IOS)
  if(MAC_CATALYST)
    configure_file("${OpenCV_SOURCE_DIR}/platforms/ios/Info.plist.in"
                  "${CMAKE_BINARY_DIR}/osx/Info.plist")
  elseif(APPLE_FRAMEWORK AND DYNAMIC_PLIST)
    configure_file("${OpenCV_SOURCE_DIR}/platforms/ios/Info.Dynamic.plist.in"
                   "${CMAKE_BINARY_DIR}/ios/Info.plist")
  else()
    configure_file("${OpenCV_SOURCE_DIR}/platforms/ios/Info.plist.in"
                   "${CMAKE_BINARY_DIR}/ios/Info.plist")
  endif()
elseif(XROS)
  # visionOS Info.plist handling
elseif(APPLE)
  # macOS Info.plist handling
endif()
```

**Plist Templates:**
- `/home/user/opencv/platforms/ios/Info.plist.in` - Static framework
- `/home/user/opencv/platforms/ios/Info.Dynamic.plist.in` - Dynamic framework
- `/home/user/opencv/platforms/osx/Info.plist.in` - macOS framework

**Variables Substituted:**
- `${OPENCV_APPLE_BUNDLE_NAME}` → "OpenCV"
- `${OPENCV_APPLE_BUNDLE_ID}` → "org.opencv"
- Version information from cmake/OpenCVVersion.cmake

### 2.5 Framework Structure

**Installation Layout:**
```
OpenCV.framework/
├── OpenCV                    # Symlink to current version
├── Headers/                  # Public headers (symlink)
├── Modules/                  # Module map (for Xcode IDE)
├── Resources/                # Info.plist, PrivacyInfo.xcprivacy
├── Versions/
│   ├── A/
│   │   ├── OpenCV            # Actual binary (static or dylib)
│   │   ├── Headers/          # Real header files
│   │   ├── Modules/
│   │   └── Resources/
│   └── Current -> A           # Version symlink
└── Current -> Versions/Current  # Convenience symlink
```

**Binary Types:**
- **Static:** `/path/to/OpenCV.framework/OpenCV` (contains all object files)
- **Dynamic:** `/path/to/OpenCV.framework/OpenCV` (dylib with dynamic linking)

---

## 3. DNN MODULE BUILD SYSTEM

### 3.1 DNN Module CMakeLists.txt

**File:** `/home/user/opencv/modules/dnn/CMakeLists.txt` (340 lines)

**A. Module Declaration (Line 14):**
```cmake
ocv_add_module(dnn opencv_core opencv_imgproc WRAP python java objc js)
```

**B. Module Configuration (Lines 20-59)**

**OpenCL Backend (Lines 20-24):**
```cmake
ocv_option(OPENCV_DNN_OPENCL "Build with OpenCL support" 
    HAVE_OPENCL AND NOT APPLE)  # Disabled for Apple platforms

if(OPENCV_DNN_OPENCL AND HAVE_OPENCL)
  ocv_target_compile_definitions(${the_module} PRIVATE "CV_OCL4DNN=1")
endif()
```

**CUDA Backend (Lines 38-56):**
```cmake
ocv_option(OPENCV_DNN_CUDA "Build with CUDA support"
    HAVE_CUDA AND HAVE_CUBLAS AND HAVE_CUDNN)

if(OPENCV_DNN_CUDA)
  if(HAVE_CUDA AND HAVE_CUBLAS AND HAVE_CUDNN)
    ocv_target_compile_definitions(${the_module} PRIVATE "CV_CUDA4DNN=1")
  else()
    message(SEND_ERROR "DNN: CUDA backend requires CUDA Toolkit. 
                        Please resolve dependency or disable OPENCV_DNN_CUDA=OFF")
  endif()
endif()
```

**WebNN Backend (Lines 26-28):**
```cmake
if(WITH_WEBNN AND HAVE_WEBNN)
  ocv_target_compile_definitions(${the_module} PRIVATE "HAVE_WEBNN=1")
endif()
```

**TIM-VX Accelerator (Lines 30-32):**
```cmake
if(HAVE_TIMVX)
  ocv_target_compile_definitions(${the_module} PRIVATE "HAVE_TIMVX=1")
endif()
```

**CANN (Huawei Atlas) (Lines 34-36):**
```cmake
if(HAVE_CANN)
  ocv_target_compile_definitions(${the_module} PRIVATE "HAVE_CANN=1")
endif()
```

**C. SIMD Dispatch Configuration (Lines 7-12)**

**Dispatched Source Files:**
```cmake
# Format: ocv_add_dispatched_file(src_name [SIMD_VARIANTS])
ocv_add_dispatched_file_force_all("layers/layers_common" 
    AVX AVX2 AVX512_SKX RVV LASX NEON)
ocv_add_dispatched_file_force_all("int8layers/layers_common" 
    AVX2 AVX512_SKX RVV LASX NEON)
ocv_add_dispatched_file_force_all("layers/cpu_kernels/conv_block" 
    AVX AVX2 NEON NEON_FP16)
ocv_add_dispatched_file("layers/cpu_kernels/conv_winograd_f63" 
    AVX AVX2 NEON NEON_FP16)
```

**Key Variants:**
- `AVX`, `AVX2`, `AVX512_SKX` - x86/x86_64
- `NEON`, `NEON_FP16`, `NEON_DOTPROD` - ARM
- `RVV` - RISC-V Vector
- `LASX`, `LSX` - Loongson MIPS
- Compiles separate `.avx2.cpp`, `.neon.cpp` files with specific flags

**D. Protobuf Configuration (Lines 113-159)**

**Protobuf Generation (Lines 116-139):**
```cmake
if(PROTOBUF_UPDATE_FILES)
  # Regenerate from .proto sources
  file(GLOB proto_files 
    "${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/*.proto"
    "${CMAKE_CURRENT_LIST_DIR}/src/caffe/opencv-caffe.proto"
    "${CMAKE_CURRENT_LIST_DIR}/src/onnx/opencv-onnx.proto")
  
  if(CMAKE_VERSION VERSION_LESS "3.13.0")
    protobuf_generate_cpp(fw_srcs fw_hdrs ${proto_files})
  else()
    protobuf_generate(
      APPEND_PATH
      LANGUAGE cpp
      OUT_VAR fw_srcs
      PROTOC_EXE ${Protobuf_PROTOC_EXECUTABLE}
      PROTOS ${proto_files})
  endif()
else()
  # Use pre-generated files from misc/
  file(GLOB fw_srcs 
    "${CMAKE_CURRENT_LIST_DIR}/misc/tensorflow/*.cc"
    "${CMAKE_CURRENT_LIST_DIR}/misc/caffe/opencv-caffe.pb.cc"
    "${CMAKE_CURRENT_LIST_DIR}/misc/onnx/opencv-onnx.pb.cc")
endif()
```

**E. TensorFlow Lite Support (Lines 141-154)**

```cmake
ocv_option(OPENCV_DNN_TFLITE "Build with TFLite support" 
    (TARGET ocv.3rdparty.flatbuffers))

if(TARGET ocv.3rdparty.flatbuffers AND OPENCV_DNN_TFLITE)
  if(NOT HAVE_FLATBUFFERS)
    message(FATAL_ERROR "TFLite is not supported without flatbuffers")
  endif()
  list(APPEND libs ocv.3rdparty.flatbuffers)
  list(APPEND fw_hdrs 
    "${CMAKE_CURRENT_LIST_DIR}/misc/tflite/schema_generated.h")
endif()
```

**F. Backend Dependencies (Lines 164-209)**

**OpenCL Include Paths (Lines 165-169):**
```cmake
if(OPENCV_DNN_OPENCL AND HAVE_OPENCL)
  list(APPEND include_dirs ${OPENCL_INCLUDE_DIRS})
else()
  set(sources_options EXCLUDE_OPENCL)
endif()
```

**CUDA Dependencies (Lines 171-189):**
```cmake
if(OPENCV_DNN_CUDA AND HAVE_CUDA AND HAVE_CUBLAS AND HAVE_CUDNN)
  list(APPEND include_dirs ${CUDA_TOOLKIT_INCLUDE} ${CUDNN_INCLUDE_DIRS})
  
  # Validate CUDA compute capability >= 3.0
  set(CC_LIST ${CUDA_ARCH_BIN})
  separate_arguments(CC_LIST)
  foreach(cc ${CC_LIST})
    if(cc VERSION_LESS 3.0)
      message(FATAL_ERROR "CUDA CC 3.0+ required")
    endif()
  endforeach()
  
  if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
    list(APPEND libs 
      CUDA::cudart${CUDA_LIB_EXT} 
      ${CUDNN_LIBRARIES} 
      CUDA::cublas${CUDA_LIB_EXT})
    if(NOT CUDA_VERSION VERSION_LESS 10.1)
      list(APPEND libs CUDA::cublasLt${CUDA_LIB_EXT})
    endif()
  endif()
endif()
```

**G. OpenVINO Plugin Support (Lines 237-249)**

```cmake
ocv_option(OPENCV_DNN_OPENVINO "Build with OpenVINO support (2021.4+)" 
    (TARGET ocv.3rdparty.openvino))

if(TARGET ocv.3rdparty.openvino AND OPENCV_DNN_OPENVINO)
  if(NOT HAVE_OPENVINO AND NOT HAVE_NGRAPH)
    message(FATAL_ERROR "OpenVINO requires nGraph")
  endif()
  
  if("openvino" IN_LIST DNN_PLUGIN_LIST OR DNN_PLUGIN_LIST STREQUAL "all")
    # Build as plugin in separate subdirectory
    add_subdirectory(
      "${CMAKE_CURRENT_LIST_DIR}/misc/plugin/openvino"
      "${CMAKE_CURRENT_BINARY_DIR}/dnn_plugin_openvino")
  elseif(NOT OPENCV_DNN_BUILTIN_BACKEND)
    # Link as regular library
    list(APPEND dnn_runtime_libs ocv.3rdparty.openvino)
  endif()
endif()
```

**H. Default Backend Selection (Lines 251-254)**

```cmake
set(OPENCV_DNN_BACKEND_DEFAULT "" CACHE STRING 
    "Default backend (DNN_BACKEND_OPENCV if empty)")

if(OPENCV_DNN_BACKEND_DEFAULT)
  ocv_append_source_file_compile_definitions(
    "${CMAKE_CURRENT_LIST_DIR}/src/dnn_params.cpp"
    "OPENCV_DNN_BACKEND_DEFAULT=${OPENCV_DNN_BACKEND_DEFAULT}")
endif()
```

### 3.2 DNN Module CMake Subdirectories

**Directory:** `/home/user/opencv/modules/dnn/cmake/`

**A. init.cmake**
```cmake
# Module initialization hook (currently minimal)
# Called during first CMake pass via:
#   include("${__path}/cmake/init.cmake" OPTIONAL)
```

**B. plugin.cmake (82 lines)**

**Plugin Creation Function (Lines 1-81):**
```cmake
function(ocv_create_builtin_dnn_plugin name target)
  # Purpose: Create DNN backend as loadable plugin (MODULE library)
  # Called for: OpenVINO, TVM, TensorRT backends
  
  # Key configuration:
  add_library(${name} MODULE ${sources})
  
  # Link dependencies
  target_link_libraries(${name} PRIVATE ${target})
  target_link_libraries(${name} PRIVATE 
    opencv_dnn opencv_core opencv_imgproc)
  
  # Install plugin
  if(WIN32)
    install(TARGETS ${name} LIBRARY DESTINATION ${OPENCV_BIN_INSTALL_PATH})
  else()
    install(TARGETS ${name} LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH})
  endif()
endfunction()
```

### 3.3 DNN Source Organization

**Directory:** `/home/user/opencv/modules/dnn/src/` (~50 subdirectories)

**Key Subdirectories:**

**Framework Importers:**
- `caffe/` - Caffe model loader
- `darknet/` - Darknet/YOLO loader
- `onnx/` - ONNX Runtime loader
- `tensorflow/` - TensorFlow Lite & TensorFlow loader
- `torch/` - PyTorch/ONNX loader

**Backend Implementations:**
- `cuda/` - CUDA-accelerated kernels
- `cuda4dnn/` - CUDA DNN layer implementations
- `ocl4dnn/` - OpenCL DNN kernels
- `vkcom/` - Vulkan compute kernels
- `webnn/` - WebNN API backend

**Layer Implementations:**
- `layers/` - CPU implementations + dispatched SIMD
- `layers/cpu_kernels/` - Specialized SIMD kernels
- `int8layers/` - Integer quantization layers

**Core Components:**
- `backend.cpp` - Backend selection and routing
- `registry.cpp` - Layer registration system
- `dnn.cpp` - Module initialization
- `dnn_read.cpp` - Model file parsing

---

## 4. BUILD CONFIGURATION

### 4.1 Configuration Options Pattern

**CMakeLists.txt (Lines 195-500):** Define build options

**Core Options Structure:**
```cmake
# OCV_OPTION macro wraps option() with additional features:
OCV_OPTION(OPTION_NAME "Description" DEFAULT_VALUE 
  VISIBLE_IF condition
  VERIFY have_variable)
```

**Categories:**

**A. Build Type Selection (Lines 200-250):**
```cmake
OCV_OPTION(BUILD_ZLIB "Build zlib from source" 
  (WIN32 OR APPLE OR OPENCV_FORCE_3RDPARTY_BUILD))
OCV_OPTION(BUILD_TIFF "Build libtiff from source" 
  (WIN32 OR ANDROID OR APPLE OR OPENCV_FORCE_3RDPARTY_BUILD))
OCV_OPTION(BUILD_JPEG "Build libjpeg from source"
  (WIN32 OR ANDROID OR APPLE OR OPENCV_FORCE_3RDPARTY_BUILD))
OCV_OPTION(BUILD_PNG "Build libpng from source"
  (WIN32 OR ANDROID OR APPLE OR OPENCV_FORCE_3RDPARTY_BUILD))
```

**B. Hardware Acceleration (Lines 241-298):**
```cmake
OCV_OPTION(WITH_CUDA "Include NVidia Cuda Runtime support" OFF
  VISIBLE_IF NOT IOS AND NOT XROS AND NOT WINRT
  VERIFY HAVE_CUDA)

OCV_OPTION(WITH_CUDNN "Include NVIDIA CUDA Deep Neural Network support" 
  WITH_CUDA
  VISIBLE_IF WITH_CUDA
  VERIFY HAVE_CUDNN)

OCV_OPTION(WITH_OPENCL "Include OpenCL Runtime support" 
  (NOT ANDROID AND NOT CV_DISABLE_OPTIMIZATION)
  VISIBLE_IF NOT IOS AND NOT XROS AND NOT WINRT
  VERIFY HAVE_OPENCL)

OCV_OPTION(WITH_VULKAN "Include Vulkan support" OFF
  VISIBLE_IF TRUE
  VERIFY HAVE_VULKAN)
```

**C. Platform-Specific Restrictions (Lines 217-301):**
```cmake
# iOS/visionOS exclusions
OCV_OPTION(WITH_FFMPEG "Include FFMPEG support" (NOT ANDROID)
  VISIBLE_IF NOT IOS AND NOT XROS AND NOT WINRT
  VERIFY HAVE_FFMPEG)

OCV_OPTION(WITH_GSTREAMER "Include Gstreamer support" ON
  VISIBLE_IF NOT ANDROID AND NOT IOS AND NOT XROS AND NOT WINRT
  VERIFY HAVE_GSTREAMER AND GSTREAMER_VERSION VERSION_GREATER "0.99")

# Apple-specific
OCV_OPTION(WITH_AVFOUNDATION "Use AVFoundation for Video I/O" ON
  VISIBLE_IF APPLE
  VERIFY HAVE_AVFOUNDATION)

# iOS-specific
OCV_OPTION(WITH_CAP_IOS "Enable iOS video capture" ON
  VISIBLE_IF IOS
  VERIFY HAVE_CAP_IOS)
```

### 4.2 Platform Detection

**File:** `/home/user/opencv/cmake/OpenCVDetectCXXCompiler.cmake`

**Three-level Detection:**
1. **CMAKE_SYSTEM_NAME**: Operating system
   - `Darwin` → macOS
   - `iOS` → iOS
   - `Linux` → Linux
   - `Windows` → Windows

2. **CMAKE_SYSTEM_PROCESSOR**: CPU architecture
   - `x86_64`, `x86`, `armv7`, `armv8`, etc.

3. **Compiler**: GCC/Clang/MSVC variants
   - Version extraction for feature availability
   - Flag compatibility checking

**Default Values:**
```cmake
# If not explicitly set, detect from CMAKE_SYSTEM_NAME
if(NOT DEFINED CMAKE_SYSTEM_NAME)
  if(APPLE)
    # Detect Darwin vs iOS from CMAKE_OSX_SYSROOT
    if(CMAKE_OSX_SYSROOT MATCHES "iphoneos|iphonesimulator")
      set(CMAKE_SYSTEM_NAME iOS)
    else()
      set(CMAKE_SYSTEM_NAME Darwin)
    endif()
  endif()
endif()
```

### 4.3 Compiler Flags

**File:** `/home/user/opencv/cmake/OpenCVCompilerOptions.cmake` (25,883 bytes)

**MSVC-Specific (Lines ~200-300):**
```cmake
if(MSVC)
  # Runtime selection: static vs dynamic
  if(BUILD_WITH_STATIC_CRT)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  endif()
  
  # Warning suppression
  add_compile_options(/wd4127 /wd4251 /wd4275 /wd4512)
endif()
```

**GCC/Clang-Specific (Lines ~350-400):**
```cmake
if(CMAKE_COMPILER_IS_GNUCXX)
  # C++11/14/17 selection
  add_compile_options(-std=c++11)
  
  # Warning flags
  add_compile_options(-Wall -Wextra)
  if(ENABLE_NOISY_WARNINGS)
    add_compile_options(-pedantic)
  endif()
endif()
```

**Apple-Specific (Lines ~100-150):**
```cmake
if(APPLE)
  # Framework linking
  set(CMAKE_OSX_ARCHITECTURES x86_64)  # Can be set to universal
  
  # Deployment target
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.12)
  endif()
  
  # Visibility
  add_compile_options(-fvisibility=hidden)
endif()
```

### 4.4 SIMD Configuration

**File:** `/home/user/opencv/cmake/OpenCVCompilerOptimizations.cmake` (41,987 bytes)

**CPU_BASELINE vs CPU_DISPATCH:**

**Baseline (Lines 33-45):**
```cmake
# Compiled with architecture base flags into every file
# Default: SSE2 for x86_64, NEON for ARM
CPU_BASELINE="SSE2;SSE3;SSSE3"

# CMake variables generated:
CPU_BASELINE_FINAL  # Final resolved list
CPU_BASELINE_FEATURES  # Parsed into individual flags
```

**Dispatch (Lines 45-70):**
```cmake
# Compiled separately in dispatch files
# Provides runtime selection via cv::getCPUFeatures()
CPU_DISPATCH="AVX;AVX2;AVX512_SKX"

# Results in:
# - layers_common.avx.cpp compiled with -mavx
# - layers_common.avx2.cpp compiled with -mavx2
# - layers_common.avx512_skx.cpp compiled with -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl
```

**Optimization Implications (Lines 100-150):**
```
SSE2:       Always enabled on x86_64
NEON:       Always enabled on ARMv7+
AVX:        Optional dispatch for Haswell+
AVX2:       Optional dispatch for Haswell+
AVX512_SKX: Optional dispatch for Skylake-X+

For Apple (iOS/macOS):
- Automatically selects NEON for ARM (arm64)
- Automatically selects SSE2/AVX for Intel (x86_64)
- Universal binaries include both via fat binary mechanism
```

---

## 5. BUILD FLOW DIAGRAMS

### 5.1 Complete Build System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INVOKES CMAKE                            │
│                  cmake -B build -S . [options]                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ROOT CMakeLists.txt (Entry Point)                 │
│  • Set CMake policies (CMP0042, CMP0051, CMP0054, etc.)             │
│  • Load cmake/OpenCVUtils.cmake, cmake/OpenCVMinDepVersions.cmake   │
│  • Execute hooks: CMAKE_INIT, PRE_CMAKE_BOOTSTRAP                   │
│  • project(OpenCV CXX C) - Triggers platform detection              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PLATFORM DETECTION                              │
│  • CMAKE_SYSTEM_NAME (Darwin, Linux, Windows, Android, etc.)       │
│  • CMAKE_SYSTEM_PROCESSOR (x86_64, arm64, armv7, etc.)             │
│  • Load platform-specific: platforms/${CMAKE_SYSTEM_NAME}.cmake    │
│  • iOS: Load toolchain from platforms/ios/cmake/Toolchains/        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPILER & TOOLCHAIN SETUP                        │
│  • cmake/OpenCVDetectCXXCompiler.cmake - Detect compiler            │
│  • cmake/OpenCVCompilerOptions.cmake - Set flags                    │
│  • cmake/OpenCVCompilerOptimizations.cmake - SIMD config            │
│  • cmake/OpenCVCompilerDefenses.cmake - Security hardening          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              BUILD OPTIONS CONFIGURATION                             │
│  • Process BUILD_* options (BUILD_opencv_dnn, BUILD_SHARED_LIBS)   │
│  • Process WITH_* options (WITH_CUDA, WITH_OPENCL, WITH_VULKAN)    │
│  • Detect 3rdparty dependencies (protobuf, CUDA, OpenCL, etc.)     │
│  • Set CPU_BASELINE, CPU_DISPATCH for SIMD                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  TWO-PASS MODULE CONFIGURATION                       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  PASS 1: Module Discovery & Dependency Resolution        │      │
│  │  • set(OPENCV_INITIAL_PASS ON)                           │      │
│  │  • Scan modules/ directory                               │      │
│  │  • For each module CMakeLists.txt:                       │      │
│  │    ├─ option(BUILD_opencv_xxx "Include module")          │      │
│  │    ├─ ocv_add_module(name deps...) collects metadata     │      │
│  │    └─ return() - Don't create targets yet                │      │
│  │  • __ocv_resolve_dependencies() builds dep graph         │      │
│  │  • Disable modules with unmet dependencies               │      │
│  │  • Sort modules topologically                            │      │
│  │  • set(OPENCV_INITIAL_PASS OFF)                          │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                 │                                    │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  PASS 2: Target Creation & Build Rules                   │      │
│  │  • For each enabled module (in dependency order):        │      │
│  │    ├─ project(opencv_xxx)                                │      │
│  │    ├─ ocv_glob_module_sources() - Find .cpp/.hpp files   │      │
│  │    ├─ ocv_create_module() - add_library(opencv_xxx)      │      │
│  │    ├─ Add SIMD dispatch sources                          │      │
│  │    ├─ target_link_libraries() - Link dependencies        │      │
│  │    ├─ Generate bindings (Python/Java/ObjC/JS)           │      │
│  │    └─ install(TARGETS ...) - Installation rules          │      │
│  └──────────────────────────────────────────────────────────┘      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  CONFIGURATION GENERATION                            │
│  • OpenCVConfig.cmake - For find_package(OpenCV)                    │
│  • opencv_modules.hpp - Module list for C++                         │
│  • cvconfig.h - Platform-specific defines                           │
│  • cv_cpu_config.h - CPU feature defines                            │
│  • Info.plist (Apple platforms)                                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BUILD EXECUTION                                 │
│  cmake --build build [-j N]  OR  make -j N  OR  xcodebuild         │
│  • Compile baseline implementations                                 │
│  • Compile SIMD dispatched variants                                 │
│  • Link libraries                                                   │
│  • Generate bindings                                                │
│  • (Apple) Build framework bundles                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         INSTALLATION                                 │
│  cmake --install build  OR  make install                            │
│  • ${PREFIX}/lib/ - Libraries (.so, .dylib, .a)                    │
│  • ${PREFIX}/include/opencv2/ - Headers                            │
│  • ${PREFIX}/lib/cmake/opencv4/ - CMake config                     │
│  • ${PREFIX}/share/opencv4/ - Data files                           │
│  • (Apple) ${PREFIX}/opencv2.framework/ - Framework bundle         │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Apple Platform Build Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   APPLE PLATFORM BUILD ENTRY                         │
│                                                                      │
│  Option A: Direct CMake (macOS native)                              │
│    cmake -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" ...              │
│                                                                      │
│  Option B: Framework Build Script (iOS/macOS)                       │
│    python platforms/ios/build_framework.py                          │
│    python platforms/osx/build_framework.py                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PLATFORM DETECTION                              │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐    │
│  │  macOS Native  │  │  iOS Framework │  │  Catalyst Build   │    │
│  │  • x86_64      │  │  • armv7/arm64 │  │  • x86_64 Mac +   │    │
│  │  • arm64       │  │  • i386/x86_64 │  │    iOS APIs       │    │
│  │  (Universal)   │  │  (Multi-arch)  │  │  • arm64 (M1+)    │    │
│  └────────┬───────┘  └────────┬───────┘  └─────────┬─────────┘    │
│           │                    │                     │               │
│           └────────────────────┴─────────────────────┘               │
│                                │                                     │
└────────────────────────────────┼─────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  TOOLCHAIN CONFIGURATION                             │
│                                                                      │
│  macOS:                                                              │
│    • CMAKE_OSX_DEPLOYMENT_TARGET=10.12                              │
│    • CMAKE_OSX_ARCHITECTURES=x86_64;arm64                           │
│    • No custom toolchain file needed                                │
│                                                                      │
│  iOS Device (iPhoneOS):                                              │
│    • Toolchain: Toolchain-iPhoneOS_Xcode.cmake                      │
│    • CMAKE_OSX_SYSROOT=iphoneos                                     │
│    • IOS_ARCH=armv7/armv7s/arm64                                    │
│    • IPHONEOS_DEPLOYMENT_TARGET=9.0                                 │
│                                                                      │
│  iOS Simulator:                                                      │
│    • Toolchain: Toolchain-iPhoneSimulator_Xcode.cmake               │
│    • CMAKE_OSX_SYSROOT=iphonesimulator                              │
│    • IOS_ARCH=i386/x86_64/arm64                                     │
│                                                                      │
│  visionOS (Apple Vision Pro):                                        │
│    • Toolchain: Toolchain-XROS_Xcode.cmake                          │
│    • CMAKE_OSX_SYSROOT=xros/xrsimulator                             │
│    • VISIONOS_ARCH=arm64/x86_64                                     │
│    • XROS_DEPLOYMENT_TARGET=1.0                                     │
│                                                                      │
│  Catalyst (Mac Catalyst):                                            │
│    • Toolchain: Toolchain-Catalyst_Xcode.cmake                      │
│    • CMAKE_OSX_SYSROOT=macosx                                       │
│    • Target triple: x86_64-apple-ios14.0-macabi                     │
│    • -iframework /System/iOSSupport/System/Library/Frameworks       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FRAMEWORK-SPECIFIC CONFIGURATION                        │
│                                                                      │
│  • APPLE_FRAMEWORK=ON                                                │
│  • BUILD_SHARED_LIBS=ON/OFF (Dynamic vs Static framework)          │
│  • ENABLE_BITCODE=ON (iOS device builds)                            │
│  • Disable incompatible features:                                   │
│    ├─ CV_OCL4DNN=0 (OpenCL for DNN disabled on Apple)              │
│    ├─ WITH_FFMPEG=OFF (iOS/visionOS)                               │
│    └─ WITH_GSTREAMER=OFF (iOS/visionOS)                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│         MULTI-ARCHITECTURE BUILD (Framework Scripts)                 │
│                                                                      │
│  For iOS Framework:                                                  │
│    ┌───────────────────────────────────────────────────────┐       │
│    │ Build arm64 (device)                                  │       │
│    │  • mkdir build-arm64                                  │       │
│    │  • cmake -DIOS_ARCH=arm64 ...                         │       │
│    │  • xcodebuild -arch arm64 -sdk iphoneos               │       │
│    └───────────────────────────────────────────────────────┘       │
│                                 │                                    │
│    ┌───────────────────────────────────────────────────────┐       │
│    │ Build x86_64 (simulator)                              │       │
│    │  • mkdir build-x86_64                                 │       │
│    │  • cmake -DIOS_ARCH=x86_64 ...                        │       │
│    │  • xcodebuild -arch x86_64 -sdk iphonesimulator       │       │
│    └───────────────────────────────────────────────────────┘       │
│                                 │                                    │
│    ┌───────────────────────────────────────────────────────┐       │
│    │ Create Universal Binary (lipo)                        │       │
│    │  • lipo -create build-arm64/libopencv_xxx.a \         │       │
│    │             build-x86_64/libopencv_xxx.a \            │       │
│    │        -output opencv2.framework/opencv2              │       │
│    └───────────────────────────────────────────────────────┘       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FRAMEWORK BUNDLE CREATION                          │
│                                                                      │
│  opencv2.framework/                                                  │
│  ├── opencv2 (or OpenCV) - Universal binary                         │
│  ├── Headers/                                                       │
│  │   ├── opencv2/                                                   │
│  │   │   ├── core.hpp                                               │
│  │   │   ├── dnn.hpp                                                │
│  │   │   └── ... (all public headers)                              │
│  │   └── opencv2.hpp (master header)                               │
│  ├── Modules/                                                       │
│  │   └── module.modulemap (for Xcode auto-complete)                │
│  └── Resources/                                                     │
│      ├── Info.plist                                                 │
│      │   • CFBundleName: OpenCV                                     │
│      │   • CFBundleIdentifier: org.opencv                           │
│      │   • CFBundleVersion: 4.x.x                                   │
│      │   • MinimumOSVersion: 9.0                                    │
│      └── PrivacyInfo.xcprivacy (iOS 17+)                           │
│                                                                      │
│  Code Signing (Optional):                                            │
│    codesign -s "Developer ID" opencv2.framework                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 DNN Module Build & Backend Configuration

```
┌─────────────────────────────────────────────────────────────────────┐
│              DNN MODULE BUILD INITIALIZATION                         │
│  modules/dnn/CMakeLists.txt                                          │
│                                                                      │
│  ocv_add_module(dnn opencv_core opencv_imgproc                      │
│                 WRAP python java objc js)                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MODULE DEPENDENCIES                                │
│                                                                      │
│   opencv_core ────────┐                                             │
│                       ├──→ opencv_dnn ──→ Language Bindings        │
│   opencv_imgproc ─────┘                  • python (OpenCV-Python)   │
│                                           • java (OpenCV4Android)   │
│                                           • objc (iOS framework)    │
│                                           • js (OpenCV.js)          │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              BACKEND DETECTION & CONFIGURATION                       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  OpenCL Backend (CV_OCL4DNN)                              │      │
│  │  • if(NOT APPLE): CV_OCL4DNN=1                            │      │
│  │  • if(APPLE): CV_OCL4DNN=0 (disabled)                     │      │
│  │  • Requires: OpenCL headers, libOpenCL                    │      │
│  │  • Source: src/ocl4dnn/                                   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  CUDA Backend (CV_CUDA4DNN)                               │      │
│  │  • if(WITH_CUDA AND CUDA_VERSION >= 10.0)                 │      │
│  │  • Requires: CUDA toolkit, cuDNN >= 7.6                   │      │
│  │  • Source: src/cuda4dnn/                                  │      │
│  │  • Detection: cmake/OpenCVDetectCUDA.cmake                │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Vulkan Backend (CV_VULKAN)                               │      │
│  │  • if(WITH_VULKAN): CV_VULKAN=1                           │      │
│  │  • Requires: Vulkan SDK                                   │      │
│  │  • Source: src/vkcom/                                     │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  WebNN Backend (CV_DNN_WEBNN)                             │      │
│  │  • if(WITH_WEBNN): CV_DNN_WEBNN=1                         │      │
│  │  • Source: src/webnn/                                     │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  TensorFlow Lite (CV_DNN_TFLITE)                          │      │
│  │  • if(WITH_TFLITE): Link TensorFlow Lite                  │      │
│  │  • Source: src/tflite/tflite_importer.cpp                 │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  OpenVINO / Inference Engine (Plugin System)              │      │
│  │  • ocv_create_builtin_dnn_plugin()                        │      │
│  │  • Separate .so/.dylib plugin                             │      │
│  │  • Runtime loading via dlopen                             │      │
│  └──────────────────────────────────────────────────────────┘      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PROTOBUF CONFIGURATION                              │
│                                                                      │
│  cmake/OpenCVFindProtobuf.cmake:                                     │
│    if(BUILD_PROTOBUF)  # Default: ON                                │
│      # Use 3rdparty/protobuf                                        │
│      add_subdirectory(3rdparty/protobuf)                            │
│      set(Protobuf_LIBRARIES libprotobuf)                            │
│    else()                                                            │
│      # Use system protobuf                                          │
│      find_package(Protobuf REQUIRED)                                │
│    endif()                                                           │
│                                                                      │
│  Generate Protocol Buffer sources:                                  │
│    src/opencv-caffe.proto  → opencv-caffe.pb.cc/.h                  │
│    src/opencv-onnx.proto   → opencv-onnx.pb.cc/.h                   │
│    src/opencv-tensorflow.proto → opencv-tensorflow.pb.cc/.h         │
│                                                                      │
│  Fallback: Pre-generated in misc/                                   │
│    modules/dnn/misc/caffe/opencv-caffe.pb.cc                        │
│    modules/dnn/misc/onnx/opencv-onnx.pb.cc                          │
│    modules/dnn/misc/tensorflow/opencv-tensorflow.pb.cc              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SIMD DISPATCH CONFIGURATION                        │
│                                                                      │
│  modules/dnn/CMakeLists.txt (Lines 7-12):                           │
│    ocv_add_dispatched_file(                                         │
│      "layers/layers_common"                                         │
│      AVX AVX2 AVX512_SKX RVV LASX NEON)                            │
│                                                                      │
│  Generated files:                                                    │
│    layers/layers_common.avx.cpp       (-mavx)                       │
│    layers/layers_common.avx2.cpp      (-mavx2)                      │
│    layers/layers_common.avx512_skx.cpp (-mavx512f -mavx512cd ...)  │
│    layers/layers_common.rvv.cpp       (RISC-V Vector)               │
│    layers/layers_common.lasx.cpp      (LoongArch LASX)              │
│    layers/layers_common.neon.cpp      (-mfpu=neon / arm64)          │
│                                                                      │
│  Runtime dispatch in layers_common.simd.hpp:                        │
│    CV_CPU_CALL_AVX2(func, args...)   - Calls AVX2 if available      │
│    CV_CPU_CALL_AVX(func, args...)    - Falls back to AVX            │
│    CV_CPU_CALL_BASELINE(func, args...) - Ultimate fallback          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DNN SOURCE COMPILATION                            │
│                                                                      │
│  Core Implementation:                                                │
│    src/dnn.cpp, src/net.cpp, src/layer.cpp                         │
│    src/layer_factory.cpp, src/init.cpp                             │
│    src/net_impl.cpp, src/net_impl_fuse.cpp                         │
│    src/graph_simplifier.cpp                                         │
│                                                                      │
│  Layer Implementations (50+ files):                                  │
│    src/layers/convolution_layer.cpp                                 │
│    src/layers/pooling_layer.cpp                                     │
│    src/layers/fully_connected_layer.cpp                             │
│    src/layers/*.cpp                                                 │
│                                                                      │
│  Model Importers:                                                    │
│    src/caffe/caffe_importer.cpp                                     │
│    src/tensorflow/tf_importer.cpp                                   │
│    src/onnx/onnx_importer.cpp                                       │
│    src/darknet/darknet_importer.cpp                                 │
│    src/torch/torch_importer.cpp                                     │
│    src/tflite/tflite_importer.cpp                                   │
│                                                                      │
│  Backend Implementations:                                            │
│    src/cuda4dnn/ (CUDA)                                             │
│    src/ocl4dnn/ (OpenCL)                                            │
│    src/vkcom/ (Vulkan)                                              │
│    src/webnn/ (WebNN)                                               │
│                                                                      │
│  Quantization:                                                       │
│    src/int8layers/ (INT8 quantized variants)                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LINK & INSTALL                                 │
│                                                                      │
│  target_link_libraries(opencv_dnn                                   │
│    opencv_core opencv_imgproc                                       │
│    ${Protobuf_LIBRARIES}                                            │
│    ${CUDA_LIBRARIES}  # If WITH_CUDA                                │
│    ${OpenCL_LIBRARIES}  # If NOT APPLE                              │
│    ${Vulkan_LIBRARIES}  # If WITH_VULKAN                            │
│  )                                                                   │
│                                                                      │
│  install(TARGETS opencv_dnn                                         │
│    LIBRARY DESTINATION lib                                          │
│    ARCHIVE DESTINATION lib                                          │
│    FRAMEWORK DESTINATION .  # For Apple frameworks                  │
│  )                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Module Dependency Resolution Algorithm

```
┌─────────────────────────────────────────────────────────────────────┐
│                   TWO-PASS DEPENDENCY RESOLUTION                     │
└─────────────────────────────────────────────────────────────────────┘

PASS 1: MODULE DISCOVERY
────────────────────────

For each module in modules/:
  │
  ├─→ Read CMakeLists.txt
  │   │
  │   └─→ ocv_add_module(name [REQ_DEPS] [OPT_DEPS])
  │       │
  │       ├─→ Create BUILD_opencv_name option
  │       ├─→ Store module location
  │       ├─→ Record dependencies in global lists:
  │       │   • OPENCV_MODULE_${name}_REQ_DEPS
  │       │   • OPENCV_MODULE_${name}_OPT_DEPS
  │       │   • OPENCV_MODULE_${name}_PRIVATE_REQ_DEPS
  │       │   • OPENCV_MODULE_${name}_PRIVATE_OPT_DEPS
  │       │
  │       └─→ return()  # Don't process further
  │
  └─→ Module registered, move to next

After all modules scanned:
  │
  └─→ __ocv_resolve_dependencies()
      │
      ├─→ Initialize 4 lists:
      │   • OPENCV_MODULES_BUILD (enabled)
      │   • OPENCV_MODULES_DISABLED_USER (user disabled)
      │   • OPENCV_MODULES_DISABLED_AUTO (dep failed)
      │   • OPENCV_MODULES_DISABLED_FORCE (incompatible)
      │
      ├─→ Apply BUILD_LIST whitelist if specified
      │
      ├─→ For each module in BUILD list:
      │   │
      │   └─→ Check required dependencies:
      │       │
      │       ├─→ If REQ_DEP missing or disabled:
      │       │   └─→ Move module to DISABLED_AUTO
      │       │
      │       └─→ If OPT_DEP available:
      │           └─→ Add to resolved dependencies
      │
      ├─→ Detect circular dependencies (error if found)
      │
      ├─→ Topological sort by dependency order:
      │   │
      │   └─→ OPENCV_MODULES_BUILD = sorted list
      │       Example order:
      │         1. opencv_core (no deps)
      │         2. opencv_imgproc (→ core)
      │         3. opencv_dnn (→ core, imgproc)
      │         4. opencv_python (→ WRAP all)
      │
      └─→ Print summary of enabled/disabled modules


PASS 2: TARGET CREATION
────────────────────────

set(OPENCV_INITIAL_PASS OFF)

For each module in OPENCV_MODULES_BUILD (sorted order):
  │
  ├─→ add_subdirectory(modules/${name})
  │   │
  │   └─→ modules/${name}/CMakeLists.txt
  │       │
  │       ├─→ project(opencv_${name})
  │       │
  │       ├─→ ocv_glob_module_sources()
  │       │   ├─ Find .cpp, .hpp, .cu files
  │       │   └─ Exclude tests, perf, samples
  │       │
  │       ├─→ ocv_add_dispatched_file() for SIMD
  │       │   └─ Generate AVX/AVX2/NEON variants
  │       │
  │       ├─→ ocv_create_module()
  │       │   │
  │       │   ├─→ add_library(opencv_${name} ${SOURCES})
  │       │   │
  │       │   ├─→ target_link_libraries()
  │       │   │   • Link resolved dependencies
  │       │   │   • Link 3rdparty libraries
  │       │   │
  │       │   ├─→ set_target_properties()
  │       │   │   • VERSION, SOVERSION
  │       │   │   • OUTPUT_NAME
  │       │   │   • FRAMEWORK properties (Apple)
  │       │   │
  │       │   └─→ install(TARGETS ...)
  │       │
  │       ├─→ Generate language bindings:
  │       │   ├─ Python: modules/python/src2/cv2_${name}.cpp
  │       │   ├─ Java: modules/java/generator/gen_java.py
  │       │   ├─ ObjC: modules/objc/gen_objc.py
  │       │   └─ JS: modules/js/generator/embindgen.py
  │       │
  │       └─→ Add tests/perf subdirectories
  │
  └─→ Module built, move to next

Result: All libraries linked in correct dependency order
```

### 5.5 SIMD Optimization Dispatch Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│               SIMD CONFIGURATION & RUNTIME DISPATCH                  │
└─────────────────────────────────────────────────────────────────────┘

BUILD TIME CONFIGURATION
────────────────────────

cmake/OpenCVCompilerOptimizations.cmake:
  │
  ├─→ Detect CPU architecture:
  │   • x86/x86_64: SSE2, SSE3, SSSE3, SSE4_1, SSE4_2, AVX, FMA3, AVX2, AVX512_*
  │   • ARM: NEON, VFPV3, FP16, NEON_DOTPROD, NEON_FP16, NEON_BF16
  │   • PowerPC: VSX, VSX3
  │   • MIPS: MSA
  │   • RISC-V: RVV (Vector Extension)
  │   • LoongArch: LSX, LASX
  │
  ├─→ Set CPU_BASELINE (always compiled):
  │   │
  │   ├─ x86_64:  SSE2 (minimum for 64-bit x86)
  │   ├─ ARMv7:   NEON (if hardware supports)
  │   ├─ ARMv8:   NEON (always available on arm64)
  │   └─ Default: No special instructions
  │
  └─→ Set CPU_DISPATCH (runtime selection):
      │
      ├─ x86_64:  AVX, AVX2, AVX512_SKX, AVX512_ICL, AVX512_COMMON
      ├─ ARM:     NEON (if not baseline), NEON_FP16
      ├─ RISC-V:  RVV
      └─ LoongArch: LASX

Module declares dispatched files:
  │
  └─→ ocv_add_dispatched_file("path/file" AVX AVX2 NEON)
      │
      ├─→ Generate: file.avx.cpp, file.avx2.cpp, file.neon.cpp
      │
      └─→ Compile each with target flags:
          • file.avx.cpp    → -mavx
          • file.avx2.cpp   → -mavx2 -mf16c -mfma
          • file.neon.cpp   → -mfpu=neon (ARMv7) or default (ARMv8)


RUNTIME DISPATCH MECHANISM
───────────────────────────

Application starts:
  │
  └─→ cv::checkHardwareSupport() called
      │
      ├─→ Read CPUID (x86) or HWCAP (ARM)
      │
      ├─→ Detect available features:
      │   • cv::CPU_SSE2, cv::CPU_AVX2, etc.
      │
      └─→ Store in global cv::cpu_features variable

User calls OpenCV function (e.g., cv::dnn::convolution):
  │
  └─→ Function implementation in layers_common.simd.hpp:
      │
      ├─→ CV_CPU_DISPATCH(func, args...,
      │       (CV_CPU_CALL_AVX512_SKX(func_avx512, args...),
      │        CV_CPU_CALL_AVX2(func_avx2, args...),
      │        CV_CPU_CALL_AVX(func_avx, args...),
      │        CV_CPU_CALL_BASELINE(func_baseline, args...)))
      │   │
      │   ├─→ if (cv::cpu_features & CV_CPU_AVX512_SKX)
      │   │   └─→ call func_avx512(args...)
      │   │
      │   ├─→ else if (cv::cpu_features & CV_CPU_AVX2)
      │   │   └─→ call func_avx2(args...)
      │   │
      │   ├─→ else if (cv::cpu_features & CV_CPU_AVX)
      │   │   └─→ call func_avx(args...)
      │   │
      │   └─→ else
      │       └─→ call func_baseline(args...)
      │
      └─→ Returns result from optimal implementation

Example: DNN Layer Forward Pass
  │
  ├─→ Baseline: Generic C++ loop
  ├─→ AVX:      Process 8 floats per iteration
  ├─→ AVX2:     Process 8 floats with FMA (fused multiply-add)
  └─→ AVX512:   Process 16 floats per iteration

Performance gain: 2x-8x depending on operation and CPU
```

---

## 6. FILE REFERENCE SUMMARY

### Critical Files by Function:

| Function | Primary File | Backup/Detail | Line Numbers |
|----------|-------------|----------------|-------------|
| Module registration | `cmake/OpenCVModule.cmake` | `CMakeLists.txt` | 124-223 (pass 1), 893-920 (pass 2) |
| Dependency resolution | `cmake/OpenCVModule.cmake` | - | 500-695 |
| Two-pass initialization | `cmake/OpenCVModule.cmake` | `CMakeLists.txt` | 354-398, 1097 |
| iOS toolchain | `platforms/ios/cmake/Toolchains/common-ios-toolchain.cmake` | - | 1-223 |
| iOS framework build | `platforms/ios/build_framework.py` | - | 1-500 |
| macOS framework build | `platforms/osx/build_framework.py` | - | 1-100 |
| Info.plist generation | `cmake/OpenCVGenInfoPlist.cmake` | - | 1-28 |
| DNN module config | `modules/dnn/CMakeLists.txt` | - | 1-340 |
| Protobuf setup | `cmake/OpenCVFindProtobuf.cmake` | `modules/dnn/CMakeLists.txt` | 1-117, 113-159 |
| SIMD dispatch | `cmake/OpenCVCompilerOptimizations.cmake` | `modules/dnn/CMakeLists.txt` | 1-1000, 7-12 |
| Compiler options | `cmake/OpenCVCompilerOptions.cmake` | - | 1-800 |
| CUDA detection | `cmake/OpenCVDetectCUDA.cmake` | - | 1-250 |
| Build options | `CMakeLists.txt` | - | 195-500 |

---

## CONCLUSION

OpenCV's build system represents a sophisticated multi-level architecture designed for portability across platforms while maintaining clean separation of concerns:

1. **Two-Pass CMake** enables dependency resolution before target creation
2. **Platform Abstraction** via toolchain files simplifies iOS/macOS/Android support  
3. **SIMD Dispatch** provides CPU optimization for multiple ISA variants
4. **Modular Plugin System** allows optional backends (OpenVINO, CUDA, OpenCL, Vulkan)
5. **Framework Building** integrates Python build scripts with CMake for Apple platforms

The DNN module exemplifies the system's complexity, supporting multiple inference backends with conditional compilation based on available third-party libraries (protobuf, CUDA, OpenVINO, TensorFlow Lite).

