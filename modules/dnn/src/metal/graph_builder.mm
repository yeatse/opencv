// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "graph_builder.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Forward declaration of MPSGraphNetImpl from op_metal.mm
@interface MPSGraphNetImpl : NSObject
@property (nonatomic, strong) MPSGraph* graph;
@property (nonatomic, strong) NSMutableDictionary<NSString*, MPSGraphTensor*>* namedTensors;
@end

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

void* MetalGraphBuilder::Max(void* tensor1, void* tensor2, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor1 || !tensor2) return nullptr;

        MPSGraphTensor* t1 = (__bridge MPSGraphTensor*)tensor1;
        MPSGraphTensor* t2 = (__bridge MPSGraphTensor*)tensor2;
        MPSGraphTensor* output = [netImpl.graph maximumWithPrimaryTensor:t1
                                                         secondaryTensor:t2
                                                                    name:[NSString stringWithUTF8String:name.c_str()]];

        // Store named tensor
        AddTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalGraphBuilder::Min(void* tensor1, void* tensor2, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor1 || !tensor2) return nullptr;

        MPSGraphTensor* t1 = (__bridge MPSGraphTensor*)tensor1;
        MPSGraphTensor* t2 = (__bridge MPSGraphTensor*)tensor2;
        MPSGraphTensor* output = [netImpl.graph minimumWithPrimaryTensor:t1
                                                         secondaryTensor:t2
                                                                    name:[NSString stringWithUTF8String:name.c_str()]];

        // Store named tensor
        AddTensor(name, (__bridge void*)output);

        return (__bridge void*)output;
    }
}

void* MetalGraphBuilder::Div(void* tensor1, void* tensor2, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl || !tensor1 || !tensor2) return nullptr;

        MPSGraphTensor* t1 = (__bridge MPSGraphTensor*)tensor1;
        MPSGraphTensor* t2 = (__bridge MPSGraphTensor*)tensor2;
        MPSGraphTensor* output = [netImpl.graph divisionWithPrimaryTensor:t1
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

void* MetalGraphBuilder::Constant(const cv::Mat& data, const std::string& name) {
    @autoreleasepool {
        MPSGraphNetImpl* netImpl = (__bridge MPSGraphNetImpl*)impl;
        if (!netImpl) return nullptr;

        // Ensure data is continuous for MPSGraph
        cv::Mat continuousData = data;
        if (!data.isContinuous()) {
            continuousData = data.clone();
        }

        // Create NSData from Mat
        NSData* nsData = [NSData dataWithBytes:continuousData.data
                                        length:continuousData.total() * continuousData.elemSize()];

        // Create shape array
        NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
        for (int d = 0; d < continuousData.dims; d++) {
            [shape addObject:@(continuousData.size[d])];
        }

        // Get data type
        MPSDataType dataType = getMPSDataType(continuousData.type());

        // Create constant tensor
        MPSGraphTensor* tensor = [netImpl.graph constantWithData:nsData
                                                           shape:shape
                                                        dataType:dataType];

        // Store in named tensors
        AddTensor(name, (__bridge void*)tensor);

        return (__bridge void*)tensor;
    }
}

}}  // namespace cv::dnn

#endif  // HAVE_METAL
