// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "mps_activation.hpp"
#include "../metal_context.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#include "opencv2/core/utils/logger.hpp"

namespace cv { namespace dnn {

MetalReLUNode::MetalReLUNode(float slope)
    : MetalBackendNode()
    , slope_(slope)
    , reluKernel_(nil)
{
    @autoreleasepool {
        MetalContext& ctx = MetalContext::getInstance();
        id<MTLDevice> device = ctx.getDevice();

        if (device) {
            if (slope_ == 0.0f) {
                // Standard ReLU
                reluKernel_ = [[MPSCNNNeuronReLU alloc] initWithDevice:device a:0.0f];
            } else {
                // Leaky ReLU
                reluKernel_ = [[MPSCNNNeuronReLU alloc] initWithDevice:device a:slope_];
            }

            [reluKernel_ retain];
        }
    }
}

MetalReLUNode::~MetalReLUNode()
{
    @autoreleasepool {
        if (reluKernel_) {
            [reluKernel_ release];
            reluKernel_ = nil;
        }
    }
}

void MetalReLUNode::execute(const std::vector<Ptr<MetalTensor>>& inputs,
                            const std::vector<Ptr<MetalTensor>>& outputs)
{
    CV_Assert(inputs.size() == 1);
    CV_Assert(outputs.size() == 1);
    CV_Assert(inputs[0]->isValid());
    CV_Assert(outputs[0]->isValid());

    if (!reluKernel_) {
        CV_Error(Error::StsError, "DNN/Metal: ReLU kernel not initialized");
        return;
    }

    @autoreleasepool {
        MetalContext& ctx = MetalContext::getInstance();

        // Get input and output buffers
        id<MTLBuffer> inputBuffer = inputs[0]->getBuffer();
        id<MTLBuffer> outputBuffer = outputs[0]->getBuffer();

        size_t total = inputs[0]->total();

        // For simple element-wise operations like ReLU, we can use a simpler approach
        // Create compute pipeline using Metal compute shader
        id<MTLCommandBuffer> commandBuffer = ctx.createCommandBuffer();
        if (!commandBuffer) {
            CV_Error(Error::StsError, "DNN/Metal: Failed to create command buffer");
            return;
        }

        // For MVP, use simple approach with Metal compute shader
        // We'll create a basic ReLU kernel using Metal Shading Language

        // Get or create compute pipeline
        static id<MTLComputePipelineState> reluPipeline = nil;
        if (!reluPipeline) {
            // Create simple ReLU compute kernel
            NSString* kernelSource = @R"(
                #include <metal_stdlib>
                using namespace metal;

                kernel void relu_kernel(device const float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       constant uint& count [[buffer(2)]],
                                       uint gid [[thread_position_in_grid]])
                {
                    if (gid < count) {
                        output[gid] = max(input[gid], 0.0f);
                    }
                }
            )";

            NSError* error = nil;
            id<MTLLibrary> library = [ctx.getDevice() newLibraryWithSource:kernelSource
                                                                   options:nil
                                                                     error:&error];
            if (error || !library) {
                NSLog(@"Error creating Metal library: %@", error);
                CV_Error(Error::StsError, "DNN/Metal: Failed to compile ReLU kernel");
                return;
            }

            id<MTLFunction> kernelFunction = [library newFunctionWithName:@"relu_kernel"];
            if (!kernelFunction) {
                CV_Error(Error::StsError, "DNN/Metal: Failed to find relu_kernel function");
                return;
            }

            reluPipeline = [ctx.getDevice() newComputePipelineStateWithFunction:kernelFunction
                                                                           error:&error];
            [reluPipeline retain];

            if (error || !reluPipeline) {
                NSLog(@"Error creating compute pipeline: %@", error);
                CV_Error(Error::StsError, "DNN/Metal: Failed to create compute pipeline");
                return;
            }
        }

        // Encode compute command
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:reluPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        uint32_t count = static_cast<uint32_t>(total);
        [encoder setBytes:&count length:sizeof(uint32_t) atIndex:2];

        // Calculate grid size
        NSUInteger threadGroupSize = reluPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > total) {
            threadGroupSize = total;
        }

        MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((total + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:threadgroupsPerGrid
                threadsPerThreadgroup:threadsPerThreadgroup];

        [encoder endEncoding];

        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

Ptr<BackendNode> createMetalReLUNode(float slope)
{
    return makePtr<MetalReLUNode>(slope);
}

}} // namespace cv::dnn

#endif // HAVE_METAL
