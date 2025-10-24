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

        const MatShape& shape = inputs[0]->shape();
        size_t total = inputs[0]->total();

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = ctx.createCommandBuffer();
        if (!commandBuffer) {
            CV_Error(Error::StsError, "DNN/Metal: Failed to create command buffer");
            return;
        }

        // Determine image dimensions from tensor shape
        // For OpenCV tensors: shape can be [N, C, H, W] or just [N, C] or [N]
        NSUInteger channels = 1, height = 1, width = 1;

        if (shape.size() == 4) {
            // Standard 4D tensor: [N, C, H, W]
            channels = shape[1];
            height = shape[2];
            width = shape[3];
        } else if (shape.size() == 3) {
            // 3D tensor: [C, H, W]
            channels = shape[0];
            height = shape[1];
            width = shape[2];
        } else if (shape.size() == 2) {
            // 2D tensor: [N, C] - treat as 1D image with C channels
            channels = shape[1];
            width = shape[0];
        } else {
            // 1D tensor or other: treat as single row with all elements
            width = total;
        }

        // Create MPSImage descriptor
        MPSImageDescriptor* imageDesc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:width
                                      height:height
                             featureChannels:channels];

        // Create MPSImages pointing to our buffers
        // Note: MPSImage can wrap an MTLBuffer
        MPSImage* inputImage = [[MPSImage alloc]
            initWithDevice:ctx.getDevice()
           imageDescriptor:imageDesc];

        MPSImage* outputImage = [[MPSImage alloc]
            initWithDevice:ctx.getDevice()
           imageDescriptor:imageDesc];

        // Copy data from our buffers to MPSImage textures
        // For MVP, we do a simple copy. Future optimization: avoid this copy
        NSUInteger bytesPerRow = width * sizeof(float);
        NSUInteger bytesPerImage = bytesPerRow * height;

        for (NSUInteger c = 0; c < channels; ++c) {
            float* srcPtr = (float*)[inputBuffer contents] + c * height * width;
            MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, 1);

            [inputImage.texture replaceRegion:region
                                  mipmapLevel:0
                                        slice:c
                                    withBytes:srcPtr
                                  bytesPerRow:bytesPerRow
                                bytesPerImage:bytesPerImage];
        }

        // Encode ReLU using MPS kernel
        [reluKernel_ encodeToCommandBuffer:commandBuffer
                               sourceImage:inputImage
                          destinationImage:outputImage];

        // Copy results back to output buffer
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder synchronizeResource:outputImage.texture];
        [blitEncoder endEncoding];

        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy data from MPSImage texture back to our buffer
        for (NSUInteger c = 0; c < channels; ++c) {
            float* dstPtr = (float*)[outputBuffer contents] + c * height * width;
            MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, 1);

            [outputImage.texture getBytes:dstPtr
                              bytesPerRow:bytesPerRow
                            bytesPerImage:bytesPerImage
                               fromRegion:region
                              mipmapLevel:0
                                    slice:c];
        }

        // Clean up
        [inputImage release];
        [outputImage release];
    }
}

Ptr<BackendNode> createMetalReLUNode(float slope)
{
    return makePtr<MetalReLUNode>(slope);
}

}} // namespace cv::dnn

#endif // HAVE_METAL
