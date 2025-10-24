// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_context.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#include "opencv2/core/utils/logger.hpp"

namespace cv { namespace dnn {

MetalContext::MetalContext()
    : device_(nil)
    , commandQueue_(nil)
    , initialized_(false)
{
    initialized_ = initialize();
}

MetalContext::~MetalContext()
{
    @autoreleasepool {
        if (commandQueue_) {
            [commandQueue_ release];
            commandQueue_ = nil;
        }
        if (device_) {
            [device_ release];
            device_ = nil;
        }
    }
}

MetalContext& MetalContext::getInstance()
{
    static MetalContext instance;
    return instance;
}

bool MetalContext::initialize()
{
    @autoreleasepool {
        // Get the default Metal device
        device_ = MTLCreateSystemDefaultDevice();

        if (!device_) {
            CV_LOG_WARNING(NULL, "DNN/Metal: No Metal device found on this system");
            return false;
        }

        [device_ retain];

        // Create command queue
        commandQueue_ = [device_ newCommandQueue];
        if (!commandQueue_) {
            CV_LOG_ERROR(NULL, "DNN/Metal: Failed to create command queue");
            [device_ release];
            device_ = nil;
            return false;
        }

        [commandQueue_ retain];

        // Log device information
        CV_LOG_INFO(NULL, "DNN/Metal: Initialized with device: " << [[device_ name] UTF8String]);

        return true;
    }
}

bool MetalContext::isAvailable() const
{
    return initialized_ && device_ != nil && commandQueue_ != nil;
}

id<MTLDevice> MetalContext::getDevice() const
{
    return device_;
}

id<MTLCommandQueue> MetalContext::getCommandQueue() const
{
    return commandQueue_;
}

id<MTLCommandBuffer> MetalContext::createCommandBuffer() const
{
    if (!isAvailable()) {
        return nil;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
        return commandBuffer;
    }
}

bool MetalContext::supportsMPS() const
{
    if (!isAvailable()) {
        return false;
    }

    // MPS is available on macOS 11.0+ (Big Sur) and iOS 14.0+
    // We're targeting these versions, so if device exists, MPS should be available
    return true;
}

bool MetalContext::supportsFamily(MTLGPUFamily family) const
{
    if (!isAvailable()) {
        return false;
    }

    @autoreleasepool {
        return [device_ supportsFamily:family];
    }
}

}} // namespace cv::dnn

#endif // HAVE_METAL
