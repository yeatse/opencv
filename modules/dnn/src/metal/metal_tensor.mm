// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_tensor.hpp"
#include "metal_context.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#include "opencv2/core/utils/logger.hpp"
#include <string.h>

namespace cv { namespace dnn {

MetalTensor::MetalTensor(const Mat& mat, bool copyToDevice)
    : hostMat_(mat)
    , buffer_(nil)
    , total_(0)
{
    CV_Assert(!mat.empty());
    CV_Assert(mat.depth() == CV_32F); // Only support float32 for now

    // Store shape
    shape_ = shape(mat);
    total_ = mat.total();

    // Allocate Metal buffer
    if (allocateBuffer())
    {
        // Optionally copy to device immediately
        if (copyToDevice)
        {
            this->copyToDevice();
        }
    }
}

MetalTensor::~MetalTensor()
{
    @autoreleasepool {
        if (buffer_) {
            [buffer_ release];
            buffer_ = nil;
        }
    }
}

bool MetalTensor::allocateBuffer()
{
    MetalContext& ctx = MetalContext::getInstance();

    if (!ctx.isAvailable())
    {
        CV_LOG_ERROR(NULL, "DNN/Metal: Metal context not available");
        return false;
    }

    @autoreleasepool {
        id<MTLDevice> device = ctx.getDevice();

        size_t bufferSize = sizeInBytes();
        buffer_ = [device newBufferWithLength:bufferSize
                                      options:MTLResourceStorageModeShared];

        if (!buffer_) {
            CV_LOG_ERROR(NULL, "DNN/Metal: Failed to allocate buffer of size " << bufferSize);
            return false;
        }

        [buffer_ retain];
        return true;
    }
}

bool MetalTensor::copyToDevice()
{
    if (!buffer_) {
        CV_LOG_ERROR(NULL, "DNN/Metal: Cannot copy to device - buffer not allocated");
        return false;
    }

    if (hostMat_.empty()) {
        CV_LOG_ERROR(NULL, "DNN/Metal: Cannot copy to device - host mat is empty");
        return false;
    }

    @autoreleasepool {
        // Ensure host mat is continuous
        Mat continuous = hostMat_;
        if (!hostMat_.isContinuous()) {
            continuous = hostMat_.clone();
        }

        // Copy data to Metal buffer
        void* bufferPointer = [buffer_ contents];
        memcpy(bufferPointer, continuous.data, sizeInBytes());

        return true;
    }
}

bool MetalTensor::copyFromDevice(Mat& dst)
{
    if (!buffer_) {
        CV_LOG_ERROR(NULL, "DNN/Metal: Cannot copy from device - buffer not allocated");
        return false;
    }

    @autoreleasepool {
        // Ensure destination mat has correct size and type
        if (dst.size != hostMat_.size || dst.type() != hostMat_.type()) {
            dst.create(shape_, CV_32F);
        }

        // Ensure destination is continuous
        CV_Assert(dst.isContinuous());

        // Copy data from Metal buffer
        void* bufferPointer = [buffer_ contents];
        memcpy(dst.data, bufferPointer, sizeInBytes());

        return true;
    }
}

}} // namespace cv::dnn

#endif // HAVE_METAL
