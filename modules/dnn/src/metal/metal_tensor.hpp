// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_METAL_METAL_TENSOR_HPP
#define OPENCV_DNN_SRC_METAL_METAL_TENSOR_HPP

#ifdef HAVE_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "opencv2/core.hpp"

namespace cv { namespace dnn {

/**
 * @brief Wrapper class for managing OpenCV Mat data in Metal buffers
 *
 * This class handles:
 * - CPU to GPU memory transfers
 * - GPU to CPU memory transfers
 * - Metal buffer lifecycle management
 * - Tensor shape and layout information
 */
class MetalTensor
{
public:
    /**
     * @brief Construct MetalTensor from OpenCV Mat
     * @param mat Input OpenCV Mat (CV_32F type)
     * @param copyToDevice If true, immediately copy data to GPU
     */
    MetalTensor(const Mat& mat, bool copyToDevice = true);

    /**
     * @brief Destructor - releases Metal resources
     */
    ~MetalTensor();

    /**
     * @brief Copy data from CPU to GPU
     */
    bool copyToDevice();

    /**
     * @brief Copy data from GPU to CPU
     * @param dst Destination Mat to copy into
     */
    bool copyFromDevice(Mat& dst);

    /**
     * @brief Get the Metal buffer
     */
    id<MTLBuffer> getBuffer() const { return buffer_; }

    /**
     * @brief Get tensor dimensions
     */
    const MatShape& shape() const { return shape_; }

    /**
     * @brief Get total number of elements
     */
    size_t total() const { return total_; }

    /**
     * @brief Get size in bytes
     */
    size_t sizeInBytes() const { return total_ * sizeof(float); }

    /**
     * @brief Check if buffer is valid
     */
    bool isValid() const { return buffer_ != nil; }

private:
    Mat hostMat_;                  // Host side data (CPU)
    id<MTLBuffer> buffer_;         // Device side data (GPU)
    MatShape shape_;               // Tensor shape (e.g., [N, C, H, W])
    size_t total_;                 // Total number of elements

    bool allocateBuffer();
};

}} // namespace cv::dnn

#endif // HAVE_METAL
#endif // OPENCV_DNN_SRC_METAL_METAL_TENSOR_HPP
