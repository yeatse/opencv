// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_METAL_METAL_CONTEXT_HPP
#define OPENCV_DNN_SRC_METAL_METAL_CONTEXT_HPP

#ifdef HAVE_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace cv { namespace dnn {

/**
 * @brief Singleton class for managing Metal device and command queue
 *
 * This class provides a centralized interface for Metal resources
 * used throughout the DNN Metal backend.
 */
class MetalContext
{
public:
    /**
     * @brief Get the singleton instance
     */
    static MetalContext& getInstance();

    /**
     * @brief Check if Metal is available on this system
     */
    bool isAvailable() const;

    /**
     * @brief Get the Metal device
     */
    id<MTLDevice> getDevice() const;

    /**
     * @brief Get the Metal command queue
     */
    id<MTLCommandQueue> getCommandQueue() const;

    /**
     * @brief Create a new command buffer
     */
    id<MTLCommandBuffer> createCommandBuffer() const;

    /**
     * @brief Get device capabilities info
     */
    bool supportsMPS() const;
    bool supportsFamily(MTLGPUFamily family) const;

    // Prevent copying
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

private:
    MetalContext();
    ~MetalContext();

    bool initialize();

    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
    bool initialized_;
};

}} // namespace cv::dnn

#endif // HAVE_METAL
#endif // OPENCV_DNN_SRC_METAL_METAL_CONTEXT_HPP
