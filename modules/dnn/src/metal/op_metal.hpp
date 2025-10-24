// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_METAL_OP_METAL_HPP
#define OPENCV_DNN_SRC_METAL_OP_METAL_HPP

#ifdef HAVE_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "opencv2/dnn/dnn.hpp"
#include "metal_tensor.hpp"

namespace cv { namespace dnn {

/**
 * @brief Backend node for Metal operations
 *
 * Encapsulates Metal-specific operations for layer execution.
 * Each layer that supports Metal backend will create a MetalBackendNode
 * that contains the MPS operation or compute kernel.
 */
class MetalBackendNode : public BackendNode
{
public:
    MetalBackendNode();
    virtual ~MetalBackendNode();

    /**
     * @brief Execute the Metal operation
     * @param inputs Input Metal tensors
     * @param outputs Output Metal tensors
     * @param context Metal context for execution
     */
    virtual void execute(const std::vector<Ptr<MetalTensor>>& inputs,
                        const std::vector<Ptr<MetalTensor>>& outputs) = 0;
};

/**
 * @brief Backend wrapper for Metal tensors
 *
 * Wraps cv::Mat for Metal backend, managing CPU<->GPU data transfers
 * and providing access to underlying Metal buffers.
 */
class MetalBackendWrapper : public BackendWrapper
{
public:
    /**
     * @brief Create wrapper from cv::Mat
     * @param targetId Target identifier (DNN_TARGET_METAL)
     * @param m Input Mat to wrap
     */
    MetalBackendWrapper(int targetId, const Mat& m);

    /**
     * @brief Create wrapper from another wrapper (for reshaping)
     * @param base Base wrapper to reuse memory from
     * @param shape New shape for the tensor
     */
    MetalBackendWrapper(const Ptr<BackendWrapper>& base, const MatShape& shape);

    virtual ~MetalBackendWrapper();

    /**
     * @brief Copy data from GPU to CPU
     */
    virtual void copyToHost() CV_OVERRIDE;

    /**
     * @brief Mark that CPU data has been modified
     */
    virtual void setHostDirty() CV_OVERRIDE;

    /**
     * @brief Get the underlying Metal tensor
     */
    Ptr<MetalTensor> getMetalTensor() const { return metalTensor_; }

    /**
     * @brief Get the host Mat
     */
    Mat& getHostMat() { return hostMat_; }
    const Mat& getHostMat() const { return hostMat_; }

private:
    Mat hostMat_;                    // CPU side data
    Ptr<MetalTensor> metalTensor_;   // GPU side data
    bool hostDirty_;                 // True if CPU data is newer than GPU
    bool deviceDirty_;               // True if GPU data is newer than CPU
};

/**
 * @brief Check if Metal backend is available
 */
bool haveMetalSupport();

/**
 * @brief Helper to create Metal backend wrappers from input Mats
 */
void createMetalWrappers(const std::vector<Mat>& inputs,
                        std::vector<Ptr<BackendWrapper>>& wrappers);

}} // namespace cv::dnn

#endif // HAVE_METAL
#endif // OPENCV_DNN_SRC_METAL_OP_METAL_HPP
