// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_METAL_OPS_MPS_ACTIVATION_HPP
#define OPENCV_DNN_SRC_METAL_OPS_MPS_ACTIVATION_HPP

#ifdef HAVE_METAL

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../op_metal.hpp"

namespace cv { namespace dnn {

/**
 * @brief Metal backend node for ReLU activation
 *
 * Implements ReLU (Rectified Linear Unit) activation function:
 * f(x) = max(0, x)
 *
 * Uses Metal Performance Shaders MPSCNNNeuronReLU kernel
 */
class MetalReLUNode : public MetalBackendNode
{
public:
    /**
     * @brief Constructor
     * @param slope Slope for negative values (0 for standard ReLU, >0 for Leaky ReLU)
     */
    explicit MetalReLUNode(float slope = 0.0f);

    virtual ~MetalReLUNode();

    /**
     * @brief Execute ReLU operation
     */
    virtual void execute(const std::vector<Ptr<MetalTensor>>& inputs,
                        const std::vector<Ptr<MetalTensor>>& outputs) CV_OVERRIDE;

private:
    float slope_;
    MPSCNNNeuronReLU* reluKernel_;
};

/**
 * @brief Create a Metal ReLU backend node
 * @param slope Slope for negative values (0 for standard ReLU)
 */
Ptr<BackendNode> createMetalReLUNode(float slope = 0.0f);

}} // namespace cv::dnn

#endif // HAVE_METAL
#endif // OPENCV_DNN_SRC_METAL_OPS_MPS_ACTIVATION_HPP
