// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_METAL_GRAPH_BUILDER_HPP__
#define __OPENCV_DNN_METAL_GRAPH_BUILDER_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"
#include <string>
#include <vector>

namespace cv { namespace dnn {

#ifdef HAVE_METAL

namespace metal {

// Opaque pointer types for Metal/MPSGraph objects
// These hide Objective-C types from C++ headers
typedef void* MPSGraphNetImplPtr;    // Opaque pointer to MPSGraphNetImpl*
typedef void* MPSGraphTensorPtr;     // Opaque pointer to MPSGraphTensor*

// Graph builder class for Metal backend
// Encapsulates graph building operations similar to ml::GraphBuilder in WebNN
class MetalGraphBuilder
{
public:
    MetalGraphBuilder(MPSGraphNetImplPtr graphImpl);

    // Operation builders (return MPSGraphTensorPtr)
    MPSGraphTensorPtr Relu(MPSGraphTensorPtr inputTensor, const std::string& name);
    MPSGraphTensorPtr Identity(MPSGraphTensorPtr inputTensor, const std::string& name);
    MPSGraphTensorPtr Add(MPSGraphTensorPtr tensor1, MPSGraphTensorPtr tensor2, const std::string& name);
    MPSGraphTensorPtr Mul(MPSGraphTensorPtr tensor1, MPSGraphTensorPtr tensor2, const std::string& name);
    MPSGraphTensorPtr Max(MPSGraphTensorPtr tensor1, MPSGraphTensorPtr tensor2, const std::string& name);
    MPSGraphTensorPtr Min(MPSGraphTensorPtr tensor1, MPSGraphTensorPtr tensor2, const std::string& name);
    MPSGraphTensorPtr Div(MPSGraphTensorPtr tensor1, MPSGraphTensorPtr tensor2, const std::string& name);
    MPSGraphTensorPtr Conv2d(MPSGraphTensorPtr inputTensor, MPSGraphTensorPtr weightsTensor, MPSGraphTensorPtr biasTensor,
                             const std::vector<int>& strides, const std::vector<int>& pads_begin, const std::vector<int>& pads_end,
                             const std::vector<int>& dilations, int groups, const std::string& name);

    // Tensor management
    MPSGraphTensorPtr GetTensor(const std::string& name);
    void AddTensor(const std::string& name, MPSGraphTensorPtr tensor);

    // Constant tensor creation
    MPSGraphTensorPtr Constant(const cv::Mat& data, const std::string& name);

private:
    MPSGraphNetImplPtr impl;
};

}  // namespace metal

#endif  // HAVE_METAL

}}  // namespace cv::dnn

#endif  // __OPENCV_DNN_METAL_GRAPH_BUILDER_HPP__
