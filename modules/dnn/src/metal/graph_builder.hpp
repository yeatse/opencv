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

// Graph builder class for Metal backend
// Encapsulates graph building operations similar to ml::GraphBuilder in WebNN
class MetalGraphBuilder
{
public:
    MetalGraphBuilder(void* graphImpl);  // MPSGraphNetImpl* (opaque)

    // Operation builders (return MPSGraphTensor* as void*)
    void* Relu(void* inputTensor, const std::string& name);
    void* Add(void* tensor1, void* tensor2, const std::string& name);
    void* Mul(void* tensor1, void* tensor2, const std::string& name);
    void* Max(void* tensor1, void* tensor2, const std::string& name);
    void* Min(void* tensor1, void* tensor2, const std::string& name);
    void* Div(void* tensor1, void* tensor2, const std::string& name);
    void* Conv2d(void* inputTensor, void* weightsTensor, void* biasTensor,
                 const std::vector<int>& strides, const std::vector<int>& paddings,
                 const std::vector<int>& dilations, int groups, const std::string& name);

    // Tensor management
    void* GetTensor(const std::string& name);
    void AddTensor(const std::string& name, void* tensor);

    // Constant tensor creation
    void* Constant(const cv::Mat& data, const std::string& name);

private:
    void* impl;  // MPSGraphNetImpl* (opaque pointer)
};

#endif  // HAVE_METAL

}}  // namespace cv::dnn

#endif  // __OPENCV_DNN_METAL_GRAPH_BUILDER_HPP__
