// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_OP_METAL_HPP__
#define __OPENCV_DNN_OP_METAL_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

namespace cv { namespace dnn {

constexpr bool haveMetal() {
#ifdef HAVE_METAL
        return true;
#else
        return false;
#endif
}

#ifdef HAVE_METAL

class MetalBackendNode;
class MetalBackendWrapper;

class MetalBackendNode : public BackendNode
{
public:
    MetalBackendNode();
};

class MetalBackendWrapper : public BackendWrapper
{
public:
    MetalBackendWrapper(const Mat& m);
    MetalBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, const Mat& m);
    ~MetalBackendWrapper() CV_OVERRIDE;

    virtual void copyToHost() CV_OVERRIDE;
    virtual void setHostDirty() CV_OVERRIDE;
};

#endif  // HAVE_METAL

void forwardMetal(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, Ptr<BackendNode>& node);


}}  // namespace cv::dnn

#endif  // __OPENCV_DNN_OP_METAL_HPP__
