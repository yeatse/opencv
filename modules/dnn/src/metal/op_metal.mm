// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "op_metal.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#include "metal_context.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace cv { namespace dnn {

// MetalBackendNode implementation

MetalBackendNode::MetalBackendNode()
    : BackendNode(DNN_BACKEND_METAL)
{
}

MetalBackendNode::~MetalBackendNode()
{
}

// MetalBackendWrapper implementation

MetalBackendWrapper::MetalBackendWrapper(int targetId, const Mat& m)
    : BackendWrapper(DNN_BACKEND_METAL, targetId)
    , hostMat_(m)
    , hostDirty_(true)
    , deviceDirty_(false)
{
    // Create Metal tensor (but don't copy data yet)
    metalTensor_ = makePtr<MetalTensor>(m, false);
}

MetalBackendWrapper::MetalBackendWrapper(const Ptr<BackendWrapper>& base, const MatShape& shape)
    : BackendWrapper(DNN_BACKEND_METAL, base->targetId)
    , hostDirty_(false)
    , deviceDirty_(false)
{
    // Reuse memory from base wrapper
    Ptr<MetalBackendWrapper> metalBase = base.dynamicCast<MetalBackendWrapper>();
    CV_Assert(!metalBase.empty());

    // Get the host mat from base and reshape it
    hostMat_ = metalBase->getHostMat();
    hostMat_ = hostMat_.reshape(1, shape);

    // Share the Metal tensor (same underlying buffer, different shape)
    metalTensor_ = metalBase->getMetalTensor();
}

MetalBackendWrapper::~MetalBackendWrapper()
{
}

void MetalBackendWrapper::copyToHost()
{
    if (deviceDirty_ && metalTensor_)
    {
        metalTensor_->copyFromDevice(hostMat_);
        deviceDirty_ = false;
        hostDirty_ = false;
    }
}

void MetalBackendWrapper::setHostDirty()
{
    hostDirty_ = true;
    deviceDirty_ = false;

    // Copy updated data to device
    if (metalTensor_ && !hostMat_.empty())
    {
        metalTensor_->copyToDevice();
        hostDirty_ = false;
    }
}

// Helper functions

bool haveMetalSupport()
{
    MetalContext& ctx = MetalContext::getInstance();
    return ctx.isAvailable() && ctx.supportsMPS();
}

void createMetalWrappers(const std::vector<Mat>& inputs,
                        std::vector<Ptr<BackendWrapper>>& wrappers)
{
    wrappers.clear();
    wrappers.reserve(inputs.size());

    for (const Mat& m : inputs)
    {
        Ptr<BackendWrapper> wrapper = makePtr<MetalBackendWrapper>(DNN_TARGET_METAL, m);
        wrappers.push_back(wrapper);
    }
}

}} // namespace cv::dnn

#endif // HAVE_METAL
