#include "precomp.hpp"
#include "op_metal.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {

#ifdef HAVE_METAL

CV__DNN_INLINE_NS_BEGIN

void Net::Impl::initMetalBackend()
{
}

CV__DNN_INLINE_NS_END

MetalBackendNode::MetalBackendNode()
    : BackendNode(DNN_BACKEND_METAL)
{
}

MetalBackendWrapper::MetalBackendWrapper(const Mat& m)
    : BackendWrapper(DNN_BACKEND_METAL, DNN_TARGET_METAL)
{
}

MetalBackendWrapper::MetalBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, const Mat& m)
    : BackendWrapper(DNN_BACKEND_METAL, DNN_TARGET_METAL)
{
}

MetalBackendWrapper::~MetalBackendWrapper()
{
}

void MetalBackendWrapper::copyToHost()
{
}

void MetalBackendWrapper::setHostDirty()
{
}

#else

#endif // HAVE_METAL

void forwardMetal(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, Ptr<BackendNode>& node)
{
}

}}  // namespace cv::dnn