// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "op_metal.hpp"

#ifdef HAVE_METAL

namespace cv { namespace dnn {

// Helper to add outputs when layer doesn't support Metal
void Net::Impl::addMetalOutputs(LayerData& ld)
{
    Ptr<MetalBackendNode> metalNode = ld.backendNodes[DNN_BACKEND_METAL].dynamicCast<MetalBackendNode>();
    if (!metalNode.empty() && !metalNode->net.empty())
    {
        metalNode->net->addOutput(ld.name);
    }
}

// Initialize Metal backend for the network
void Net::Impl::initMetalBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    CV_Assert_N(preferableBackend == DNN_BACKEND_METAL, haveMetal());

    Ptr<MetalNet> net;

    // First pass: Set wrapper names
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;
        if (ld.id == 0)
        {
            CV_Assert((netInputLayer->outNames.empty() && ld.outputBlobsWrappers.size() == 1) ||
                      (netInputLayer->outNames.size() == ld.outputBlobsWrappers.size()));
            for (size_t i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                Ptr<MetalBackendWrapper> wrapper = ld.outputBlobsWrappers[i].dynamicCast<MetalBackendWrapper>();
                if (wrapper.empty()) continue;
                std::string outputName = netInputLayer->outNames.empty() ? ld.name : netInputLayer->outNames[i];
                outputName = ld.outputBlobsWrappers.size() > 1 ? (outputName + "." + std::to_string(i)) : outputName;
                wrapper->name = outputName;
            }
        }
        else
        {
            for (size_t i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                Ptr<MetalBackendWrapper> wrapper = ld.outputBlobsWrappers[i].dynamicCast<MetalBackendWrapper>();
                if (wrapper.empty()) continue;
                std::string outputName = ld.outputBlobsWrappers.size() > 1 ? (ld.name + "." + std::to_string(i)) : ld.name;
                wrapper->name = outputName;
            }
        }
    }

    // Second pass: Build Metal graphs
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;

        if (ld.id == 0 && ld.skip)
            continue;

        bool fused = ld.skip;
        Ptr<Layer> layer = ld.layerInstance;
        if (!fused && !layer->supportBackend(preferableBackend))
        {
            CV_LOG_WARNING(NULL, "Layer " + ld.type + " name " + ld.name + " is unsupported by Metal backend, falling back to CPU");

            addMetalOutputs(ld);
            net = Ptr<MetalNet>();
            layer->preferableTarget = DNN_TARGET_CPU;

            // Mark input nodes as unconnected
            for (size_t i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
                Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
                if (!inpNode.empty()) {
                    Ptr<MetalBackendNode> metalNode = inpNode.dynamicCast<MetalBackendNode>();
                    if (!metalNode.empty() && !metalNode->net.empty()) {
                        metalNode->net->addOutput(metalNode->name);
                    }
                }
            }
            continue;
        }
        ld.skip = true; // Initially skip all Metal supported layers

        // Collect input nodes
        std::vector<Ptr<BackendNode>> inputNodes;
        for (size_t i = 0; i < ld.inputBlobsId.size(); ++i)
        {
            if (inputNodes.size() == ld.inputBlobsId.size()) break;

            LayerData &inpLd = layers[ld.inputBlobsId[i].lid];
            Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
            if (!inpNode.empty())
            {
                Ptr<MetalBackendNode> metalInpNode = inpNode.dynamicCast<MetalBackendNode>();
                if (!metalInpNode.empty() && !metalInpNode->net.empty())
                {
                    if (metalInpNode->net == net && !fused) {
                        inputNodes.push_back(inpNode);
                        continue;
                    }
                }
            }

            // Create new network if needed
            if (net.empty()) {
                net = Ptr<MetalNet>(new MetalNet());
                net->init(preferableTarget);
            }

            if (!fused) {
                std::vector<std::string> inputNames;
                std::vector<cv::Mat> inputs;

                // Find this layer in the consumers
                auto curr_pos = inpLd.consumers.begin();
                auto compare = [&ld] (const LayerPin& lp) { return lp.lid == ld.id; };
                auto cons = curr_pos;
                while ((cons = std::find_if(curr_pos, inpLd.consumers.end(), compare)) != inpLd.consumers.end())
                {
                    int cons_inp = cons->oid;
                    Ptr<MetalBackendWrapper> inpWrapper = inpLd.outputBlobsWrappers[cons_inp].dynamicCast<MetalBackendWrapper>();
                    if (!inpWrapper.empty())
                    {
                        auto iter = std::find(inputNames.begin(), inputNames.end(), inpWrapper->name);
                        if (iter == inputNames.end()) {
                            inputNames.push_back(inpWrapper->name);
                            inputs.push_back(inpLd.outputBlobs[cons_inp]);
                        }
                    }
                    curr_pos = cons + 1;
                }

                auto inps = net->setInputs(inputs, inputNames);
                for (auto& inp : inps) {
                    MetalBackendNode* node = new MetalBackendNode(inp);
                    node->net = net;
                    inputNodes.emplace_back(Ptr<BackendNode>(node));
                }

                // Share device from network to input wrappers
                void* sharedDevice = net->getDevice();
                if (sharedDevice) {
                    for (size_t i = 0; i < inpLd.outputBlobsWrappers.size(); ++i) {
                        Ptr<MetalBackendWrapper> metalWrapper = inpLd.outputBlobsWrappers[i].dynamicCast<MetalBackendWrapper>();
                        if (!metalWrapper.empty()) {
                            metalWrapper->setDevice(sharedDevice);
                        }
                    }
                }
            }
        }

        Ptr<BackendNode> node;
        if (!net.empty())
        {
            if (fused)
            {
                bool inPlace = ld.inputBlobsId.size() == 1 && ld.outputBlobs.size() == 1 &&
                               ld.inputBlobs[0]->data == ld.outputBlobs[0].data;
                CV_Assert(inPlace);
                node = layers[ld.inputBlobsId[0].lid].backendNodes[preferableBackend];
                ld.inputBlobsWrappers = layers[ld.inputBlobsId[0].lid].inputBlobsWrappers;
            }
        }
        else {
            net = Ptr<MetalNet>(new MetalNet());
            net->init(preferableTarget);
        }

        if (!fused)
        {
            CV_Assert(ld.inputBlobsId.size() == inputNodes.size());
            if (inputNodes.size())
            {
                // Call layer's initMetal to add operations to the graph
                try
                {
                    node = layer->initMetal(ld.inputBlobsWrappers, inputNodes);
                }
                catch (const cv::Exception&)
                {
                    CV_LOG_WARNING(NULL, "Layer " + ld.name + " initMetal failed, falling back to CPU");
                    layer->preferableTarget = DNN_TARGET_CPU;
                    addMetalOutputs(ld);
                    net = Ptr<MetalNet>();
                    continue;
                }
            }
        }

        if (!node.empty())
        {
            ld.backendNodes[DNN_BACKEND_METAL] = node;
            Ptr<MetalBackendNode> metalNode = node.dynamicCast<MetalBackendNode>();
            CV_Assert(!metalNode.empty());
            metalNode->name = ld.name;
            metalNode->net = net;

            // Add to Metal network's blob management
            net->addBlobs(ld.outputBlobsWrappers);

            // Share device from network to all wrappers
            void* sharedDevice = net->getDevice();
            if (sharedDevice) {
                for (auto& wrapper : ld.outputBlobsWrappers) {
                    Ptr<MetalBackendWrapper> metalWrapper = wrapper.dynamicCast<MetalBackendWrapper>();
                    if (!metalWrapper.empty()) {
                        metalWrapper->setDevice(sharedDevice);
                    }
                }
            }
        }
    }

    // Finalize all Metal networks
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData &ld = it->second;
        Ptr<BackendNode> node = ld.backendNodes[DNN_BACKEND_METAL];
        if (!node.empty())
        {
            Ptr<MetalBackendNode> metalNode = node.dynamicCast<MetalBackendNode>();
            if (!metalNode.empty() && !metalNode->net.empty())
            {
                metalNode->net->createGraph(preferableTarget);
            }
        }
    }
}

}}  // namespace cv::dnn

#endif  // HAVE_METAL
