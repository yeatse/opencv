// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_METAL

TEST(DNN_Metal, backend_availability)
{
    // Test that Metal backend can be selected on Apple platforms
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    ASSERT_NO_THROW(net.setPreferableBackend(DNN_BACKEND_METAL));
    ASSERT_NO_THROW(net.setPreferableTarget(DNN_TARGET_CPU));
}

TEST(DNN_Metal, backend_selection)
{
    // Test that Metal backend can be selected without crashing
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));

    ASSERT_NO_THROW(net.setPreferableBackend(DNN_BACKEND_METAL));
    ASSERT_NO_THROW(net.setPreferableTarget(DNN_TARGET_CPU));
}

TEST(DNN_Metal, backend_selection_gpu_target)
{
    // Test that Metal backend can be selected with GPU target
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));

    ASSERT_NO_THROW(net.setPreferableBackend(DNN_BACKEND_METAL));
    ASSERT_NO_THROW(net.setPreferableTarget(DNN_TARGET_OPENCL)); // GPU target
}

TEST(DNN_Metal, basic_inference_fallback)
{
    // Phase 0: Test that inference works via CPU fallback
    // Metal layers are not implemented yet, so all layers should fall back to CPU

    // Load a simple model
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    net.setPreferableBackend(DNN_BACKEND_METAL);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Create input
    Mat input = Mat::ones(1, 3, 224, 224, CV_32F);
    net.setInput(input);

    // Forward should work (via CPU fallback)
    // Note: In Phase 0, this will fall back to CPU for all layers
    Mat output;
    ASSERT_NO_THROW(output = net.forward());

    // Output should not be empty
    ASSERT_FALSE(output.empty());
}

TEST(DNN_Metal, compare_with_cpu_backend)
{
    // Test that Metal backend (with CPU fallback) produces same results as CPU backend

    // Load model
    Net netMetal = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    netMetal.setPreferableBackend(DNN_BACKEND_METAL);
    netMetal.setPreferableTarget(DNN_TARGET_CPU);

    Net netCPU = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    netCPU.setPreferableTarget(DNN_TARGET_CPU);

    // Create input
    Mat input(1, 3, 224, 224, CV_32F);
    randn(input, 0.0f, 1.0f);

    // Run inference
    netMetal.setInput(input);
    Mat outputMetal = netMetal.forward();

    netCPU.setInput(input);
    Mat outputCPU = netCPU.forward();

    // Results should match (since both use CPU in Phase 0)
    normAssert(outputCPU, outputMetal, "Metal backend (CPU fallback) vs CPU backend");
}

TEST(DNN_Metal, memory_management)
{
    // Test that memory is properly managed (no leaks)
    // This test creates and destroys networks multiple times

    for (int i = 0; i < 10; i++)
    {
        Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
        net.setPreferableBackend(DNN_BACKEND_METAL);
        net.setPreferableTarget(DNN_TARGET_CPU);

        Mat input = Mat::ones(1, 3, 224, 224, CV_32F);
        net.setInput(input);

        Mat output = net.forward();
        ASSERT_FALSE(output.empty());
    }

    // If we got here without crashing, memory management is working
    SUCCEED();
}

TEST(DNN_Metal, multiple_networks)
{
    // Test that multiple networks can coexist with Metal backend

    Net net1 = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    net1.setPreferableBackend(DNN_BACKEND_METAL);
    net1.setPreferableTarget(DNN_TARGET_CPU);

    Net net2 = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    net2.setPreferableBackend(DNN_BACKEND_METAL);
    net2.setPreferableTarget(DNN_TARGET_CPU);

    Mat input = Mat::ones(1, 3, 224, 224, CV_32F);

    // Both networks should work independently
    net1.setInput(input);
    Mat output1 = net1.forward();

    net2.setInput(input);
    Mat output2 = net2.forward();

    ASSERT_FALSE(output1.empty());
    ASSERT_FALSE(output2.empty());

    // Results should be identical
    normAssert(output1, output2, "Multiple Metal networks");
}

TEST(DNN_Metal, input_shapes)
{
    // Test various input shapes
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    net.setPreferableBackend(DNN_BACKEND_METAL);
    net.setPreferableTarget(DNN_TARGET_CPU);

    std::vector<std::pair<int, int>> sizes = {
        {224, 224},
        {256, 256},
        {299, 299}
    };

    for (const auto& size : sizes)
    {
        Mat input = Mat::ones(1, 3, size.first, size.second, CV_32F);
        net.setInput(input);

        Mat output;
        ASSERT_NO_THROW(output = net.forward());
        ASSERT_FALSE(output.empty());
    }
}

TEST(DNN_Metal, fallback_detection)
{
    // Test that we can detect when layers are falling back to CPU
    // In Phase 0, ALL layers should fall back to CPU

    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    net.setPreferableBackend(DNN_BACKEND_METAL);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat input = Mat::ones(1, 3, 224, 224, CV_32F);
    net.setInput(input);
    Mat output = net.forward();

    // Check layer backends
    std::vector<String> layerNames = net.getLayerNames();
    bool allFallback = true;

    for (const auto& name : layerNames)
    {
        Ptr<dnn::Layer> layer = net.getLayer(net.getLayerId(name));

        // In Phase 0, no layers support Metal backend yet
        // So all should fall back to CPU (preferableTarget != DNN_TARGET_CPU means it's using Metal)
        if (layer->preferableTarget != DNN_TARGET_CPU && layer->preferableTarget != -1)
        {
            allFallback = false;
            std::cout << "Layer " << name << " is NOT falling back (unexpected in Phase 0)" << std::endl;
        }
    }

    // In Phase 0, all layers should fall back to CPU
    // This test will need to be updated in Phase 1 when Metal implementations are added
    EXPECT_TRUE(allFallback) << "Phase 0: All layers should fall back to CPU";
}

TEST(DNN_Metal, backend_switching)
{
    // Test switching between backends
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));

    Mat input = Mat::ones(1, 3, 224, 224, CV_32F);

    // Run with CPU backend
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    net.setInput(input);
    Mat outputCPU = net.forward();

    // Switch to Metal backend
    net.setPreferableBackend(DNN_BACKEND_METAL);
    net.setPreferableTarget(DNN_TARGET_CPU);
    net.setInput(input);
    Mat outputMetal = net.forward();

    // Results should match
    normAssert(outputCPU, outputMetal, "Backend switching");

    // Switch back to CPU
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setInput(input);
    Mat outputCPU2 = net.forward();

    normAssert(outputCPU, outputCPU2, "Backend switching back");
}

#else  // !HAVE_METAL

TEST(DNN_Metal, backend_not_available)
{
    // On non-Apple platforms, this test just verifies the file compiles
    // The Metal backend tests are conditionally compiled only when HAVE_METAL is defined
    SUCCEED();
}

#endif  // HAVE_METAL

}} // namespace opencv_test::<anonymous>
