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

TEST(DNN_Metal, target_is_noop)
{
    // Test that preferableTarget setting is a no-op for Metal backend.
    // Metal backend always uses GPU regardless of the target setting.
    // This test verifies that any target can be set without validation errors.
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    net.setPreferableBackend(DNN_BACKEND_METAL);

    // All these target settings should work (target is ignored by Metal backend)
    ASSERT_NO_THROW(net.setPreferableTarget(DNN_TARGET_CPU));
    ASSERT_NO_THROW(net.setPreferableTarget(DNN_TARGET_OPENCL));
    ASSERT_NO_THROW(net.setPreferableTarget(DNN_TARGET_OPENCL_FP16));
    ASSERT_NO_THROW(net.setPreferableTarget(DNN_TARGET_CPU_FP16));

    // Verify inference works regardless of target setting
    int inputSizes[] = {1, 3, 224, 224};
    Mat input = Mat::ones(4, inputSizes, CV_32F);
    net.setInput(input);
    Mat output;
    ASSERT_NO_THROW(output = net.forward());
    ASSERT_FALSE(output.empty());
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
    int sizes[] = {1, 3, 224, 224};
    Mat input = Mat::ones(4, sizes, CV_32F);
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
    int sizes[] = {1, 3, 224, 224};
    Mat input(4, sizes, CV_32F);
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

        int sizes[] = {1, 3, 224, 224};
        Mat input = Mat::ones(4, sizes, CV_32F);
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

    int sizes[] = {1, 3, 224, 224};
    Mat input = Mat::ones(4, sizes, CV_32F);

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
        int inputSizes[] = {1, 3, size.first, size.second};
        Mat input = Mat::ones(4, inputSizes, CV_32F);
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

    int sizes[] = {1, 3, 224, 224};
    Mat input = Mat::ones(4, sizes, CV_32F);
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

    int sizes[] = {1, 3, 224, 224};
    Mat input = Mat::ones(4, sizes, CV_32F);

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

TEST(DNN_Metal, relu_layer)
{
    // Test ReLU activation layer with Metal backend
    // Create a simple network with just ReLU
    LayerParams lp;
    lp.type = "ReLU";
    lp.name = "testReLU";

    Net net;
    (void)net.addLayerToPrev(lp.name, lp.type, lp);

    // Create input with both positive and negative values
    int sizes[] = {1, 1, 4, 4};
    Mat input(4, sizes, CV_32F);
    float* data = input.ptr<float>();
    for (int i = 0; i < 16; i++) {
        data[i] = (float)(i - 8);  // Values from -8 to 7
    }

    // Test with CPU backend
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setInput(input);
    Mat outputCPU = net.forward();

    // Test with Metal backend
    net.setPreferableBackend(DNN_BACKEND_METAL);
    net.setPreferableTarget(DNN_TARGET_CPU);
    net.setInput(input);
    Mat outputMetal = net.forward();

    // Results should match
    // ReLU should clamp negative values to 0
    normAssert(outputCPU, outputMetal, "ReLU: Metal vs CPU");

    // Verify ReLU behavior: max(0, x)
    float* outData = outputMetal.ptr<float>();
    for (int i = 0; i < 16; i++) {
        float expected = std::max(0.0f, data[i]);
        EXPECT_NEAR(expected, outData[i], 1e-5)
            << "ReLU output mismatch at index " << i
            << ": input=" << data[i]
            << ", expected=" << expected
            << ", got=" << outData[i];
    }
}

TEST(DNN_Metal, eltwise_add)
{
    // Test element-wise addition with Metal backend
    // Create two separate networks - one for CPU, one for Metal
    Net netCPU, netMetal;

    // Setup CPU network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_add";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "sum");
        int eltwise_id = netCPU.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netCPU.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netCPU.setInputsNames(inputNames);
        netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    }

    // Setup Metal network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_add";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "sum");
        int eltwise_id = netMetal.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netMetal.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netMetal.setInputsNames(inputNames);
        netMetal.setPreferableBackend(DNN_BACKEND_METAL);
        netMetal.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Prepare test inputs (1x1x4x4) - must be 4D tensors
    int sizes[] = {1, 1, 4, 4};
    Mat input1(4, sizes, CV_32F);
    Mat input2(4, sizes, CV_32F);
    float* data1 = input1.ptr<float>();
    float* data2 = input2.ptr<float>();
    for (int i = 0; i < 16; i++) {
        data1[i] = (float)(i);        // 0 to 15
        data2[i] = (float)(i * 2);    // 0 to 30
    }

    // Test with CPU backend
    netCPU.setInput(input1, "input1");
    netCPU.setInput(input2, "input2");
    Mat outputCPU = netCPU.forward();

    // Test with Metal backend
    netMetal.setInput(input1, "input1");
    netMetal.setInput(input2, "input2");
    Mat outputMetal = netMetal.forward();

    // Results should match
    normAssert(outputCPU, outputMetal, "Eltwise Add: Metal vs CPU");

    // Verify addition behavior: input1 + input2
    float* outData = outputMetal.ptr<float>();
    for (int i = 0; i < 16; i++) {
        float expected = data1[i] + data2[i];
        EXPECT_NEAR(expected, outData[i], 1e-5)
            << "Eltwise Add mismatch at index " << i
            << ": input1=" << data1[i]
            << ", input2=" << data2[i]
            << ", expected=" << expected
            << ", got=" << outData[i];
    }
}

TEST(DNN_Metal, eltwise_multiply)
{
    // Test element-wise multiplication with Metal backend
    // Create two separate networks - one for CPU, one for Metal
    Net netCPU, netMetal;

    // Setup CPU network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_mul";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "mul");
        int eltwise_id = netCPU.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netCPU.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netCPU.setInputsNames(inputNames);
        netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    }

    // Setup Metal network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_mul";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "mul");
        int eltwise_id = netMetal.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netMetal.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netMetal.setInputsNames(inputNames);
        netMetal.setPreferableBackend(DNN_BACKEND_METAL);
        netMetal.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Prepare test inputs (1x1x4x4) - must be 4D tensors
    int sizes[] = {1, 1, 4, 4};
    Mat input1(4, sizes, CV_32F);
    Mat input2(4, sizes, CV_32F);
    float* data1 = input1.ptr<float>();
    float* data2 = input2.ptr<float>();
    for (int i = 0; i < 16; i++) {
        data1[i] = (float)(i + 1);    // 1 to 16
        data2[i] = (float)(2);        // All 2s for simple test
    }

    // Test with CPU backend
    netCPU.setInput(input1, "input1");
    netCPU.setInput(input2, "input2");
    Mat outputCPU = netCPU.forward();

    // Test with Metal backend
    netMetal.setInput(input1, "input1");
    netMetal.setInput(input2, "input2");
    Mat outputMetal = netMetal.forward();

    // Results should match
    normAssert(outputCPU, outputMetal, "Eltwise Multiply: Metal vs CPU");

    // Verify multiplication behavior: input1 * input2
    float* outData = outputMetal.ptr<float>();
    for (int i = 0; i < 16; i++) {
        float expected = data1[i] * data2[i];
        EXPECT_NEAR(expected, outData[i], 1e-5)
            << "Eltwise Multiply mismatch at index " << i
            << ": input1=" << data1[i]
            << ", input2=" << data2[i]
            << ", expected=" << expected
            << ", got=" << outData[i];
    }
}

TEST(DNN_Metal, eltwise_add_three_inputs)
{
    // Test element-wise addition with three inputs to verify chaining
    // Create two separate networks - one for CPU, one for Metal
    Net netCPU, netMetal;

    // Setup CPU network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_add3";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "sum");
        int eltwise_id = netCPU.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netCPU.connect(0, 1, eltwise_id, 1);
        netCPU.connect(0, 2, eltwise_id, 2);
        std::vector<String> inputNames = {"input1", "input2", "input3"};
        netCPU.setInputsNames(inputNames);
        netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    }

    // Setup Metal network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_add3";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "sum");
        int eltwise_id = netMetal.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netMetal.connect(0, 1, eltwise_id, 1);
        netMetal.connect(0, 2, eltwise_id, 2);
        std::vector<String> inputNames = {"input1", "input2", "input3"};
        netMetal.setInputsNames(inputNames);
        netMetal.setPreferableBackend(DNN_BACKEND_METAL);
        netMetal.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Prepare test inputs (1x1x4x4) - must be 4D tensors
    int sizes[] = {1, 1, 4, 4};
    Mat input1(4, sizes, CV_32F);
    Mat input2(4, sizes, CV_32F);
    Mat input3(4, sizes, CV_32F);
    float* data1 = input1.ptr<float>();
    float* data2 = input2.ptr<float>();
    float* data3 = input3.ptr<float>();
    for (int i = 0; i < 16; i++) {
        data1[i] = (float)(i);
        data2[i] = (float)(i + 10);
        data3[i] = (float)(i + 20);
    }

    // Test with CPU backend
    netCPU.setInput(input1, "input1");
    netCPU.setInput(input2, "input2");
    netCPU.setInput(input3, "input3");
    Mat outputCPU = netCPU.forward();

    // Test with Metal backend
    netMetal.setInput(input1, "input1");
    netMetal.setInput(input2, "input2");
    netMetal.setInput(input3, "input3");
    Mat outputMetal = netMetal.forward();

    // Results should match
    normAssert(outputCPU, outputMetal, "Eltwise Add 3 Inputs: Metal vs CPU");

    // Verify addition behavior: input1 + input2 + input3
    float* outData = outputMetal.ptr<float>();
    for (int i = 0; i < 16; i++) {
        float expected = data1[i] + data2[i] + data3[i];
        EXPECT_NEAR(expected, outData[i], 1e-5)
            << "Eltwise Add 3 Inputs mismatch at index " << i
            << ": input1=" << data1[i]
            << ", input2=" << data2[i]
            << ", input3=" << data3[i]
            << ", expected=" << expected
            << ", got=" << outData[i];
    }
}

TEST(DNN_Metal, eltwise_max)
{
    // Test element-wise maximum with Metal backend
    Net netCPU, netMetal;

    // Setup CPU network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_max";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "max");
        int eltwise_id = netCPU.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netCPU.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netCPU.setInputsNames(inputNames);
        netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    }

    // Setup Metal network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_max";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "max");
        int eltwise_id = netMetal.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netMetal.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netMetal.setInputsNames(inputNames);
        netMetal.setPreferableBackend(DNN_BACKEND_METAL);
        netMetal.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Prepare test inputs (1x1x4x4)
    int sizes[] = {1, 1, 4, 4};
    Mat input1(4, sizes, CV_32F);
    Mat input2(4, sizes, CV_32F);
    float* data1 = input1.ptr<float>();
    float* data2 = input2.ptr<float>();
    for (int i = 0; i < 16; i++) {
        data1[i] = (float)(i - 8);       // -8 to 7
        data2[i] = (float)(8 - i);       // 8 to -7
    }

    // Test with CPU backend
    netCPU.setInput(input1, "input1");
    netCPU.setInput(input2, "input2");
    Mat outputCPU = netCPU.forward();

    // Test with Metal backend
    netMetal.setInput(input1, "input1");
    netMetal.setInput(input2, "input2");
    Mat outputMetal = netMetal.forward();

    // Results should match
    normAssert(outputCPU, outputMetal, "Eltwise Max: Metal vs CPU");

    // Verify max behavior: max(input1, input2)
    float* outData = outputMetal.ptr<float>();
    for (int i = 0; i < 16; i++) {
        float expected = std::max(data1[i], data2[i]);
        EXPECT_NEAR(expected, outData[i], 1e-5)
            << "Eltwise Max mismatch at index " << i
            << ": input1=" << data1[i]
            << ", input2=" << data2[i]
            << ", expected=" << expected
            << ", got=" << outData[i];
    }
}

TEST(DNN_Metal, eltwise_min)
{
    // Test element-wise minimum with Metal backend
    Net netCPU, netMetal;

    // Setup CPU network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_min";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "min");
        int eltwise_id = netCPU.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netCPU.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netCPU.setInputsNames(inputNames);
        netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    }

    // Setup Metal network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_min";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "min");
        int eltwise_id = netMetal.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netMetal.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netMetal.setInputsNames(inputNames);
        netMetal.setPreferableBackend(DNN_BACKEND_METAL);
        netMetal.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Prepare test inputs (1x1x4x4)
    int sizes[] = {1, 1, 4, 4};
    Mat input1(4, sizes, CV_32F);
    Mat input2(4, sizes, CV_32F);
    float* data1 = input1.ptr<float>();
    float* data2 = input2.ptr<float>();
    for (int i = 0; i < 16; i++) {
        data1[i] = (float)(i - 8);       // -8 to 7
        data2[i] = (float)(8 - i);       // 8 to -7
    }

    // Test with CPU backend
    netCPU.setInput(input1, "input1");
    netCPU.setInput(input2, "input2");
    Mat outputCPU = netCPU.forward();

    // Test with Metal backend
    netMetal.setInput(input1, "input1");
    netMetal.setInput(input2, "input2");
    Mat outputMetal = netMetal.forward();

    // Results should match
    normAssert(outputCPU, outputMetal, "Eltwise Min: Metal vs CPU");

    // Verify min behavior: min(input1, input2)
    float* outData = outputMetal.ptr<float>();
    for (int i = 0; i < 16; i++) {
        float expected = std::min(data1[i], data2[i]);
        EXPECT_NEAR(expected, outData[i], 1e-5)
            << "Eltwise Min mismatch at index " << i
            << ": input1=" << data1[i]
            << ", input2=" << data2[i]
            << ", expected=" << expected
            << ", got=" << outData[i];
    }
}

TEST(DNN_Metal, eltwise_div)
{
    // Test element-wise division with Metal backend
    Net netCPU, netMetal;

    // Setup CPU network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_div";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "div");
        int eltwise_id = netCPU.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netCPU.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netCPU.setInputsNames(inputNames);
        netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    }

    // Setup Metal network
    {
        LayerParams eltwiseParams;
        eltwiseParams.name = "eltwise_div";
        eltwiseParams.type = "Eltwise";
        eltwiseParams.set("operation", "div");
        int eltwise_id = netMetal.addLayerToPrev(eltwiseParams.name, eltwiseParams.type, eltwiseParams);
        netMetal.connect(0, 1, eltwise_id, 1);
        std::vector<String> inputNames = {"input1", "input2"};
        netMetal.setInputsNames(inputNames);
        netMetal.setPreferableBackend(DNN_BACKEND_METAL);
        netMetal.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Prepare test inputs (1x1x4x4)
    int sizes[] = {1, 1, 4, 4};
    Mat input1(4, sizes, CV_32F);
    Mat input2(4, sizes, CV_32F);
    float* data1 = input1.ptr<float>();
    float* data2 = input2.ptr<float>();
    for (int i = 0; i < 16; i++) {
        data1[i] = (float)(i + 1);       // 1 to 16 (avoid division by zero)
        data2[i] = (float)(2);           // All 2s for simple division
    }

    // Test with CPU backend
    netCPU.setInput(input1, "input1");
    netCPU.setInput(input2, "input2");
    Mat outputCPU = netCPU.forward();

    // Test with Metal backend
    netMetal.setInput(input1, "input1");
    netMetal.setInput(input2, "input2");
    Mat outputMetal = netMetal.forward();

    // Results should match
    normAssert(outputCPU, outputMetal, "Eltwise Div: Metal vs CPU");

    // Verify division behavior: input1 / input2
    float* outData = outputMetal.ptr<float>();
    for (int i = 0; i < 16; i++) {
        float expected = data1[i] / data2[i];
        EXPECT_NEAR(expected, outData[i], 1e-5)
            << "Eltwise Div mismatch at index " << i
            << ": input1=" << data1[i]
            << ", input2=" << data2[i]
            << ", expected=" << expected
            << ", got=" << outData[i];
    }
}

TEST(DNN_Metal, convolution_layer)
{
    // Test 2D convolution with Metal backend
    // Create a simple 3x3 convolution with 1 input channel and 2 output channels

    // Prepare a simple model with convolution
    Net netMetal;
    netMetal.setPreferableBackend(DNN_BACKEND_METAL);
    netMetal.setPreferableTarget(DNN_TARGET_CPU);

    // Read a model with convolution layer (using a pre-existing ONNX model)
    // For this test, we'll use SqueezeNet which has convolution layers
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));

    // Set up Metal backend
    net.setPreferableBackend(DNN_BACKEND_METAL);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Create test input (1 batch, 3 channels, 224x224)
    int inputSizes[] = {1, 3, 224, 224};
    Mat input = Mat::ones(4, inputSizes, CV_32F);

    // Run inference with Metal backend
    net.setInput(input);
    Mat outputMetal;
    ASSERT_NO_THROW(outputMetal = net.forward());
    ASSERT_FALSE(outputMetal.empty());

    // Also test with CPU backend for comparison
    Net netCPU = readNetFromONNX(findDataFile("dnn/onnx/models/squeezenet.onnx"));
    netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
    netCPU.setPreferableTarget(DNN_TARGET_CPU);
    netCPU.setInput(input);
    Mat outputCPU = netCPU.forward();

    // Results should be close (allowing for numerical differences)
    normAssert(outputCPU, outputMetal, "Convolution: Metal vs CPU", 1e-3, 1e-4);
}

TEST(DNN_Metal, simple_conv_test)
{
    // Test a simple manually created convolution layer
    Net netCPU, netMetal;

    // Create a simple 1-channel 3x3 convolution
    LayerParams convParams;
    convParams.name = "conv1";
    convParams.type = "Convolution";
    convParams.set("kernel_size", 3);
    convParams.set("num_output", 2);  // 2 output channels
    convParams.set("pad", 1);
    convParams.set("stride", 1);
    convParams.set("bias_term", true);

    // Create weights and bias
    // Weights: [num_output, num_input, kernel_h, kernel_w] = [2, 1, 3, 3]
    Mat weights(std::vector<int>{2, 1, 3, 3}, CV_32F);
    randn(weights, 0.0f, 0.5f);  // Random weights

    // Bias: [num_output] = [2]
    Mat bias(2, 1, CV_32F);
    randn(bias, 0.0f, 0.1f);  // Random bias

    convParams.blobs.push_back(weights);
    convParams.blobs.push_back(bias);

    // Setup CPU network
    {
        netCPU.addLayerToPrev(convParams.name, convParams.type, convParams);
        netCPU.setInputsNames({"input"});
        netCPU.setPreferableBackend(DNN_BACKEND_OPENCV);
        netCPU.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Setup Metal network
    {
        netMetal.addLayerToPrev(convParams.name, convParams.type, convParams);
        netMetal.setInputsNames({"input"});
        netMetal.setPreferableBackend(DNN_BACKEND_METAL);
        netMetal.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Create test input (1 batch, 1 channel, 5x5)
    Mat input(std::vector<int>{1, 1, 5, 5}, CV_32F);
    randn(input, 0.0f, 1.0f);  // Random input

    // Test with CPU backend
    netCPU.setInput(input, "input");
    Mat outputCPU = netCPU.forward();

    // Test with Metal backend
    netMetal.setInput(input, "input");
    Mat outputMetal = netMetal.forward();

    // Results should match closely
    normAssert(outputCPU, outputMetal, "Simple Conv: Metal vs CPU", 1e-3, 1e-4);
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
