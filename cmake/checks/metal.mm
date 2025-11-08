@import Metal;
@import MetalPerformanceShadersGraph;

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 1;

        MPSGraph* graph = [[MPSGraph alloc] init];
        if (!graph) return 1;

        MPSGraphTensor* a = [graph placeholderWithShape:@[@2,@3]
                                              dataType:MPSDataTypeFloat32
                                                  name:@"a"];
        MPSGraphTensor* b = [graph placeholderWithShape:@[@2,@3]
                                              dataType:MPSDataTypeFloat32
                                                  name:@"b"];
        MPSGraphTensor* c = [graph additionWithPrimaryTensor:a
                                             secondaryTensor:b
                                                        name:@"c"];

        return (a && b && c) ? 0 : 1;
    }
}
