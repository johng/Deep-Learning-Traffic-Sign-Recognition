
# Architecture

Layers

1. Input: Out (3,32,32)

2. Convolution/Relu: 32 features Kernel 5x5 Stride 1 Pad 2 : Out (32,32,32)
3. Average Pooling: Kernel 3x3 Stride 2  Pad [0 1 0 1]: Out (32, 16,16)

4. Convolution/Relu: 32 features Kernel 5x5 Stride 1 Pad 2 : Out (32,16,16)
5. Average Pooling: Kernel 3x3 Stride 2  Pad [0 1 0 1]: Out (32, 8,8)

6. Convolution/Relu: 64 features Kernel 5x5 Stride 1 Pad 2 : Out (64,8,8)
7. Average Pooling: Kernel 3x3 Stride 2  Pad [0 1 0 1]: Out (64, 4,4)

8. Convolution/Relu: 64 features Kernel 4x4 Stride 1 Pad 0 : Out (64,1,1)
9. Fully Connected: Out (32, 4,4)

10 Softmax-loss Out: (43)