
# Architecture

Layers



**Layer**|**Name**|**Kernel**|**Stride**|**Padding**|**Output**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
1|Input|-|-|-|(32,32,3)
2|Convolution/Relu|5x5|1|2|(32,32,32)
3|Average Pool|3x3 |2|[0 1 0 1]|(32,16,16)
4|Convolution/Relu|5x5|1|2|(32,16,16)
5|Average Pool|3x3 |2|[0 1 0 1]|(32,8,8)
6|Convolution/Relu|5x5|1|2|(64,8,8)
7|Max Pool|3x3 |2|[0 1 0 1]|(64,4,4)
8|Convolution/Relu|4x4|1|2|(64,1,1)
9|Fully Connected|1x1|1|0|(64)
10|Softmax Loss| | | |(43)