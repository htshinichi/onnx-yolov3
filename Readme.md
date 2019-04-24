## onnx-yolov3
### 依赖库
- onnxruntime
- numpy
- cv2
### 模型来源
darknet--->caffe--->onnx
1.[darknet转caffe参考](https://blog.csdn.net/Chen_yingpeng/article/details/80692018)
2.[caffe转onnx](https://github.com/htshinichi/caffe-onnx)。
### 模型介绍
转换了输入尺寸为416、608的yolov3模型，以及输入尺寸为416的yolov3-tiny模型。
#### yolov3-416  
##### 模型输出  
输入为416x416的图像，输入名为input。
输出为三个feature map，维度分别是255x13x13，255x26x26，255x52x52，其中255=3 x (80 + 5)，80个类的概率加$t_x,t_y,t_w,t_h,t_o$(置信度)。
**节点类型种类**  
```
各类型节点数为：
Upsample:2个
Concat:4个
Add:23个
BatchNormalization:72个
Conv:75个
LeakyRelu:72个
```

#### yolov3-608  
##### 模型输出  
输入为608x608的图像，输入名为input。
输出为三个feature map，维度分别是255x19x19，255x38x38，255x76x76，其中255=3 x (80 + 5)，80个类的概率加$t_x,t_y,t_w,t_h,t_o$(置信度)。
**节点类型种类**  
```
各类型节点数为：
Upsample:2个
MaxPool:3个
Concat:7个
Add:23个
BatchNormalization:73个
Conv:76个
LeakyRelu:73个
```

#### yolov3-tiny  
##### 模型输出  
输入为416x416的图像，输入名为input。
输出为两个feature map，维度分别是255x13x13，255x26x26，其中255=3 x (80 + 5)，80个类的概率加$t_x,t_y,t_w,t_h,t_o$(置信度)。
**节点类型种类**  
```
各类型节点数为：
Upsample:1个
MaxPool:6个
Concat:2个
BatchNormalization:11个
Conv:13个
LeakyRelu:11个
```




### 测试图像结果  
#### yolov3-416

