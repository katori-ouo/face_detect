# 基于Faster-RCNN的遮挡人脸检测

## model_data文件夹
- simhei.ttf为程序画图的字体
- voc_classes.txt为数据集内目标的种类
- voc_weights.h5为训练的预权重

## logs文件夹
- 用来存放训练得到的.h5格式网络模型和训练日志

## nets文件夹
- frcnn.py：搭建RPN网络和Faster R-CNN的整体结构
- frcnn_training.py：定义训练过程的各种损失函数、描述训练过程的操作
- resnet.py：搭建主干特征提取网络Resnet
- RoiPoolingConv.py：搭建池化层RoIPooling

## utils文件夹
- anchors.py：生成先验框、解码先验框
- config.py：定义训练和预测过程中使用的参数
- roi_helpers.py：描述池化层RoIPooling层中计算IoU、筛选样本的操作
- utils.py：定义训练和预测过程中使用的功能函数

## VOCdevkit/VOC2007文件夹
- Annotations：数据集的.xml注释文件
- ImageSets：数据集中训练集、验证集和测试集的划分
- JPEGImages：数据集的图像内容
- voc2faster-rcnn.py：根据Annotations的内容，生成ImageSets文件夹内容

## frcnn.py
- 加载model_data/voc_classes.txt，定义Faster R-CNN网络和接口函数

## train.py
- 加载model_data/voc_weights.h5作为预权重，训练神经网络模型

## predict.py
- 利用train.py得到的网络模型，对输入图像做人脸检测

## viedo.py
- 调用电脑摄像头，进行实时人脸检测

## get_dr_txt.py
- 对测试集中的图像做人脸检测，保存检测结果

## get_gt_txt.py
- 根据注释文件，得到测试集中人脸的真实结果

## get_mAP
- 根据检测结果和真实结果，计算模型的精准率、准确率、召回率和平均准确率

## Vision_for_anchor.py
- 可视化生成anchor先验框

## voc_annotation.py
- 根据VOCdevkit中的内容，生成训练文件的路径+注释集合

## mask_generation.py
- 生成佩戴口罩的遮挡人脸图像

## show.py
- 可视化生成口罩