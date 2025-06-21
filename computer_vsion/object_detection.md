# 目标检测

## 概述

目标检测任务：找到图像中的目标物体（感兴趣区域），并确定其类别和位置信息。 

目标检测算法分类： 

1. One-stage（End-to-End，端到端）: 选取候选区域和目标检测（分类回归）放在一个网络中进行，如：YOLO系列。

2. Two-stage（两阶段）: 选取候选区域和目标检测分两阶段进行，如：R-CNN、SPPNet、Fast R-CNN、Faster R-CNN。

常见损失函数： 

- 分类：CrossEntropy。

- 回归：MSE、L1、L2、IoU等。

两种bounding box（边界框）： 

1. ground-truth bounding box：人为标记的真实边界框。

2. predicted bounding box：网络预测得出的边界框。

## 评价指标

准确率（Accuracy）：所有样本中预测正确的比例，$\rm(TP+TN)/(TP+FN+FP+TN)$。

精确率（Precision）：预测结果为正例样本中真实为正例的比例（查准率），$\rm TP/(TP+FP)$。

召回率（Recall）：真实为正例的样本中预测结果为正例的比例（查全率）：$\rm TP/(TP+FN)$ 

平均精确率（Mean Average Precision）mAP：${\rm mAP}=\sum\limits^{n}_{c=0}{\rm AP}/n$ 

在数据集上用mAP评估目标检测模型步骤： 

1. 计算每个类别的AP：求数据集所有图片中每个类别的AP。
   
   1. 设定一个IoU阈值，对每张图中某一类别的所有预测框（目标检测模型的最终输出结果）与该类别的所有GT进行IoU匹配。 
   
   2. 根据预测框与GT的匹配情况（一个预测框与某个GT匹配意味着该预测框来预测此目标）判断该类别下TP、FP、FN的值（TP为匹配到GT的预测框，FP为未匹配到GT的预测框，FN为未匹配到预测框的GT，不需要TN）。 
   
   3. 对该类别绘制PR曲线，依次计算置信度阈值为[0:1:0.1]时的(Recall,Precision)坐标（按置信度由高到低排序，剔除置信度低于阈值的预测框）。 
   
   4. 根据计算得到的不同置信度阈值下的(Recall,Precision)坐标绘制PR曲线，并计算该类别的AP（PR曲线的线下面积，同AUC，即ROC线下面积）。 

2. 计算所有类别的AP并求平均值得到mAP。 

预测框与GT的匹配原则（同一张图内）： 

1. 与所有GT的IoU均小于IoU阈值的预测框不匹配GT。 

2. 与多个GT的IoU均大于IoU阈值的预测框，匹配IoU最大的GT。 

3. 每个GT最多只与一个预测框匹配。 

TP/FP/FN确定原则： 

针对数据集所有图片每一个类别（模型分类结果）的预测框执行下述操作： 

1. 匹配到GT的预测框为TP。 

2. 未匹配到GT的预测框为FP。 

3. 未匹配到预测框的GT为FN。 

P.S. 混淆矩阵：
