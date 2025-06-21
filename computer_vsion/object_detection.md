# 目标检测

## 概述

目标检测任务：找到图像中的目标物体（感兴趣区域），并确定其类别和位置信息。 

目标检测算法分类： 

1. One-stage（End-to-End，端到端）: 选取候选区域和目标检测（分类回归）放在一个网络中进行，如：YOLO系列。

2. Two-stage（两阶段）: 选取候选区域和目标检测分两阶段进行，如：R-CNN、SPPNet、Fast R-CNN、Faster R-CNN。

常见损失函数： 

- 分类：CrossEntropy。

- 回归：L1、L2、MSE、IoU等。

两种bounding box（边界框）： 

1. ground-truth bounding box：人为标记的真实边界框。

2. predicted bounding box：网络推理的预测边界框。

## 评价指标

准确率（Accuracy）：所有样本中预测正确的比例，$\rm Acc=(TP+TN)/(TP+FN+FP+TN)$。

精确率（Precision）：预测结果为正例样本中真实为正例的比例（查准率），$\rm P=TP/(TP+FP)$。

召回率（Recall）：真实为正例的样本中预测结果为正例的比例（查全率）：$\rm R=TP/(TP+FN)$ 

平均精确率（Mean Average Precision）：${\rm mAP}=\sum\limits^{n}_{c=0}{\rm AP}/n$ 

在数据集上用$\rm mAP$评估目标检测模型步骤： 

1. 计算每个类别的$\rm AP$：求数据集所有图片中每个类别的AP。
   
   1. 设定一个$\rm IoU$阈值，对每张图中某一类别的所有预测框（目标检测模型的最终输出结果）与该类别的所有真实框进行$\rm IoU$匹配。 
   
   2. 根据预测框与真实框的匹配情况（一个预测框与某个真实框匹配意味着该预测框来预测此目标）判断该类别下$\rm TP$、$\rm FP$、$\rm FN$的值（$\rm TP$为匹配到真实框的预测框，$\rm FP$为未匹配到真实框的预测框，$\rm FN$为未匹配到预测框的真实框，不需要$\rm TN$）。 
   
   3. 对该类别绘制$\rm PR$曲线，依次计算置信度阈值为$[0:1:0.1]$时的$\rm(R,P)$坐标（按置信度由高到低排序，剔除置信度低于阈值的预测框）。 
   
   4. 根据计算得到的不同置信度阈值下的$\rm(R,P)$坐标绘制$\rm PR$曲线，并计算该类别的$\rm AP$（$\rm PR$曲线的线下面积，同$\rm AUC$，即$\rm ROC$线下面积）。 

2. 计算所有类别的$\rm AP$并求平均值得到$\rm mAP$。 

预测框与真实框的匹配原则（同一张图内）： 

1. 与所有真实框的$\rm IoU$均小于$\rm IoU$阈值的预测框不匹配真实框。 

2. 与多个真实框的$\rm IoU$均大于$\rm IoU$阈值的预测框，匹配$\rm IoU$最大的真实框。 

3. 每个真实框最多只与一个预测框匹配。 

$\rm TP$、$\rm FP$、$\rm FN$确定原则： 

针对数据集所有图片每一个类别（模型分类结果）的预测框执行下述操作： 

1. 匹配到真实框的预测框为$\rm TP$。 

2. 未匹配到真实框的预测框为$\rm FP$。 

3. 未匹配到预测框的真实框为$\rm FN$。 

混淆矩阵：

|          | 预测为正     | 预测为反     |
| -------- | -------- | -------- |
| **真实为正** | $\rm TP$ | $\rm FN$ |
| **真实为正** | $\rm FP$ | $\rm TN$ |

$\rm mAP@0.5$是指$\rm IoU$阈值为$0.5$时的$\rm mAP$计算结果，$\rm mAP@0.5:0.95$是指$\rm IoU$阈值为$[0.5:0.95:0.05]$时的$\rm mAP$平均值。

$\rm AAP$（Approximated Average Precision）：

由于计算$\rm AP$（$\rm PR$曲线的线下面积）的公式是一个定积分（$p$代表Precision，$r$代表Recall，$p$是以$r$为自变量的函数）：${\rm AP}=\int^1_0p(r){\rm d}r$。因此在实现过程中，近似为：${\rm AP=}\sum\limits^N_{k=1}p(k)\Delta r(k)$。即，在每个阈值下分别求Precision乘以Recall的变化量，再把所有阈值下求得的乘积进行累加。

标注格式转换工具：

[GitHub - Weifeng-Chen/dl_scripts](https://github.com/Weifeng-Chen/DL_tools) 

标注格式：

- VOC：$x_1,y_1,x_2,y_2$

- YOLO：$c_x,c_y,w,h$

- COCO：$x_1, y_1, w, h$

$\rm mAP$计算工具：

[cocoapi/PythonAPI/pycocotools at master · cocodataset/cocoapi · GitHub](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)

```Python
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import numpy as np 
import pylab
import json 


if __name__ == "__main__":
    gt_path = "instances_val2017.json"  # 存放真实标签的路径 
    dt_path = "my_result.json"  # 存放检测结果的路径 
    cocoGt = COCO(gt_path) 
    cocoDt = cocoGt.loadRes(dt_path) 
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox") 
    cocoEval.evaluate() 
    cocoEval.accumulate() 
    cocoEval.summarize() 
```
