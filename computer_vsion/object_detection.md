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

```python
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

## NMS

```python
import numpy as np 


def Intersaction(box_a, box_b): 
    a1_x, a1_y, a2_x, a2_y = box_a 
    b1_x, b1_y, b2_x, b2_y = box_b 
    x1 = max(a1_x, b1_x) 
    y1 = min(a1_y, b1_y) 
    x2 = min(a2_x, b2_x) 
    y2 = max(a2_y, b2_y) 
    if y2 - y1 <= 0 or x2 - x1: 
        return 0 
    return (y2 - y1) * (x2 - x1) 


def Union(box_a, box_b): 
    a1_x, a1_y, a2_x, a2_y = box_a 
    b1_x, b1_y, b2_x, b2_y = box_b 
    area_a = (a2_x - a1_x) * (a2_y - a1_y) 
    area_b = (b2_x - b1_x) * (b2_y - b1_y) 
    return area_a + area_b - Intersaction(box_a, box_b) 


def IoU(box_a, box_b): 
    a1_x, a1_y, a2_x, a2_y = box_a 
    b1_x, b1_y, b2_x, b2_y = box_b 
    if a1_x >= a2_x or a1_y >= a2_y or b1_x >= b2_x or b1_y >= b2_y: 
        return 0 
    return float(Intersaction(box_a, box_b) / float(Union(box_a, box_b) + 1e-6)) 


def NMS(dets, thresh): 
    """ 
    :param dets: [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...] 
    :param thresh: iou threshold 
    :return: list to save index of bboxes in dets after nms 
    """ 
    dets = np.array(dets) 
    x1 = dets[:, 0] 
    y1 = dets[:, 1] 
    x2 = dets[:, 2] 
    y2 = dets[:, 3] 
    scores = dets[:, 4] 
    areas = (y2 - y1) * (x2 - x1) 
    order = scores.argsort()[::-1] 
    keep = [] 
    while order.size > 0: 
        i = order[0] 
        keep.append(i) 
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        yy1 = np.maximum(y1[i], y1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]]) 
        yy2 = np.minimum(y2[i], y2[order[1:]]) 
        w = np.maximum(0.0, xx2 - xx1) 
        h = np.maximum(0.0, yy2 - yy1) 
        inter = w * h 
        iou = inter / (areas[i] + areas[order[1:]] - inter) 
        inds = np.where(iou <= thresh)[0] 
        order = order[inds + 1] 
    return keep    
```

```cpp
#include <iostream> 
#include <vector> 
#include <algorithm> 

struct BBox { 
    float x1; 
    float y1; 
    float x2; 
    float y2; 
    float score; 
}; 

inline float IoU(float x1_box1, float y1_box1, float x2_box1, float y2_box1, float x1_box2, float y1_box2, float x2_box2, float y2_box2) { 
    float inner_x1 = x1_box1 > x1_box2 ? x1_box1 : x1_box2; 
    float inner_y1 = y1_box1 > y1_box2 ? y1_box1 : y1_box2; 
    float inner_x2 = x2_box1 < x2_box2 ? x2_box1 : x2_box2; 
    float inner_y2 = y2_box1 < y2_box2 ? y2_box1 : y2_box2; 
    float inner_h = inner_y2 - inner_y1 + 1; 
    float inner_w = inner_x2 - inner_x1 + 1; 
    if(inner_h <= 0 || inner_w <= 0) 
        return 0; 
    float inner_area = inner_h * inner_w; 
    float h1 = y2_box1 - y1_box1 + 1; 
    float w1 = x2_box1 - x1_box1 + 1; 
    float h2 = y2_box2 - y1_box2 + 1; 
    float w2 = x2_box2 - x1_box2 + 1; 
    float area1 = h1 * w1; 
    float area2 = h2 * w2; 
    float iou = inner_area / (area1 + area2 - inner_area); 
    return iou; 
} 

void NMS(std::vector<BBox>& input, std::vector<BBox>& output, float iou_threshold) { 
    std::sort(input.begin(), input.end(), [](BBox& a, BBox& b) { return a.score > b.score; }); 
    int input_num = input.size(); 
    std::vector<int> merged(input_num, 0); 
    for(int i = 0; i < input_num; ++i) { 
        if(merged[i]) 
            continue; 
        output.push_back(input[i]); 
        merged[i] = 1;  // if input i is added into output or suppressed, set merged[i] = 1, else retained 0 
        if(i == input_num - 1)  
            break; 
        for(int j = i + 1; j < input_num; ++j) { 
            if(merged[j]) 
                continue; 
            float iou = IoU(input[i].x1, input[i].y1, input[i].x2, input[i].y2, input[j].x1, input[j].y1, input[j].x2, input[j].y2); 
            if(iou > iou_threshold) 
                merged[j] = 1; 
        } 
    } 
} 

int main(int argc, char* argv[]) { 
    std::vector<BBox> series = { { 2.f, 2.f, 2.f, 2.f, 0.2f }, { 1.f, 1.f, 1.f, 1.f, 0.1f }, { 3.f, 3.f, 3.f, 3.f, 0.3f }, }; 
    std::vector<BBox> series_out; 
    int n = sizeof(series) / sizeof(series[0]); 
    float iou_thresh = 0.5; 
    NMS(series, series_out, iou_thresh); 
    for(auto& i : series_out) 
        std::cout << i.score << std::endl; 
    return 0; 
} 
```

```c
#include <stdio.h> 
#include <stdlib.h> 

typedef struct BoundingBox { 
    float x1; 
    float y1; 
    float x2; 
    float y2; 
    float score; 
} BBox; 

void ShellSort(BBox A[], int N) { 
    int i, j, increment; 
    BBox tmp; 
    for(increment = N / 2; increment > 0; increment /= 2) {
        for(i = increment; i < N; ++i) {   
            tmp = A[i]; 
            for(j = i; j >= increment; j -= increment) 
                if(tmp.score > A[j - increment].score)   
                    A[j] = A[j - increment]; 
                else 
                    break; 
            A[j] = tmp;  // put tmp in right position 
        } 
    }
} 

float IoU(float x1_box1, float y1_box1, float x2_box1, float y2_box1, float x1_box2, float y1_box2, float x2_box2, float y2_box2) { 
    float inner_x1 = x1_box1 > x1_box2 ? x1_box1 : x1_box2; 
    float inner_y1 = y1_box1 > y1_box2 ? y1_box1 : y1_box2; 
    float inner_x2 = x2_box1 < x2_box2 ? x2_box1 : x2_box2; 
    float inner_y2 = y2_box1 < y2_box2 ? y2_box1 : y2_box2; 
    float inner_h = inner_y2 - inner_y1 + 1; 
    float inner_w = inner_x2 - inner_x1 + 1; 
    if(inner_h <= 0 || inner_w <= 0) 
        return 0; 
    float inner_area = inner_h * inner_w; 
    float h1 = y2_box1 - y1_box1 + 1; 
    float w1 = x2_box1 - x1_box1 + 1; 
    float h2 = y2_box2 - y1_box2 + 1; 
    float w2 = x2_box2 - x1_box2 + 1; 
    float area1 = h1 * w1; 
    float area2 = h2 * w2; 
    float iou = inner_area / (area1 + area2 - inner_area); 
    return iou; 
} 

void NMS(BBox input[], int input_num, float iou_threshold) { 
    ShellSort(input, input_num);  // descending order 
    for(int i = 0; i < input_num; ++i) { 
        if(input[i].score == 0) 
            continue; 
        if(i == input_num - 1)  
            break; 
        for(int j = i + 1; j < input_num; ++j) { 
            if(input[j].score == 0) 
                continue; 
            float iou = IoU(input[i].x1, input[i].y1, input[i].x2, input[i].y2, input[j].x1, input[j].y1, input[j].x2, input[j].y2); 
            if(iou > iou_threshold) 
                input[j].score = 0;  // if a bbox is suppressed, set the score = 0 
        } 
    } 
} 

int main(int argc, char* argv[]) { 
    BBox series[3] = { { 2.f, 2.f, 2.f, 2.f, 0.2f }, { 1.f, 1.f, 1.f, 1.f, 0.1f }, { 3.f, 3.f, 3.f, 3.f, 0.3f }, }; 
    int n = sizeof(series) / sizeof(series[0]); 
    float iou_thresh = 0.5; 
    NMS(series, n, iou_thresh); 
    for(int i = 0; i < n; ++i) {
        if(series[i].score != 0) 
            printf("%f\n", series[i].score); 
    }
    return 0; 
} 
```

## Soft-NMS

NMS缺点：当两个目标距离很近时，score低的预测框可能被删掉。 

核心思想：不要直接删除所有IoU大于阈值的框，而是降低其score，IoU越大score越低。 

NMS可以描述为：把IoU大于阈值的预测框score置为0。

$s_i=\begin{cases}s_i,\quad{\rm iou}(\mathcal{M} )\end{cases}$
