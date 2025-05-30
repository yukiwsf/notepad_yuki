# 目标跟踪

## 概述

目标跟踪任务分类： 

1. 单目标跟踪：给定一个目标，跟踪这个目标的位置。 

2. 多目标跟踪：跟踪多个目标的位置。 

3. Person Re-ID：行人重识别，是利用计算机视觉技术判断图像或者视频序列中是否存在特定行人的技术。广泛被认为是一个图像检索的子问题。给定一个监控行人图像，检索跨设备下的该行人图像。旨在弥补固定的摄像头的视觉局限，并可与行人检测/行人跟踪技术相结合。 

4. MTMCT：多目标多摄像头跟踪（Multi-Target Multi-Camera Tracking），跟踪多个摄像头拍摄的多个人。 

5. 姿态跟踪：追踪人的姿态。 

目标跟踪任务按照计算类型又可以分为以下两类： 

1. 在线跟踪：在线跟踪需要实时处理任务，通过过去和现在帧来推断未来帧中物体的位置。 

2. 离线跟踪：离线跟踪是离线处理任务，可以通过过去、现在和未来帧来推断物体的位置，因此准确率会比在线跟踪高。 

目标跟踪困难点： 

1. 姿态变化：姿态变化是目标跟踪中常见的干扰问题。运动目标发生姿态变化时，会导致它的特征以及外观模型发生改变，容易导致跟踪失败。例如：体育比赛中的运动员、马路上的行人。 

2. 尺度变化：尺度的自适应也是目标跟踪中的关键问题。当目标尺度缩小时，由于跟踪框不能自适应跟踪，会将很多背景信息包含在内，导致目标模型的更新错误；当目标尺度增大时，由于跟踪框不能将目标完全包括在内，跟踪框内目标信息不全，也会导致目标模型的更新错误。因此，实现尺度自适应跟踪是十分必要的。 

3. 遮挡与消失：目标在运动过程中可能出现被遮挡或者短暂的消失情况。当这种情况发生时，跟踪框容易将遮挡物以及背景信息包含在跟踪框内，会导致后续帧中的跟踪目标漂移到遮挡物上面。若目标被完全遮挡时，由于找不到目标的对应模型，会导致跟踪失败。 

4. 图像模糊：光照强度变化，目标快速运动，低分辨率等情况会导致图像模型，尤其是在运动目标与背景相似的情况下更为明显。因此，选择有效的特征对目标和背景进行区分非常必要。 

目标跟踪算法按模式分类： 

1. 生成式模型：建立目标模型或者提取目标特征，在后续帧中进行相似特征搜索，逐步迭代实现目标定位（光流算法、滤波算法、MeanShift算法等）。 

2. 鉴别式模型：将目标模型和背景信息同时考虑在内，通过对比目标模型和背景信息的差异，将目标模型提取出来，从而得到当前帧中的目标位置（ML、DL等）。 

目标跟踪算法的发展历程： 

经典跟踪算法 -> 核相关的滤波算法 -> 深度学习算法 

经典跟踪算法： 

1. 光流法（Optical Flow） 
   
   对视频序列中目标外观的像素进行操作，利用视频序列在相邻帧之间的像素关系，寻找像素的位移变化来判断目标的运动状态，实现对运动目标的跟踪。所谓光流就是瞬时速率，在时间间隔很小（比如视频的连续前后两帧之间）时，也等同于目标点的位移。 

2. MeanShift（均值漂移） 
   
   通过计算漂移向量，来更新目标的位置。 
   
   通俗地讲，任选一个点，然后以这个点为球心，h为半径做一个高维球，因为有d维，d可能大于2，所以是高维球。落在这个球内的所有点和球心都会产生一个向量（以球心为起点落在球内的点为终点的向量）。然后把这些向量都加起来，结果就是漂移（MeanShift）向量。再将漂移向量与原球心坐标相加得到新的球心坐标，再以h为半径做一个高维球，计算漂移向量，然后更新球心坐标。反复迭代，直至达到终止条件。

3. 粒子滤波（Partical Filter） 
   
   是一种基于蒙特卡洛方法的粒子分布统计，从后验概率中抽取的随机粒子来表达其分布。粒子滤波是指通过寻找一组在状态空间传播的随机样本对概率密度函数进行近似，以样本均值代替积分运算，从而获得状态最小方差分布的过程。这里的样本即指粒子，当样本数量N→∞时可以逼近任何形式的概率密度分布。 
   
   粒子滤波的优点：尽管算法中的概率分布只是真实分布的一种近似，但由于非参数化的特点，它摆脱了解决非线性滤波问题时随机量必须满足高斯分布的制约，能表达比高斯模型更广泛的分布，也对变量参数的非线性特性有更强的建模能力。因此，粒子滤波能够比较精确地表达基于观测量和控制量的后验概率分布，可以用于解决SLAM问题。粒子滤波在非线性、非高斯系统表现出来的优越性，决定了它的应用范围非常广泛。另外，粒子滤波的多模态处理能力，也是它应用广泛的原因之一。 
   
   粒子滤波的缺点：需要用大量的样本数量才能很好地近似系统的后验概率密度，有效减少样本数量的自适应采样策略是该算法的优化重点。另外，重采样阶段会造成样本有效性和多样性的损失，导致样本贫化现象，保持粒子的有效性和多样性，克服样本贫化，也是该算法的优化重点。

经典目标跟踪算法缺陷： 

1. 没有将背景信息考虑在内，导致在目标遮挡，光照变化以及运动模糊等干扰下容易出现跟踪失败。 

2. 跟踪算法执行速度慢，无法满足实时性的要求。

基于核相关的滤波算法 ：

将通信领域的相关滤波（衡量两个信号的相似程度）引入到了目标跟踪中。一些基于相关滤波的跟踪算法（MOSSE、CSK、KCF、BACF、SAMF等）也随之产生，速度可以达到数百帧每秒，可以广泛地应用于实时跟踪系统中。其中，不乏一些跟踪性能优良的跟踪器，如SAMF、BACF在OTB数据集和VOT2015竞赛中取得优异成绩。

深度学习算法 ：

随着深度学习方法的广泛应用，人们开始考虑将其应用到目标跟踪中。人们开始使用深度特征并取得了很好的效果。之后，人们开始考虑用深度学习建立全新的跟踪框架，进行目标跟踪。 

把深度学习模型提取到的目标特征，直接应用到经典或者核相关的滤波跟踪框架里面，从而得到更好的跟踪结果（DeepSORT、DeepSRDCF等）。在大量训练数据的前提下，深度学习模型输出的特征表达要优于经典滤波器与传统特征描述子，但同时也带来了更大的计算量。

<img src="object_tracking/GetImage.jpg" title="" alt="" width="503">

## 评价指标

### 单目标跟踪

#### VOT（Visual Object Tracking）

评价方法：VOT认为，实验数据不等同于实际表现，为了更加精准地测出tracker的实际表现，应该通过一种算法判定，什么情况下，tracker的实际表现可以视作相同（tracker equivalence）。 

数据集：VOT认为，数据集仅规模大是完全不行的，一个可靠的数据集应能够测试出tracker在不同条件下的表现，如部分遮挡、光照变化等。因此，VOT提出，应该对每一个序列都标注出该序列的视觉属性（visual attributes），以对应不同的条件。除此之外，VOT认为序列中的每一帧都需要进行视觉属性的标注，即使是同一序列，不同帧的视觉属性也不同，这么做可以对tracker进行更精准的评价。 

评价系统：在VOT提出之前，比较流行的评价系统是让tracker在序列的第一帧进行初始化，之后让tracker一直跑到最后一帧。然而tracker可能会因为一某些因素在开始的几帧就跟丢，所以最终评价系统只利用了序列的很小一部分，造成浪费。而VOT提出，评价系统应该在tracker跟丢的时候检测到错误，并在错误发生的5帧之后对tracker重新初始化，这样可以充分利用数据集。之所以是5帧而不是立即初始化，是因为错误发生之后立即初始化很可能再次跟踪失败，又因为视频中的遮挡一般不会超过5帧，所以会有这样的设定。这是VOT的一个特色机制，即重启（重新初始化）。但重启之后的一部分帧是不能用于评价的，这些帧被称作burn-in period，大量实验结果表明，burn-in period大约为初始化以及之后的10帧。 

Accuracy（用于评价tracker跟踪目标的准确性）：

某序列第$t$帧的accuracy定义为（IoU）：

$\phi_t=\frac{A^G_t\cap A^T_t}{A^G_t\cup A^T_t}$

其中，$A^G_t$代表第$t$帧的ground truth，$A^T_t$代表第$t$帧tracker预测的bounding box。

更详细的，定义$\Phi_t(i,k)$为第$i$个tracker在第$k$次重复中（tracker会在一个序列上重复测试多次）第$t$帧上的accuracy，设重复次数为$N_{rep}$。所以，第i个tracker在第$t$帧上的accuracy定义为：

$\Phi_t(i)=\frac{1}{N_{rep}}\sum\limits^{N_{rep}}_{k=1}\Phi_t(i,k)$

第i个tracker在某序列中的average accuracy定义为：

$\rho_A(i)=\frac{1}{N_{valid}}\sum\limits^{N_{valid}}_{t=1}\Phi_t(i)$

其中，$N_{valid}$代表有效帧的数量，除了burn-in period之外的帧均为有效帧。

Robustness（用于评价tracker跟踪目标的稳定性）：

定义$F(i,k)$为第i个tracker在第$k$次重复中发生错误（跟丢）的次数。所以，定义第$i$个tracker在某序列中的average robustness定义为：

$\rho_R(i)=\frac{1}{N_{rep}}\sum\limits^{N_{rep}}_{k=1}F(i,k)$

### 多目标跟踪

#### MOT（Multiple Object Tracking）

MOTA（Multiple Object Tracking Accuracy）：某个序列中多目标跟踪的准确率。

$MOTA=1-\frac{\sum_t(FN_t+FP_t+IDSW_t)}{\sum_tGT_t}$

其中，$FN_t$是在第$t$帧中没有匹配到tracker预测的bounding box的ground truth数量。$FP$是在第$t$帧中没有匹配到ground truth的bounding box数量。$IDSW_t$是第$t$帧中发生ID Switch的次数，即匹配错误的ground truth - bounding box对数。

MOTP（Multiple Object Tracking Precision）：某个序列中多目标跟踪的精确率。

$MOTP=\frac{\sum_{t,i}d_{t,i}}{\sum_tC_t}$

其中，$d_{t,i}$表示第$t$帧的第$i$个bounding box与匹配到的ground truth之间的距离（这里使用IoU计算）。$C_t$表示第$t$帧匹配正确的ground truth - bounding box对数。

MT（Mostly Tracked）：在某个序列中至少80%的时间内都与trakcer预测的bounding box匹配成功的ground truth数量，占所有ground truth的比例。注意，计算MT和ML时，只要有tracker预测的bounding box与ground truth匹配上，无论tracker的ID是否发生变化，即视为匹配成功。 

ML（Mostly Lost）：在某个序列中不超过20%的时间内没有匹配到tracker预测的bounding box的ground truth数量，占所有ground truth的比例。 

FP（False Positives）：误报的总数，没有匹配到ground truth的bounding box数量，一般在某一帧内统计。 

FN（False Negatives）：未命中的目标总数（ground truth），没有匹配到bounding box的ground truth数量，一般在某一帧内统计。 

ID_SW（ID Switch）：发生ID切换的次数，匹配错误的ground truth - bounding box对数，一般在某一帧内统计。

## KCF

Kernelized Correlation Filters：基于核相关的目标跟踪滤波算法。

KCF是一种机器学习鉴别式的跟踪方法，核心思想： 利用第$i$帧的图像$I_i$和目标位置$P_i$初始化滤波器（回归器），利用第$i+1$帧的图像$I_{i+1}$作为滤波器的输入，计算第$I_{i+1}$帧图像中响应（相关性）最大的位置$P_{i+1}$（在目标位置$P_i$附近采样，预测每个采样位置的响应值，取响应值最大的采样位置作为$P_{i+1}$）。 

### 循环位移与循环矩阵

KCF中，每个的采样样本$X$均由目标样本$x$循环位移得到（$x$是目标位置的roi区域），将这个过程表示为$X=C(x)$。

例如，左乘循环矩阵$P$后，$x$的所有列向量向右循环移动：

$P=\begin{bmatrix}0&0&0&\cdots&1\\1&0&0&\cdots&0\\0&1&0&\cdots&0\\\vdots&\vdots&\ddots&\ddots&\vdots\\0&0&\cdots&1&0\end{bmatrix}$

令$Q=P^T$，右乘$Q$后，$x$的所有列向量向下循环移动：

$P=\begin{bmatrix}0&1&\cdots&0&0\\0&0&\cdots&0&0\\\vdots&\vdots&\ddots&\vdots&\vdots\\0&0&\cdots&0&1\\1&0&\cdots&0&0\end{bmatrix}$

```python
x = np.array([[1, 11, 111], 
              [2, 22, 222], 
              [3, 33, 333]]) 
P = np.array([[0, 0, 1], 
              [1, 0, 0], 
              [0, 1, 0]]) 
Q = np.array([[0, 1, 0], 
              [0, 0, 1], 
              [1, 0, 0]]) 
Px = np.matmul(P, x) 
xQ = np.matmul(x, Q) 
print(np.transpose(Px, (1, 0))) 
```

循环位移后的采样样本：

<img title="" src="object_tracking/2025-05-26-13-25-00-GetImage.png" alt="" width="371">

### 岭回归

KCF采用岭回归（Ridge Regression）的方法，训练的目标是找到一个函数$f(z)=w^Tz$最小化样本$x_i$与其回归目标$y_i$的误差平方和。公式表示为：

$L=\min\limits_w\sum\limits_i(f(x_i)-y_i)^2+\lambda\Vert w\Vert^2$

其中，$\lambda$是正则化参数。

在频域中，当损失函数的值$L$为$0$时，求解$w$：

$w=(X^HX+\lambda I)^{-1}X^Hy$

根据循环矩阵能通过离散傅里叶变换（DFT，Discrete Fourier Transform）对角化，使得矩阵求逆转换为特征值求逆的性质，将$w$的求解转换到频域，应用DFT提高计算速度，从而求得响应最大的解。

循环矩阵$X$可以通过DFT对角化：

$X=F{\rm diag}(\hat x)F^H$

其中，$\hat x$是由$x$经过DFT得到，$\hat x=\mathcal F(x)$，$F$是DFT矩阵（常量矩阵，与$x$无关），$\mathcal F(z)=\sqrt{n}Fz$。

于是：

$\begin{aligned}X^HX&=F{\rm diag(\hat x^*)}F^HF{\rm diag(\hat x)}F^H\\&=F{\rm diag(\hat x^*)}{\rm diag(\hat x)}F^H\\&=F{\rm diag(\hat x^*\odot\hat x)}F^H\end{aligned}$

代入得：

$\begin{aligned}w&=(X^HX+\lambda I)^{-1}X^Hy\\&=(F{\rm diag}(\hat x^*\odot\hat x)F^H+F{\rm diag}(\lambda)F^H)^{-1}X^Hy\\&=(F{\rm diag}(\hat x^*\odot\hat x+\lambda)F^H)^{-1}X^Hy\\&=(F{\rm diag}(\frac{\hat x^*}{\hat x^*\odot\hat x+\lambda})F^H)y\end{aligned}$

其中，$^*$为共轭复数，$^H$为共轭转置，$\odot$为逐元素相乘。

由$X=C(\mathcal F^{-1}(\hat x))$，可得：

$w=C(\mathcal F^{-1}(\frac{\hat x^*}{\hat x^*\odot\hat x+\lambda}))y$

$w$经过DFT得（循环位移$C$是线性变换）：

$\hat w=\mathcal F(w)=\mathcal F(C(\mathcal F^{-1}(\frac{\hat x^*}{\hat x^*\odot\hat x+\lambda})))\hat y={\rm diag}(\frac{\hat x^*}{\hat x^*\odot\hat x+\lambda})\hat y$

### 核技巧

引入核技巧来将问题扩展到非线性空间，回归系数$w$用$x$和对偶空间下的$\alpha$的线性组合表示如下：

$w=\sum\limits_i\alpha_i\varphi(x_i)$

回归问题就转为：

$f(z)=w^Tz=\sum\limits^n_{i=1}\alpha_i\kappa(z,x_i),\quad \varphi^T(x)\varphi(x')=\kappa(x,x')$

其中$\kappa$为核函数，如高斯核函数等。

使用核技巧的岭回归问题的解为（$w$由$\alpha$替代）：

$\alpha=(K+\lambda I)^{-1}y$

其中，$K$为核相关矩阵。事实上，$K$随循环位移而循环变化。因此，可以应用循环矩阵的对角化性质：

$\hat\alpha=\frac{\hat y}{\hat k^{xx}+\lambda}$

其中，$k^{xx}$是$K$的第一行（$K=C(k^{xx})$）。

### 算法流程

核心函数包含init函数（利用第一帧的图像和目标框初始化跟踪器）和update函数（利用后续帧的图像进行预测）。

1. 初始化
   
   输入：第一帧图像中的目标区域。
   
   特征提取：提取目标特征（如HOG等），得到特征$x$。
   
   响应图生成：创建高斯分布的响应图$y$，峰值位于目标中心。
   
   训练滤波器：在频域计算初始化滤波器模型参数$\alpha$与核相关矩阵$k^{xx}$。

2. 检测阶段
   
   输入：新一帧图像，以前一帧目标位置为中心裁剪搜索区域，提取特征$z$。
   
   核相关计算：计算$z$与初始目标特征$x$的核相关$k^{xz}$。
   
   响应图预测：在频域计算响应图$\hat f(z)=\mathcal F^{-1}(\hat k^{xz}\odot\hat\alpha)$。
   
   目标定位：响应图的峰值位置即为当前帧的目标位置。

3. 模型更新
   
   更新策略：采用线性插值更新模型参数，平衡历史信息与新观测。
   
   $\alpha_{new}=(1-\eta)\alpha_{prev}+\eta\alpha_{current}$
   
   $x_{new}=(1-\eta)x_{prev}+\eta x_{current}$
   
   其中，$\eta$是学习率，通常取$[0.01,0.1]$。

<img src="object_tracking/2025-05-27-17-33-14-image.png" title="" alt="" width="382">

## SiamFC

离线学习，在线跟踪。学习一个函数/模型f(z,x)，来比较模板图像z和搜索图像x的相关性，如果两个图像的相关性越高，则得分越高。
