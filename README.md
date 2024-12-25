# Proximal Direct Preference Optimization:Not Only Your Model is a Discriminator but also a Generator.

> Idea占坑，后续会更新理论优化和更多的补充实验说明。\
> Author: Hepan\
> Org: Xmov\
> Email: newtonysls@gmail.com\
> Date: 2024/12/15

## 背景介绍

在[《Direct Preference Optimization:Your Language Model is Secretly a Reward Model》](https://arxiv.org/pdf/2305.18290)一文中，该文章作者将RLHF的RL强化学习阶段的过程描述为：

$
\max_{\pi_\theta} E_{x\sim \mathcal{D}  ,y\sim \pi_\theta  (y|x)}[r_\phi (x,y)]-\beta \mathbb{D} _{KL}[\pi_\theta  (y|x)||\pi_{ref}  (y|x)]
$

进而发现能够将上述优化过程给出显示解，将上述优化问题变为

$
\min_{\pi_\theta} E_{x\sim \mathcal{D}  ,y\sim \pi_\theta  (y|x)}[log\frac{\pi_\theta  (y|x)}{\pi_{ref}  (y|x) e^{\frac{r_\phi (x,y)}{\beta}}}]
$

然后构造一个新的分布：

$
\pi^*(y|x) = \frac{\pi_{ref}  (y|x) e^{\frac{r_\phi (x,y)}{\beta}}}{Z(x)}
$
其中Z(x)归一化分母，是上面这个分布关于变量y的求和，因此z（x）只和x变量有关

因此上述第二公式优化问题就成为了

$
\min_{\pi_\theta}E_{x\sim \mathcal{D}}\mathbb{D} _{KL}[\pi_\theta  (y|x)||\pi^{*}  (y|x)]
$
易知，KL散度在两个分布相等时取得最小值。
也就是$\pi^*$就是RL阶段想要获得的最优分布，根据$\pi^*$和$r_\phi (x,y)$的关系，可以得到


$
r_\phi (x,y) = \beta log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta logZ(x)
$
因此可以将优化$\pi_\theta$的过程转化为优化$r_\phi (x,y)$，故得到DPO的最终loss

$
\max_{\pi_\theta}E_{(x,y_w,y_l)-\mathcal{D}}[log\sigma (\beta log\frac{\pi_\theta(y_w|x) }{\pi_{ref}(y_w|x) }-\beta log\frac{\pi_\theta(y_l|x) }{\pi_{ref}(y_l|x) })]
$

**在DPO理论的推导过程中，一切的成立前提是能够把RL阶段的优化过程视为：**

$
\max_{\pi_\theta} E_{x\sim \mathcal{D}  ,y\sim \pi_\theta  (y|x)}[r_\phi (x,y)]-\beta \mathbb{D} _{KL}[\pi_\theta  (y|x)||\pi_{ref}  (y|x)]
$

其中后面的KL散度约束是为了在优化$\pi_\theta$的自己生成的分布上的最大化reward的时，避免出现模型崩溃（不说人话）。

DPO的思路是将RLHF中的优化$\pi_\theta$过程转化为训练Reward model的过程，因而使用的也是训练reward model所使用的数据。结合DPO最后的优化loss式子，可以用一句话描述DPO的训练过程：以$\pi_{ref}$为基准，对于正样本的token logp尽可能大，对于负样本token logp尽可能地小。

如果把我们把DPO的loss转化为以下：
$
\max_{\pi_\theta}E_{(x,y_w,y_l)-\mathcal{D}}\{log\sigma(\beta[log\frac{\pi_\theta(y_w|x)\pi_{ref}(y_l|x)}{\pi_{ref}(y_w|x)\pi_\theta(y_l|x)}])\}
$

细心的你可以发现，DPO为了将正样本logps增大，负样本的logps减小，如果把上述式子中的ref模型去掉，也可以实现同样的效果，即
$
\max_{\pi_\theta}E_{(x,y_w,y_l)-\mathcal{D}}\{log\sigma(\beta[log\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}])\}
$

但是这样做的可能导致真样本的和负样本的logps同时变得很小，但是$\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}$却增大，因此从这个角度，DPO的优化式子中也需要ref模型的存在。

看完了DPO的优化推导过程之中，不少学者基于DPO提出了IPO、KPO等等其他对齐优化算法，但是这类算法和PPO都具有一个本质的不同。DPO算法的数据来源于训练reward model的数据，同时优化项也是对标reward model，因此训练过程中只需要两个模型$\pi_\theta$和$\pi_{ref}$，相比PPO节约了大量的计算资源。但是DPO整个过程中都没有和PPO一样，在$\pi_\theta$分布上进行采样数据，并使用策略梯度进行数据的更新。本文认为这是导致DPO的泛化性相比较于PPO不足的原因。

## PDPO
那有什么方法可以在DPO优化的过程中引入和PPO一样的针对$\pi_\theta$进行采样，并结合该部分数据进行训练的呢？
直觉上，DPO对标训练的是reward model不断增加正样本的reward，不断降低负样本的reward，那为什么不可以把$\pi_\theta$作为reward model来使用呢，即PPO中的actor model和reward model是同一个model。
那么，我们只需要对$\pi_\theta$进行采样，得到$x \sim \mathcal{D_\theta},y\sim\pi_\theta(y|x)$，那么只需要将这部分数据reward进行最大化即可了。

但是这个里面存在一个问题，由于actor model和reward model都是$\pi_\theta$，如果基于$\pi_\theta$进行采样，并且采用DPO中的方式去最大化reward，是非常容易拟合的，导致学习奔溃。另外PPO中的reward model是事先进行训练好的，但DPO中的reward model是由差变好的训练过程，因此在训练初期使用DPO的$\pi_\theta$作为reward model计算reward本身就是没有意义的。

那么就没有其他办法了么，有没有什么算法是让两个model都是从差到好的训练过程呢？
自然地，我们想到了对抗生成网络GAN。
在GAN中，由生成器G和判别器D两部分组成。GAN训练过程就是将G和D由差到好的训练而成。G用于生成能够迷惑D的样本，D用于判别真实样本和生成器G生成的样本。G的训练过程可以描述为尽可能地让D无法判别G生成的样本和真实样本。D训练过程可描述为训练二分类模型，尽可能地对正负样本识别准确。

那么，如何将GAN的思想融合到DPO中，改善没有对$\pi_\theta$进行采样并将采样数据用于训练的缺点呢？

### PDPO的主要思想
在DPO训练过程中，最大化正样本的logps和最小化负样本的logps，其中对正样本和负样本的reward计算可表示为如下的式子

$
Reward(x,y_w) = \beta log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
$

$
Reward(x,y_l) = \beta log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
$

因此DPO的模型$\pi_\theta$天然的是一个判别器模型，同时在训练过程中判别能力越来越强。
那么怎么把生成器G融合到DPO的训练过程中呢？

我们要知道对于生成器G来说，优化目标应该是尽可能地让G的生成结果迷惑D，也就是最大化D的loss。DPO作为判别器的训练loss如之前的公式所表示。DPO的优化式子中并没有由$\pi_\theta$生成的数据作为输入。因此不能直接将DPO的损失套入生成器G。

我们准备这样的一批数据，$(x \sim \mathcal{D}_{pdpo},y_\theta \sim\pi_\theta(y|x),y_{GT} \sim \mathcal{Y}_w)$。x和$y_{GT}$是由reward模型训练数据中采样的部分数据，GT代表Ground Truth，作为训练G的正样本。$y_\theta$是根据输入x经过$\pi_\theta(y|x)$的计算并采样得到的，将其视为训练G的负样本。
那么针对G的正负样本基于DPO的reward的计算，对应的reward如下所示。

$
Reward(x,y_\theta) = \beta log\frac{\pi_\theta(y_\theta|x)}{\pi_{ref}(y_\theta|x)}
$

$
Reward(x,y_{GT}) = \beta log\frac{\pi_\theta(y_{GT}|x)}{\pi_{ref}(y_{GT}|x)}
$

因此，对于生成器G来说，为了让判别器无法判断由$\pi_\theta$生成的样本和Ground Truth样本，生成器优化式子应该为

$
max_{(x \sim \mathcal{D}_{pdpo},y_\theta \sim\pi_\theta(y|x),y_{GT} \sim \mathcal{Y}_w)} E[\beta log\frac{\pi_\theta(y_\theta|x)}{\pi_{ref}(y_\theta|x)}-\beta log\frac{\pi_\theta(y_{GT}|x)}{\pi_{ref}(y_{GT}|x)}]
$

在上述式子中，只有计算$\pi_\theta(y_\theta|x)$是需要计算梯度的，其他项都是不需要计算梯度的，因为对于G来说，D是独立的，因此对$\pi_\theta(y_{GT}|x)$的计算也不需要计算梯度。因此PDPO方法没有增加较大的额外计算工作量。

总的来说，PDPO的思想为
1. 将$\pi_\theta$同时作为生成器和判别器
2. DPO的loss作为判别器的loss
3. 上述优化项目取负号，作为生成器的loss

因此PDPO训练中并没有新增任何模型，同时也能够将对$\pi_\theta$采样的数据运用到训练过程中。

### PDPO的优化改进
通过对上述PDPO中生成器的优化项易发现，生成器的目的是要不断缩小生成数据（负样本）和Ground Truth（正样本）数据的reward之间的差距。
由于负样本数据是基于当前$\pi_\theta$进行采样得到的（temperature、top_p等等），因此在对于上述优化项进行拟合优化的时候，非常容易导致第一项非常容易优化，导致由$\pi_\theta$生成数据的reward非常大，导致$\pi_\theta$丧失了正常输入文本的能力。

理想情况下在PDPO优化的过程中$Reward(x,y_\theta)<=Reward(x,y_{GT})$，作为生成器$\pi_\theta$的优化目标应该是尽可能使$Reward(x,y_{GT})-Reward(x,y_\theta)$降低，但是一旦$Reward(x,y_\theta)$高于$Reward(x,y_{GT})$，就无法确定$\pi_\theta$是朝着更优的方向优化，还是朝着更差的方向优化。因此，PDPO中生成器的优化项应该是在$Reward(x,y_\theta)<=Reward(x,y_{GT})$的时候起作用，即$Reward(x,y_{GT})-Reward(x,y_\theta)>0$。因此，上述PDPO的优化就变成了：

$
min_{(x \sim \mathcal{D}_{pdpo},y_\theta \sim\pi_\theta(y|x),y_{GT} \sim \mathcal{Y}_w)} E[max(\beta log\frac{\pi_\theta(y_{GT}|x)}{\pi_{ref}(y_{GT}|x)}-\beta log\frac{\pi_\theta(y_\theta|x)}{\pi_{ref}(y_\theta|x)},0)]
$

PDPO的生成器不是像GAN随机初始化生成器权重一样，PDPO里面的生成器是初始时本身时一个具有较好输出能力的模型，同时在优化PDPO的判别器的loss（DPO原始的loss）时，$\pi_\theta$的logps也在不断靠近$y_w$的分布，因此本文引入了一个新的参数$\gamma$用于平衡生成器和判别器的loss。

## 实验

### 数据集
实验数据集：Open AI GSM8K 数学数据集。该数据由7473条训练集和1319条测试集组成。在进行后续的实验中，数据的分配如下。
DPO使用训练集进行训练，正样本由ChatGPT-4o造取的数据，负样本由Qwen2-7B-Instruct造取的数据。
PDPO的训练集分为了判别器和生成器，判别器划分6000条数据进行训练，生成器划分1473条数据进行训练。6000条数据的正负样本来源于GPT-4o造取的回答和Qwen2-7B-Instruct回答。生成的正样本也是GPT-4o造取的回复，负样本为$\pi_\theta$生成数据。

### 实验结果
|#|#|
|-|-|
| Qwen2(base)|85.7% |
| DPO|87.11% |
| PDPO(ours)| **88.10**(+2.4)%|