在《Direct Preference Optimization:Your Language Model is Secretly a Reward Model》一文中，该文章作者将RLHF的RL强化学习阶段的过程描述为：

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
\max_{\pi_\theta}
$
