# 第 6 章补充：6.1-6.4 结构梳理

> 原书 p112-132 · 学习日期 2026-06-08 · 更新 2026-06-09 · 对应 6.1-6.4

## 要解决的问题

6.1、6.2、6.3 看起来像从“算平均”跳到“求根”再跳到“收敛定理”；6.4 又突然跳到 stochastic gradient descent 随机梯度下降。这篇补充把它们串成一条线：先得到一个在线更新式，再把它放进 RM 框架，用 Dvoretzky 定理证明它为什么会收敛，最后说明 SGD 也能包装成同一个带噪声求根问题。

## 一句话总览

这些小节其实在回答同一个问题：

> 如果我只能不断拿到随机样本，而看不到真实期望或真实函数，怎样在线更新一个估计值，并证明它会收敛？

答案分三步：

| 小节 | 它在解决什么 | 得到什么 |
|---|---|---|
| 6.1 | 从样本平均出发，写成每来一个样本就更新一次 | 增量更新式 |
| 6.2 | 把这个更新式抽象成“带噪声求根” | Robbins-Monro algorithm |
| 6.3 | 给这种随机迭代一个收敛证明模板 | Dvoretzky's theorem |
| 6.4 | 把 SGD 也写成“带噪声求根” | SGD 是 RM 的特例 |

用箭头串起来：

$$
\text{样本平均}
\Longrightarrow
\text{增量均值更新}
\Longrightarrow
\text{带噪声求根 RM}
\Longrightarrow
\text{误差递推 Dvoretzky}
\Longrightarrow
\text{证明收敛}
\Longrightarrow
\text{SGD 也回到 RM}.
$$

## 6.1：先把“平均”写成在线更新

原始任务是：

$$
\text{Given samples }x_1,x_2,\ldots,\quad \text{estimate }\mathbb E[X].
$$

第 5 章已经说过，可以用样本平均估计期望：

$$
\bar x=\frac1n\sum_{i=1}^n x_i.
$$

6.1 的新问题是：不想每次都重新求和，能不能每来一个样本就更新一次？

答案是可以。把样本平均改写成：

$$
w_{k+1}
=
w_k-\frac1k(w_k-x_k)
=
w_k+\frac1k(x_k-w_k).
$$

再推广成一般步长：

$$
w_{k+1}=w_k+\alpha_k(x_k-w_k).
$$

这里三个符号的角色是：

| 符号 | 含义 |
|---|---|
| $x_k$ | 第 $k$ 次采到的样本 |
| $\mathbb E[X]$ | 真正想估计的未知均值 |
| $w_k$ | 第 $k$ 步对 $\mathbb E[X]$ 的当前估计 |

所以 6.1 的核心不是“均值公式本身”，而是得到一个形状很重要的更新式：

$$
\text{新估计}
=
\text{旧估计}
+
\text{步长}
\times
(\text{样本}-\text{旧估计}).
$$

## 6.2：再把“估计均值”包装成“求根问题”

6.2 的 Robbins-Monro algorithm 研究的是求根：

$$
g(w)=0.
$$

那均值估计为什么和 $g(w)$ 有关系？

因为“估计 $\mathbb E[X]$”等价于“找一个 $w$，使得 $w=\mathbb E[X]$”。于是人为定义：

$$
g(w)\doteq w-\mathbb E[X].
$$

那么：

$$
g(w)=0
\Longleftrightarrow
w-\mathbb E[X]=0
\Longleftrightarrow
w=\mathbb E[X].
$$

所以求 $g(w)=0$ 的根，就是找 $\mathbb E[X]$。

问题是我们不知道 $\mathbb E[X]$，所以也不能直接算：

$$
g(w)=w-\mathbb E[X].
$$

但我们能拿到样本 $x_k$，于是用样本临时代替未知均值，得到带噪声观测：

$$
\tilde g(w_k,\eta_k)=w_k-x_k.
$$

它平均起来就是 $g(w_k)$：

$$
\mathbb E[w_k-x_k|\mathcal H_k]
=
w_k-\mathbb E[X]
=
g(w_k).
$$

这样 6.1 的更新式：

$$
w_{k+1}=w_k-\alpha_k(w_k-x_k)
$$

就变成 RM 形式：

$$
w_{k+1}=w_k-\alpha_k\tilde g(w_k,\eta_k).
$$

所以 6.2 的核心是：**6.1 的均值估计不是孤立技巧，它是 Robbins-Monro 带噪声求根算法的一个特例。**

## 补充：6.4 如何把“优化问题”包装成“求根问题”

6.4 的 stochastic gradient descent 随机梯度下降 研究的是优化：

$$
\min_w J(w)=\mathbb E[f(w,X)].
$$

那优化问题为什么和 6.2 的 $g(w)=0$ 有关系？

因为“最小化 $J(w)$”通常可以转成“找一个 $w$，使得梯度为 0”。也就是找：

$$
\nabla_wJ(w)=0.
$$

在凸优化或书中给出的曲率条件下，这个梯度为 0 的点就是最优解。于是人为定义：

$$
g(w)\doteq \nabla_wJ(w).
$$

而 6.4 的目标函数是期望形式：

$$
J(w)=\mathbb E[f(w,X)].
$$

所以：

$$
g(w)
=
\nabla_wJ(w)
=
\nabla_w\mathbb E[f(w,X)]
=
\mathbb E[\nabla_w f(w,X)].
$$

那么：

$$
g(w)=0
\Longleftrightarrow
\mathbb E[\nabla_w f(w,X)]=0
\Longleftrightarrow
\nabla_wJ(w)=0.
$$

所以求 $g(w)=0$ 的根，就是找 $J(w)$ 的 stationary point 驻点；在本节的条件下，也就是找最优参数 $w^*$。

问题是我们不知道 $X$ 的完整分布，所以也不能直接算：

$$
g(w)=\mathbb E[\nabla_w f(w,X)].
$$

但我们能拿到样本 $x_k$，于是用单个样本的梯度临时代替真实期望梯度，得到带噪声观测：

$$
\tilde g(w_k,\eta_k)
=
\nabla_w f(w_k,x_k).
$$

它平均起来就是 $g(w_k)$：

$$
\mathbb E[\nabla_w f(w_k,x_k)|\mathcal H_k]
=
\mathbb E[\nabla_w f(w_k,X)]
=
g(w_k).
$$

这一步可以这样读：

$$
\mathcal H_k
=
\text{第 }k\text{ 步之前的历史信息}.
$$

更具体地说，$\mathcal H_k$ 可以理解成“走到第 $k$ 步时，算法已经知道的一切”。它通常包含：

$$
\mathcal H_k
=
\{w_k,w_{k-1},\ldots;\ x_{k-1},x_{k-2},\ldots;\ \alpha_{k-1},\alpha_{k-2},\ldots\}.
$$

如果是在 RM 或 Dvoretzky 定理的写法里，它也常写成包含过去误差、过去噪声、过去步长的历史集合：

$$
\mathcal H_k
=
\{\Delta_k,\Delta_{k-1},\ldots;\ \eta_{k-1},\eta_{k-2},\ldots;\ \alpha_{k-1},\alpha_{k-2},\ldots\}.
$$

不要把它理解成普通集合运算里的“随便收集一些符号”。它更准确地表示 filtration 滤过，也就是“到当前时刻为止可用的信息”。写条件期望

$$
\mathbb E[\cdot|\mathcal H_k]
$$

就是在说：**假设过去发生的一切都已经知道了，只对下一步还没发生的随机性取平均。**

所以，给定 $\mathcal H_k$ 以后，当前参数 $w_k$ 已经由过去的样本和更新决定，在条件期望里可以把 $w_k$ 当成固定值。真正还随机的是第 $k$ 步新抽到的样本 $x_k$。

⚠️ 注意：$\mathcal H_k$ 包含第 $k$ 步之前的信息，但不包含当前刚抽到的 $x_k$。如果它已经包含 $x_k$，那 $\mathbb E[\nabla_w f(w_k,x_k)|\mathcal H_k]$ 就不会再对 $x_k$ 做平均了。

因此：

$$
\mathbb E[\nabla_w f(w_k,x_k)|\mathcal H_k]
$$

表示“固定当前参数 $w_k$，只对新样本 $x_k$ 的随机性取平均”。

又因为 $x_k$ 和随机变量 $X$ 来自同一个分布，并且 $x_k$ 与历史 $\mathcal H_k$ 独立，所以对 $x_k$ 取平均，就等于对 $X$ 取平均：

$$
\mathbb E[\nabla_w f(w_k,x_k)|\mathcal H_k]
=
\mathbb E[\nabla_w f(w_k,X)].
$$

而前面已经定义：

$$
g(w)=\mathbb E[\nabla_w f(w,X)].
$$

把 $w=w_k$ 代进去，就得到：

$$
\mathbb E[\nabla_w f(w_k,X)]=g(w_k).
$$

所以这整步的意思是：**单样本梯度 $\nabla_w f(w_k,x_k)$ 虽然是随机的，但它在条件平均意义下等于真实梯度 $g(w_k)$。** 也就是说，单样本梯度是 true gradient 真实梯度 的 unbiased estimate 无偏估计。

也就是说，单样本梯度可以拆成：

$$
\nabla_w f(w_k,x_k)
=
\underbrace{\mathbb E[\nabla_w f(w_k,X)]}_{g(w_k)}
+
\underbrace{\left(
\nabla_w f(w_k,x_k)-\mathbb E[\nabla_w f(w_k,X)]
\right)}_{\eta_k}.
$$

所以：

$$
\tilde g(w_k,\eta_k)=g(w_k)+\eta_k.
$$

这样 6.4 的 SGD 更新式：

$$
w_{k+1}
=
w_k-\alpha_k\nabla_w f(w_k,x_k)
$$

就变成 RM 形式：

$$
w_{k+1}
=
w_k-\alpha_k\tilde g(w_k,\eta_k).
$$

所以 6.4 的核心是：**SGD 不是突然冒出来的新框架，它是 Robbins-Monro 带噪声求根算法的一个特例；只不过这次要求的根不是 $w-\mathbb E[X]=0$，而是 $\nabla_wJ(w)=0$。**

和 6.2 对照看：

| 问题 | 真正想解的方程 | 看不到什么 | 实际用什么观测 | RM 更新 |
|---|---|---|---|---|
| 6.2 均值估计 | $g(w)=w-\mathbb E[X]=0$ | $\mathbb E[X]$ | $\tilde g(w_k,\eta_k)=w_k-x_k$ | $w_{k+1}=w_k-\alpha_k(w_k-x_k)$ |
| 6.4 SGD | $g(w)=\nabla_wJ(w)=0$ | $\mathbb E[\nabla_w f(w,X)]$ | $\tilde g(w_k,\eta_k)=\nabla_w f(w_k,x_k)$ | $w_{k+1}=w_k-\alpha_k\nabla_w f(w_k,x_k)$ |

这就是 6.4 和 6.2 的同构关系：

$$
\text{未知均值}
\quad\leftrightarrow\quad
\text{未知真实梯度},
$$

$$
\text{样本 }x_k
\quad\leftrightarrow\quad
\text{样本梯度 }\nabla_w f(w_k,x_k).
$$

## 6.3：最后证明误差真的会归零

6.2 告诉我们 RM 算法长什么样，但还没有证明它收敛。6.3 的 Dvoretzky 定理就是证明工具。

证明收敛时，不直接盯着 $w_k$，而是盯着误差：

$$
\Delta_k=\text{当前估计}-\text{真实目标}.
$$

如果能证明：

$$
\Delta_k\to0,
$$

就说明当前估计收敛到了真实目标。

Dvoretzky 定理研究的统一模板是：

$$
\Delta_{k+1}
=
(1-\alpha_k)\Delta_k+\beta_k\eta_k.
$$

这个式子可以读成：

$$
\text{新误差}
=
\underbrace{\text{旧误差被压小}}_{(1-\alpha_k)\Delta_k}
+
\underbrace{\text{新噪声影响}}_{\beta_k\eta_k}.
$$

只要满足：

$$
\sum\alpha_k=\infty,
\qquad
\sum\alpha_k^2<\infty,
\qquad
\sum\beta_k^2<\infty,
$$

并且噪声条件均值为 0、方差有界，就能得到：

$$
\Delta_k\to0.
$$

于是 6.3 做了两次“套模板”：

**套均值估计**

令：

$$
w^*=\mathbb E[X],
\qquad
\Delta_k=w_k-w^*.
$$

从：

$$
w_{k+1}=w_k+\alpha_k(x_k-w_k)
$$

推出：

$$
\Delta_{k+1}
=
(1-\alpha_k)\Delta_k+\alpha_k(x_k-w^*).
$$

这就是 Dvoretzky 模板。

**套 Robbins-Monro**

令 $w^*$ 是根：

$$
g(w^*)=0,
\qquad
\Delta_k=w_k-w^*.
$$

从：

$$
w_{k+1}=w_k-a_k[g(w_k)+\eta_k]
$$

利用中值定理改写成：

$$
\Delta_{k+1}
=
\left[1-a_k\nabla_wg(w_k')\right]\Delta_k+a_k(-\eta_k).
$$

这也变成 Dvoretzky 模板。

所以 6.3 的核心是：**只要能把某个随机迭代的误差改写成 Dvoretzky 模板，就能用同一套条件证明它收敛。**

## 补充：拟鞅收敛定理在这里做什么

6.3 证明 Dvoretzky 定理时，会用到 quasimartingale convergence theorem 拟鞅收敛定理。它可以先理解成一句话：

> 一个非负随机序列即使不是单调下降，只要它“往上跳的总量”是有限的，它也会收敛。

先看确定性版本。一个非负序列 $x_k$ 如果不是单调的，但所有“上涨部分”加起来有限：

$$
\sum_{k=1}^{\infty}(x_{k+1}-x_k)^+<\infty,
$$

那么它会收敛。直觉是：它不能无限次大幅往上跳；另一方面它又非负，不会掉到 $-\infty$。

拟鞅收敛定理是这件事的随机版本。随机序列不能直接只看 $X_{k+1}-X_k$，因为下一步本身是随机的；我们看的是“知道历史 $\mathcal H_k$ 后，下一步在条件期望意义下会不会上涨”。

定义事件：

$$
A_k
\doteq
\left\{
\mathbb E[X_{k+1}-X_k|\mathcal H_k]\ge0
\right\}.
$$

它表示：第 $k$ 步在条件期望意义下是往上涨的。

这里的

$$
\mathbf 1_{A_k}
$$

读作事件 $A_k$ 的 indicator function 指示函数。它不是数字 1 乘以 $A_k$，而是一个“开关”：

$$
\mathbf 1_{A_k}
=
\begin{cases}
1, & \text{如果事件 }A_k\text{ 发生},\\
0, & \text{如果事件 }A_k\text{ 不发生}.
\end{cases}
$$

所以：

$$
(X_{k+1}-X_k)\mathbf 1_{A_k}
$$

表示只保留事件 $A_k$ 发生时的增量；如果 $A_k$ 不发生，这一项就变成 0。也就是说，它专门筛选“条件期望意义下上涨”的那部分。

拟鞅收敛定理说：若 $X_k\ge0$，并且这些“期望上涨部分”的总量有限：

$$
\sum_{k=1}^{\infty}
\mathbb E[(X_{k+1}-X_k)\mathbf 1_{A_k}]
<\infty,
$$

那么 $X_k$ 几乎必然收敛。

放回 6.3 里，取：

$$
X_k=h_k=\Delta_k^2.
$$

它一定非负。前面的推导给出：

$$
\mathbb E[h_{k+1}-h_k|\mathcal H_k]
\le
\beta_k^2C.
$$

又因为：

$$
\sum_k\beta_k^2<\infty,
$$

所以 $h_k$ 的“期望上涨总额度”被控制住了。换句话说，**平方误差**虽然可以偶尔上升，但它期望意义下的上升幅度总量有限。

于是拟鞅收敛定理帮我们得到：

$$
h_k=\Delta_k^2
$$

会收敛到某个有限随机变量。

注意，它只负责证明：

$$
h_k\text{ 有极限}.
$$

它还没有证明这个极限是 0。后面还要继续用：

$$
\sum_k\alpha_k=\infty
$$

和：

$$
\sum_k\alpha_k\Delta_k^2<\infty
$$

把这个极限逼成 0。

## 这些小节的符号关系

| 符号 | 在 6.1 里的角色 | 在 6.2/6.3 里的角色 | 在 6.4 里的角色 |
|---|---|---|---|
| $x_k$ | 第 $k$ 个样本 | 构造带噪声观测的材料 | 构造单样本梯度的材料 |
| $w_k$ | 当前均值估计 | 当前根/目标的估计 | 当前参数估计 |
| $\mathbb E[X]$ | 真正要估计的均值 | 均值估计里的真实目标 $w^*$ | 均值估计作为 SGD 特例时的最优解 |
| $J(w)$ | 暂时没有 | 可通过 $\nabla_wJ(w)=0$ 转成求根 | 要最小化的目标函数 |
| $g(w)$ | 暂时没有 | 把目标包装成求根问题 | $g(w)=\nabla_wJ(w)=\mathbb E[\nabla_w f(w,X)]$ |
| $\tilde g(w,\eta)$ | 暂时没有 | $g(w)$ 的带噪声观测 | 单样本梯度 $\nabla_w f(w,x)$ |
| $\Delta_k$ | 暂时没有 | 当前估计和真实目标之间的误差 | 收敛证明时仍可表示 $w_k-w^*$ |
| $\alpha_k$ 或 $a_k$ | 更新步长 | 控制收缩和噪声影响的步长 | SGD 的 learning rate 学习率 |

## 读这些小节时最重要的一条线

不要把 6.1、6.2、6.3、6.4 看成几个互不相干的新主题。它们是一条逐步抽象、再回收工具的链：

$$
\begin{aligned}
&\text{6.1：我想估计均值 } \mathbb E[X]\\
&\Downarrow\\
&\text{6.2：把估计均值写成求 } g(w)=0\\
&\Downarrow\\
&\text{6.3：把求根算法的误差写成 } \Delta_{k+1}=(1-\alpha_k)\Delta_k+\beta_k\eta_k\\
&\Downarrow\\
&\text{证明 } \Delta_k\to0,\text{ 也就是 } w_k\text{ 收敛到目标}\\
&\Downarrow\\
&\text{6.4：把优化问题 } \min_wJ(w)\text{ 写成求 } \nabla_wJ(w)=0\\
&\Downarrow\\
&\text{用单样本梯度当带噪声观测，得到 SGD 是 RM 的特例。}
\end{aligned}
$$

这一条线抓住后，6.3 那些看起来突然出现的 $\Delta_k,\alpha_k,\beta_k,\eta_k$ 就有位置了：它们不是凭空来的，而是为了证明“当前估计离真实目标越来越近”。6.4 的 SGD 也不再是新话题，而是把“真实函数值看不到，只能用带噪声观测替代”这件事，从均值估计推广到了梯度优化。


