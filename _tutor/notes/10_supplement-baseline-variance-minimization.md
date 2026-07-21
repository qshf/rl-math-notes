# 第 10 章补充：策略梯度中的基线与方差最小化

> 对应原书 10.2 Advantage actor-critic (A2C) 与 Box 10.1。  
> 本文解释三件事：为什么 baseline 基线不改变策略梯度的期望、为什么最小化方差可以转成最小化平方范数期望、以及最优 baseline 为什么在实践中常被 $v_\pi(s)$ 近似。

---

## 1. 问题从哪里来

策略梯度的基本形式是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{S\sim\eta,\ A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)
q_\pi(S,A)
\right].
$$

直接用样本估计时，一条样本给出的随机梯度是：

$$
g
=
\nabla_\theta\ln\pi(A\mid S,\theta)
q_\pi(S,A).
$$

问题是：$q_\pi(S,A)$ 是绝对动作价值，数值可能很大，样本波动也可能很大。为了降低方差，可以减去一个只依赖状态的 baseline 基线 $b(S)$：

$$
g_b
=
\nabla_\theta\ln\pi(A\mid S,\theta)
\left[
q_\pi(S,A)-b(S)
\right].
$$

这一步最关键的问题是：减去 $b(S)$ 会不会把策略梯度方向改掉？

答案是不会。只要 $b$ 不依赖当前采样动作 $A$，它只会改变样本方差，不会改变期望梯度。

---

## 2. 基线不改变梯度期望

先固定一个状态 $s$。需要证明被减掉的 baseline 项期望为 0：

$$
\mathbb{E}_{A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid s,\theta)b(s)
\right]
=0.
$$

因为 $b(s)$ 对当前状态下所有动作都是同一个数，可以提到动作期望外：

$$
\begin{aligned}
&\mathbb{E}_{A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid s,\theta)b(s)
\right] \\
&=
b(s)
\sum_{a\in\mathcal{A}}
\pi(a\mid s,\theta)
\nabla_\theta\ln\pi(a\mid s,\theta).
\end{aligned}
$$

用 log-derivative trick 对数求导技巧：

$$
\nabla_\theta\ln\pi(a\mid s,\theta)
=
\frac{\nabla_\theta\pi(a\mid s,\theta)}
{\pi(a\mid s,\theta)}.
$$

代入后：

$$
\begin{aligned}
&b(s)
\sum_{a\in\mathcal{A}}
\pi(a\mid s,\theta)
\frac{\nabla_\theta\pi(a\mid s,\theta)}
{\pi(a\mid s,\theta)} \\
&=
b(s)
\sum_{a\in\mathcal{A}}
\nabla_\theta\pi(a\mid s,\theta) \\
&=
b(s)
\nabla_\theta
\sum_{a\in\mathcal{A}}
\pi(a\mid s,\theta) \\
&=
b(s)\nabla_\theta 1
=0.
\end{aligned}
$$

所以：

$$
\mathbb{E}_{A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid s,\theta)
\left(q_\pi(s,A)-b(s)\right)
\right]
=
\mathbb{E}_{A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid s,\theta)q_\pi(s,A)
\right].
$$

如果再对状态 $S\sim\eta$ 求期望，结论仍然成立：

$$
\mathbb{E}_{S\sim\eta,\ A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)b(S)
\right]
=0.
$$

读法：baseline 对同一个状态下所有动作都一样，它不会偏向任何一个动作；而所有动作概率之和永远是 1，所以它对期望梯度没有贡献。

⚠️ 易错点：baseline 可以依赖状态 $S$，但不能依赖当前样本动作 $A$。如果写成 $b(S,A)$，它就可能改变不同动作之间的相对更新权重，期望梯度不再保证不变。

---

## 3. 为什么最小化方差等价于最小化平方范数期望

令加 baseline 后的随机梯度为：

$$
X
\doteq
\nabla_\theta\ln\pi(A\mid S,\theta)
\left[
q_\pi(S,A)-b(S)
\right].
$$

由于上一节已经证明 $\mathbb{E}[X]$ 不随 $b(S)$ 改变，所以随机梯度的均值是常数。对向量随机变量 $X$，常用方差矩阵的迹作为标量目标：

$$
\operatorname{tr}[\operatorname{var}(X)]
=
\mathbb{E}
\left[
\|X\|^2
\right]
-
\left\|
\mathbb{E}[X]
\right\|^2.
$$

第二项 $\|\mathbb{E}[X]\|^2$ 与 baseline 无关，因此选择最小方差 baseline 等价于最小化：

$$
\mathbb{E}
\left[
\|X\|^2
\right].
$$

把 $X$ 代入：

$$
\begin{aligned}
\mathbb{E}
\left[
\|X\|^2
\right]
&=
\mathbb{E}
\left[
\left\|
\nabla_\theta\ln\pi(A\mid S,\theta)
\right\|^2
\left(q_\pi(S,A)-b(S)\right)^2
\right].
\end{aligned}
$$

因为 baseline 是按状态选的，所以可以固定某个状态 $s$，单独最小化该状态下的条件目标：

$$
J_s(b)
=
\mathbb{E}_{A\sim\pi}
\left[
\left\|
\nabla_\theta\ln\pi(A\mid s,\theta)
\right\|^2
\left(q_\pi(s,A)-b(s)\right)^2
\right].
$$

---

## 4. 最优基线的推导

为了简化记号，固定状态 $s$ 后记：

$$
W(A)
\doteq
\left\|
\nabla_\theta\ln\pi(A\mid s,\theta)
\right\|^2,
\qquad
Q(A)
\doteq
q_\pi(s,A).
$$

于是目标函数变成：

$$
J_s(b)
=
\mathbb{E}_{A\sim\pi}
\left[
W(A)(Q(A)-b(s))^2
\right].
$$

对 $b(s)$ 求导：

$$
\frac{\partial J_s(b)}
{\partial b(s)}
=
\mathbb{E}_{A\sim\pi}
\left[
-2W(A)(Q(A)-b(s))
\right].
$$

令导数为 0：

$$
\mathbb{E}_{A\sim\pi}
\left[
W(A)(Q(A)-b(s))
\right]
=0.
$$

展开：

$$
\mathbb{E}_{A\sim\pi}
\left[
W(A)Q(A)
\right]
-
b(s)
\mathbb{E}_{A\sim\pi}
\left[
W(A)
\right]
=0.
$$

因此：

$$
b^*(s)
=
\frac{
\mathbb{E}_{A\sim\pi}
\left[
W(A)Q(A)
\right]
}{
\mathbb{E}_{A\sim\pi}
\left[
W(A)
\right]
}.
$$

把 $W(A)$ 和 $Q(A)$ 换回原来的符号，就得到最优 baseline：

$$
\boxed{
b^*(s)
=
\frac{
\mathbb{E}_{A\sim\pi}
\left[
\left\|
\nabla_\theta\ln\pi(A\mid s,\theta)
\right\|^2
q_\pi(s,A)
\right]
}{
\mathbb{E}_{A\sim\pi}
\left[
\left\|
\nabla_\theta\ln\pi(A\mid s,\theta)
\right\|^2
\right]
}.
}
$$

读法：这个 baseline 不是普通平均，而是按动作对参数的敏感度加权。哪个动作的 score function $\nabla_\theta\ln\pi(A\mid s,\theta)$ 范数越大，它越容易放大随机梯度波动，于是最优 baseline 会更重视这个动作的 $q_\pi(s,A)$。

---

## 5. 加权平均视角

如果只记住结构，可以把公式压缩成：

$$
b^*
=
\frac{\mathbb{E}[WQ]}{\mathbb{E}[W]}.
$$

其中：

| 记号 | 含义 |
|---|---|
| $Q=q_\pi(s,A)$ | 当前状态下动作 $A$ 的动作价值 |
| $W=\|\nabla_\theta\ln\pi(A\mid s,\theta)\|^2$ | 该动作对策略参数的敏感度权重 |

所以 $b^*(s)$ 是一个加权平均。它会更贴近那些“对梯度方差影响更大”的动作价值，从而尽量缩小高敏感动作上的差值 $q_\pi(s,A)-b(s)$。

直觉上，baseline 的作用不是判断动作好坏，而是把每个动作价值都减去一个参照物，使随机梯度的波动变小。最优 baseline 选择的是最能压低这种波动的参照物。

---

## 6. 从最优基线到实用基线

最优 baseline $b^*(s)$ 在数学上最小化方差，但实践中并不方便。原因是它需要计算：

$$
\left\|
\nabla_\theta\ln\pi(A\mid s,\theta)
\right\|^2
$$

并对动作分布求加权期望。这在大动作空间或神经网络策略中会增加计算开销。

于是实践中常用一个更便宜的近似：去掉敏感度权重 $W(A)$。这样：

$$
b^\dagger(s)
=
\mathbb{E}_{A\sim\pi}
\left[
q_\pi(s,A)
\right].
$$

根据第 2.8 节 state value 状态价值与 action value 动作价值的关系：

$$
v_\pi(s)
=
\sum_{a\in\mathcal{A}}
\pi(a\mid s)
q_\pi(s,a)
=
\mathbb{E}_{A\sim\pi}
\left[
q_\pi(s,A)
\right].
$$

所以：

$$
\boxed{
b^\dagger(s)=v_\pi(s).
}
$$

这就是为什么 A2C、PPO 等算法常用 value function $V(s)$ 作为 baseline。

---

## 7. 和 advantage function 的关系

当 baseline 取成 $v_\pi(s)$ 时，策略梯度中的动作价值项变成：

$$
q_\pi(s,a)-v_\pi(s).
$$

这就是 advantage function 优势函数：

$$
A_\pi(s,a)
\doteq
q_\pi(s,a)-v_\pi(s).
$$

读法：优势函数衡量的是“这个动作比当前状态下的平均动作好多少”。

| 情况 | 含义 | actor 的倾向 |
|---|---|---|
| $A_\pi(s,a)>0$ | 动作比平均水平好 | 增大该动作概率 |
| $A_\pi(s,a)<0$ | 动作比平均水平差 | 减小该动作概率 |
| $A_\pi(s,a)=0$ | 动作接近平均水平 | 更新信号较弱 |

所以 baseline 的意义可以压缩成一句话：

> 不改变策略梯度的期望方向，只把“绝对动作价值”改写成“相对平均水平的动作优势”，从而降低样本更新的方差。

---

## 8. 小结

| 问题 | 结论 |
|---|---|
| 为什么可以减 baseline？ | 因为 $\mathbb{E}_{A\sim\pi}[\nabla_\theta\ln\pi(A\mid s,\theta)b(s)]=0$ |
| baseline 改变了什么？ | 不改期望梯度，只可能改变方差 |
| 最优 baseline 是什么？ | $b^*(s)=\dfrac{\mathbb{E}[Wq_\pi(s,A)]}{\mathbb{E}[W]}$ |
| $W$ 是什么？ | $W=\|\nabla_\theta\ln\pi(A\mid s,\theta)\|^2$，表示动作对参数的敏感度 |
| 实践中常用什么？ | 去掉权重后得到 $b^\dagger(s)=v_\pi(s)$ |
| 为什么引出 advantage？ | 因为 $q_\pi(s,a)-v_\pi(s)=A_\pi(s,a)$，表示动作相对平均水平的优势 |

