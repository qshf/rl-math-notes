# 补充：Lemma 8.1 中 b - Aw_t 的详细推导

> 对应原书 8.2.5 / Box 8.3 · 学习日期 2026-06-16

本补充要解释为什么：

$$
\mathbb E\!\left[
\left(
r_{t+1}
+\gamma\phi^T(s_{t+1})w_t
-\phi^T(s_t)w_t
\right)\phi(s_t)
\right]
=
b-Aw_t.
$$

## 0. 要解决的问题

我们要看懂这一行：

$$
\mathbb {E} \left[
\left(
r _ {t + 1}
+ \gamma \phi^ {T} (s _ {t + 1}) w _ {t}
- \phi^ {T} (s _ {t}) w _ {t}
\right)
\phi (s _ {t})
\right]
= b - A w _ {t}.
$$

其中：

$$
A \doteq \Phi^T D(I-\gamma P_\pi)\Phi,
\qquad
b \doteq \Phi^T D r_\pi.
$$

这一步的作用是：把“随机样本的平均 TD 更新方向”写成一个简单的线性形式：

$$
b-Aw_t.
$$

这样后面的确定性 TD 更新就能写成：

$$
w_{t+1}=w_t+\alpha_t(b-Aw_t).
$$

## 1. 先把所有符号和维度摆清楚

设状态数是 $n$，特征维度是 $m$。

| 符号 | 含义 | 维度 |
|---|---|---:|
| $s_t$ | 时刻 $t$ 的状态 | 一个状态 |
| $s_{t+1}$ | 下一时刻状态 | 一个状态 |
| $r_{t+1}$ | 从 $s_t$ 到 $s_{t+1}$ 得到的奖励 | scalar |
| $\phi(s)$ | 状态 $s$ 的 feature vector 特征向量 | $m\times 1$ |
| $\phi^T(s)$ | 特征行向量 | $1\times m$ |
| $w_t$ | 当前参数 | $m\times 1$ |
| $\phi^T(s)w_t$ | 状态 $s$ 的估计价值 | scalar |
| $P_\pi$ | 策略 $\pi$ 下的状态转移矩阵 | $n\times n$ |
| $[P_\pi]_{ss'}$ | 从 $s$ 到 $s'$ 的概率 | scalar |
| $d_\pi(s)$ | 策略 $\pi$ 下长期访问状态 $s$ 的概率 | scalar |
| $D$ | 平稳分布对角矩阵 | $n\times n$ |
| $r_\pi$ | 每个状态的期望即时奖励向量 | $n\times 1$ |
| $\Phi$ | 所有状态特征按行叠成的矩阵 | $n\times m$ |

其中：

$$
\Phi
=
\begin{bmatrix}
\phi^T(s_1)\\
\phi^T(s_2)\\
\vdots\\
\phi^T(s_n)
\end{bmatrix}
\in\mathbb R^{n\times m},
\qquad
D=
\operatorname{diag}(d_\pi(s_1),\dots,d_\pi(s_n)).
$$

$r_\pi$ 的第 $s$ 个分量是：

$$
r_\pi(s)
=
\mathbb E[r_{t+1}\mid s_t=s].
$$

它的意思是：当前状态固定为 $s$ 时，下一步奖励的期望。

## 2. 原式先拆成 reward 项和 TD 项

原式是：

$$
\mathbb {E} \left[
\left(
r _ {t + 1}
+ \gamma \phi^ {T} (s _ {t + 1}) w _ {t}
- \phi^ {T} (s _ {t}) w _ {t}
\right)
\phi (s _ {t})
\right].
$$

把括号乘到 $\phi(s_t)$ 上：

$$
\mathbb E\left[
r_{t+1}\phi(s_t)
+
\left(
\gamma\phi^T(s_{t+1})w_t
-
\phi^T(s_t)w_t
\right)\phi(s_t)
\right].
$$

为了和原书矩阵形式一致，通常把第二项写成：

$$
\phi(s_t)
\left(
\gamma\phi^T(s_{t+1})
-
\phi^T(s_t)
\right)w_t.
$$

于是整体拆成：

$$
\underbrace{
\mathbb E[r_{t+1}\phi(s_t)]
}_{\text{reward 项}}
+
\underbrace{
\mathbb E\left[
\phi(s_t)
\left(
\gamma\phi^T(s_{t+1})
-
\phi^T(s_t)
\right)w_t
\right]
}_{\text{TD 项}}.
$$

⚠️ **这里有一个容易迷糊的方向问题。**  
$\phi(s_t)$ 是 $m\times 1$ 列向量，$\phi^T(s_{t+1})$ 是 $1\times m$ 行向量，所以：

$$
\phi(s_t)\phi^T(s_{t+1})
$$

是 $m\times m$ 矩阵，再乘 $w_t$ 才得到 $m\times 1$ 向量。这个方向正好和参数更新方向一致。

## 3. 为什么可以按 $s_t=s$ 分解

这里使用 law of total expectation 全期望公式：

$$
\mathbb E[X]
=
\sum_{s\in\mathcal S}P(s_t=s)\mathbb E[X\mid s_t=s].
$$

在这段理论分析中，原书假设：

$$
s_t\sim d_\pi.
$$

所以：

$$
P(s_t=s)=d_\pi(s).
$$

因此原式可以写成：

$$
\sum_{s\in\mathcal S}
d_\pi(s)
\mathbb E\left[
\left(
r_{t+1}
+\gamma\phi^T(s_{t+1})w_t
-\phi^T(s_t)w_t
\right)
\phi(s_t)
\mid s_t=s
\right].
$$

一旦条件里固定了 $s_t=s$，那么：

$$
\phi(s_t)=\phi(s)
$$

就不再是随机的，可以从条件期望里拿出来。

## 4. reward 项为什么等于 $\Phi^TDr_\pi$

先看 reward 项：

$$
\mathbb E[r_{t+1}\phi(s_t)].
$$

按 $s_t=s$ 分解：

$$
\mathbb E[r_{t+1}\phi(s_t)]
=
\sum_{s\in\mathcal S}
d_\pi(s)
\mathbb E[r_{t+1}\phi(s_t)\mid s_t=s].
$$

在条件 $s_t=s$ 下，$\phi(s_t)=\phi(s)$ 是常量：

$$
\mathbb E[r_{t+1}\phi(s_t)\mid s_t=s]
=
\phi(s)\mathbb E[r_{t+1}\mid s_t=s].
$$

而：

$$
\mathbb E[r_{t+1}\mid s_t=s]
=r_\pi(s).
$$

所以：

$$
\mathbb E[r_{t+1}\phi(s_t)]
=
\sum_{s\in\mathcal S}
d_\pi(s)\phi(s)r_\pi(s).
$$

现在看它为什么等于矩阵形式 $\Phi^TDr_\pi$。

先算：

$$
Dr_\pi
=
\begin{bmatrix}
d_\pi(s_1)r_\pi(s_1)\\
d_\pi(s_2)r_\pi(s_2)\\
\vdots\\
d_\pi(s_n)r_\pi(s_n)
\end{bmatrix}.
$$

再左乘：

$$
\Phi^T
=
\begin{bmatrix}
\phi(s_1) & \phi(s_2) & \cdots & \phi(s_n)
\end{bmatrix}.
$$

于是：

$$
\Phi^TDr_\pi
=
\begin{bmatrix}
\phi(s_1) & \phi(s_2) & \cdots & \phi(s_n)
\end{bmatrix}
\begin{bmatrix}
d_\pi(s_1)r_\pi(s_1)\\
d_\pi(s_2)r_\pi(s_2)\\
\vdots\\
d_\pi(s_n)r_\pi(s_n)
\end{bmatrix}.
$$

矩阵乘法展开就是：

$$
\Phi^TDr_\pi
=
\sum_{s\in\mathcal S}
d_\pi(s)\phi(s)r_\pi(s).
$$

所以 reward 项：

$$
\mathbb E[r_{t+1}\phi(s_t)]
=
\Phi^TDr_\pi
\doteq b.
$$

## 5. TD 项为什么等于 $-\Phi^TD(I-\gamma P_\pi)\Phi w_t$

TD 项是：

$$
\mathbb E\left[
\phi(s_t)
\left(
\gamma\phi^T(s_{t+1})
-
\phi^T(s_t)
\right)w_t
\right].
$$

同样按 $s_t=s$ 分解：

$$
\sum_{s\in\mathcal S}
d_\pi(s)
\mathbb E\left[
\phi(s_t)
\left(
\gamma\phi^T(s_{t+1})
-
\phi^T(s_t)
\right)w_t
\mid s_t=s
\right].
$$

条件 $s_t=s$ 后，$\phi(s_t)=\phi(s)$：

$$
\sum_{s\in\mathcal S}
d_\pi(s)
\phi(s)
\mathbb E\left[
\gamma\phi^T(s_{t+1})
-
\phi^T(s)
\mid s_t=s
\right]
w_t.
$$

拆开：

$$
\sum_{s\in\mathcal S}
d_\pi(s)\phi(s)
\left[
\gamma\mathbb E[\phi^T(s_{t+1})\mid s_t=s]
-
\phi^T(s)
\right]
w_t.
$$

下一状态 $s_{t+1}$ 按 $P_\pi$ 的第 $s$ 行分布：

$$
P(s_{t+1}=s'\mid s_t=s)=P_\pi(s,s').
$$

所以：

$$
\mathbb E[\phi^T(s_{t+1})\mid s_t=s]
=
\sum_{s'\in\mathcal S}
P_\pi(s,s')\phi^T(s').
$$

代回：

$$
\sum_{s\in\mathcal S}
d_\pi(s)\phi(s)
\left[
\gamma
\sum_{s'\in\mathcal S}
P_\pi(s,s')\phi^T(s')
-
\phi^T(s)
\right]w_t.
$$

现在把它拆成两块：

$$
\gamma
\sum_s
d_\pi(s)\phi(s)
\left[
\sum_{s'}P_\pi(s,s')\phi^T(s')
\right]w_t
-
\sum_s
d_\pi(s)\phi(s)\phi^T(s)w_t.
$$

第二块比较容易：

$$
\sum_s d_\pi(s)\phi(s)\phi^T(s)
=
\Phi^TD\Phi.
$$

所以：

$$
-
\sum_s
d_\pi(s)\phi(s)\phi^T(s)w_t
=
-\Phi^TD\Phi w_t.
$$

第一块对应：

$$
\gamma \Phi^TDP_\pi\Phi w_t.
$$

为什么？直接把矩阵和向量写出来看。先有：

$$
P_\pi=
\begin{bmatrix}
P_\pi(s_1,s_1) & \cdots & P_\pi(s_1,s_n)\\
\vdots & \ddots & \vdots\\
P_\pi(s_n,s_1) & \cdots & P_\pi(s_n,s_n)
\end{bmatrix},
\qquad
\Phi=
\begin{bmatrix}
\phi^T(s_1)\\
\vdots\\
\phi^T(s_n)
\end{bmatrix}.
$$

因此：

$$
P_\pi\Phi
=
\begin{bmatrix}
\sum_{s'}P_\pi(s_1,s')\phi^T(s')\\
\vdots\\
\sum_{s'}P_\pi(s_n,s')\phi^T(s')
\end{bmatrix}.
$$

也就是说，$P_\pi\Phi$ 的第 $s$ 行是：

$$
\sum_{s'}P_\pi(s,s')\phi^T(s').
$$

这正是“从 $s$ 出发，下一状态特征行向量的期望”。

再写出 $D$ 和 $\Phi^T$：

$$
D=
\begin{bmatrix}
d_\pi(s_1) & & \\
& \ddots & \\
& & d_\pi(s_n)
\end{bmatrix},
\qquad
\Phi^T=
\begin{bmatrix}
\phi(s_1) & \cdots & \phi(s_n)
\end{bmatrix}.
$$

于是多写一层等号：

$$
\Phi^TDP_\pi\Phi
=
\begin{bmatrix}
\phi(s_1) & \cdots & \phi(s_n)
\end{bmatrix}
\begin{bmatrix}
d_\pi(s_1)\sum_{s'}P_\pi(s_1,s')\phi^T(s')\\
\vdots\\
d_\pi(s_n)\sum_{s'}P_\pi(s_n,s')\phi^T(s')
\end{bmatrix}
=
\sum_s
d_\pi(s)\phi(s)
\left[
\sum_{s'}P_\pi(s,s')\phi^T(s')
\right].
$$

所以 TD 项等于：

$$
\gamma\Phi^TDP_\pi\Phi w_t
-
\Phi^TD\Phi w_t.
$$

把公共因子 $\Phi^TD$ 和 $\Phi w_t$ 提出来：

$$
\gamma\Phi^TDP_\pi\Phi w_t
-
\Phi^TD\Phi w_t
=
\Phi^TD(\gamma P_\pi-I)\Phi w_t.
$$

也可以写成：

$$
-
\Phi^TD(I-\gamma P_\pi)\Phi w_t.
$$

因此：

$$
\text{TD 项}
=
-
\Phi^TD(I-\gamma P_\pi)\Phi w_t
=
-Aw_t.
$$

## 6. 合并 reward 项和 TD 项

我们已经得到：

$$
\text{reward 项}
=
\Phi^TDr_\pi
=b,
$$

以及：

$$
\text{TD 项}
=
-
\Phi^TD(I-\gamma P_\pi)\Phi w_t
=
-Aw_t.
$$

所以：

$$
\mathbb {E} \left[
\left(
r _ {t + 1}
+ \gamma \phi^ {T} (s _ {t + 1}) w _ {t}
- \phi^ {T} (s _ {t}) w _ {t}
\right)
\phi (s _ {t})
\right]
=
b-Aw_t.
$$

这就是 Lemma 8.1。

## 7. 一个 2 状态、1 维特征的小数值例子

为了让矩阵形式更有手感，设有两个状态：

$$
P_\pi=
\begin{bmatrix}
0.8 & 0.2\\
0.4 & 0.6
\end{bmatrix},
\qquad
d_\pi=
\begin{bmatrix}
2/3\\
1/3
\end{bmatrix}.
$$

用 1 维特征：

$$
\phi(s_1)=1,
\qquad
\phi(s_2)=2.
$$

于是：

$$
\Phi=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
D=
\begin{bmatrix}
2/3 & 0\\
0 & 1/3
\end{bmatrix}.
$$

设：

$$
r_\pi=
\begin{bmatrix}
3\\
0
\end{bmatrix},
\qquad
\gamma=0.9.
$$

先算：

$$
b=\Phi^TDr_\pi
=
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
2/3 & 0\\
0 & 1/3
\end{bmatrix}
\begin{bmatrix}
3\\
0
\end{bmatrix}
=2.
$$

再算：

$$
A=\Phi^TD(I-\gamma P_\pi)\Phi.
$$

先有：

$$
I-\gamma P_\pi
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
-0.9
\begin{bmatrix}
0.8 & 0.2\\
0.4 & 0.6
\end{bmatrix}
=
\begin{bmatrix}
0.28 & -0.18\\
-0.36 & 0.46
\end{bmatrix}.
$$

然后：

$$
(I-\gamma P_\pi)\Phi
=
\begin{bmatrix}
0.28 & -0.18\\
-0.36 & 0.46
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
=
\begin{bmatrix}
-0.08\\
0.56
\end{bmatrix}.
$$

再乘 $D$：

$$
D(I-\gamma P_\pi)\Phi
=
\begin{bmatrix}
2/3 & 0\\
0 & 1/3
\end{bmatrix}
\begin{bmatrix}
-0.08\\
0.56
\end{bmatrix}
=
\begin{bmatrix}
-0.0533\\
0.1867
\end{bmatrix}.
$$

最后左乘 $\Phi^T$：

$$
A
=
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
-0.0533\\
0.1867
\end{bmatrix}
=
0.32.
$$

所以平均更新方向是：

$$
b-Aw_t=2-0.32w_t.
$$

如果当前参数 $w_t=1$，那么：

$$
b-Aw_t=1.68.
$$

这表示在当前参数下，平均 TD 更新方向是正的，会把 $w$ 往大的方向推。

## 8. 最容易卡住的三点

1. **为什么 $\phi(s_t)$ 可以变成 $\phi(s)$？**  
   因为条件期望里已经给定 $s_t=s$，所以当前状态的特征不再随机。

2. **为什么会出现 $P_\pi\Phi$？**  
   因为下一状态 $s_{t+1}$ 是随机的，条件在 $s_t=s$ 下，它的分布就是 $P_\pi$ 第 $s$ 行。$P_\pi\Phi$ 的第 $s$ 行正是下一状态特征的期望。

3. **为什么最后有负号 $-A w_t$？**  
   TD 项里是
   $$
   \gamma\Phi^TDP_\pi\Phi w_t-\Phi^TD\Phi w_t,
   $$
   它等于
   $$
   -\Phi^TD(I-\gamma P_\pi)\Phi w_t.
   $$
   而 $A=\Phi^TD(I-\gamma P_\pi)\Phi$，所以就是 $-Aw_t$。

## 9. 一句话总结

Lemma 8.1 做的事情是把“平均 TD 更新方向”矩阵化：

$$
\mathbb E[\delta_t\phi(s_t)]
=
\underbrace{\Phi^TDr_\pi}_{b}
-
\underbrace{\Phi^TD(I-\gamma P_\pi)\Phi}_{A}w_t.
$$

它不是凭空变出来的：$D$ 来自当前状态的平稳分布，$P_\pi$ 来自下一状态的条件分布，$\Phi$ 来自把每个状态的特征向量按行堆起来。
