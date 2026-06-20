# 第 9 章补充：Theorem 9.2 和 9.3 的梯度推导

> 对应主笔记：[第 9 章策略梯度方法](./09_chapter9-policy-gradient-methods) · 补充主题：把 Theorem 9.2 / 9.3 的代数推导单独展开

## 1. Theorem 9.2：固定初始分布 $d_0$ 时的严格推导

先把 $\bar v_\pi$ 的概念垫一下。策略 $\pi$ 不只在一个状态上有表现，而是在很多状态上都有 state value 状态价值：

$$
v_\pi(s_1),\ v_\pi(s_2),\ \cdots
$$

策略梯度需要一个 scalar objective 标量目标 $J(\theta)$，不能拿一整列状态价值直接做最大化。所以我们先选一个状态分布 $d$，说明“我更关心哪些状态”，再把这些状态价值加权平均：

$$
\bar v_\pi
=
\sum_{s\in\mathcal S}d(s)v_\pi(s)
=
d^Tv_\pi.
$$

Theorem 9.2 讨论的是一个特殊选择：把这个权重分布 $d$ 固定为 initial distribution 初始分布 $d_0$。于是平均状态价值写成：

$$
\bar v_\pi^0=d_0^Tv_\pi.
$$

这里上标 $0$ 不是平方或幂，而是在提醒我们：这个平均价值用的是“时刻 0 的起点分布” $d_0$。

展开看就是：

$$
\bar v_\pi^0
=
\sum_{s_0}d_0(s_0)v_\pi(s_0).
$$

因为 $d_0$ 不依赖策略参数 $\theta$，所以梯度只作用在 $v_\pi(s_0)$ 上：

$$
\nabla_\theta\bar v_\pi^0
=
\sum_{s_0}d_0(s_0)\nabla_\theta v_\pi(s_0).
$$

把 Lemma 9.2 代进来。为了避免符号撞车，把 Lemma 9.2 里的“起点状态”记成 $s_0$，把“未来访问到的状态”记成 $s$：

$$
\nabla_\theta v_\pi(s_0)
=
\sum_s
\Pr_\pi(s\mid s_0)
\sum_a
\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

于是：

$$
\nabla_\theta\bar v_\pi^0
=
\sum_{s_0}d_0(s_0)
\sum_s
\Pr_\pi(s\mid s_0)
\sum_a
\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

交换求和顺序：先固定未来会被访问到的状态 $s$，再把所有可能起点 $s_0$ 对它的贡献加起来：

$$
\nabla_\theta\bar v_\pi^0
=
\sum_s
\left[
\sum_{s_0}d_0(s_0)\Pr_\pi(s\mid s_0)
\right]
\sum_a
\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

括号里的东西定义为：

$$
\rho_\pi(s)
\doteq
\sum_{s_0\in\mathcal S}d_0(s_0)\Pr_\pi(s\mid s_0). \tag{9.19}
$$

所以：

$$
\nabla_\theta\bar v_\pi^0
=
\sum_s
\rho_\pi(s)
\sum_a
\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

读法：$d_0$ 只告诉我们“从哪里开始”；但真正计算梯度时，要看“从这些起点出发后，整个折扣未来会访问哪些状态”。这些未来访问权重合起来，就是 $\rho_\pi$。

再用 log-trick：

$$
\nabla_\theta\pi(a\mid s,\theta)
=
\pi(a\mid s,\theta)\nabla_\theta\ln\pi(a\mid s,\theta).
$$

代入：

$$
\nabla_\theta\bar v_\pi^0
=
\sum_s
\rho_\pi(s)
\sum_a
\pi(a\mid s,\theta)
\nabla_\theta\ln\pi(a\mid s,\theta)q_\pi(s,a).
$$

把两层求和看成“先抽状态，再抽动作”：

$$
\sum_s
\rho_\pi(s)
\sum_a
\pi(a\mid s,\theta)
f(s,a)
=
\mathbb E_{S\sim\rho_\pi,\ A\sim\pi(S,\theta)}
[f(S,A)].
$$

这里：

$$
f(s,a)
=
\nabla_\theta\ln\pi(a\mid s,\theta)q_\pi(s,a).
$$

所以：

$$
\nabla_\theta\bar v_\pi^0
=
\mathbb E_{S\sim\rho_\pi,\ A\sim\pi(S,\theta)}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)q_\pi(S,A)
\right].
$$

## 2. Theorem 9.3：平稳分布 $d_\pi$ 时的近似推导

这里的 $\bar v_\pi$ 是用 stationary distribution 平稳分布 $d_\pi$ 加权的平均状态价值：

$$
\bar v_\pi=d_\pi^Tv_\pi=\sum_s d_\pi(s)v_\pi(s).
$$

而 $\bar r_\pi$ 是 average reward 平均一步奖励：

$$
\bar r_\pi=d_\pi^Tr_\pi=\sum_s d_\pi(s)r_\pi(s).
$$

由 Bellman 方程和平稳分布关系 $d_\pi^TP_\pi=d_\pi^T$ 可得：

$$
\bar r_\pi=(1-\gamma)\bar v_\pi,
$$

所以：

$$
\nabla_\theta\bar r_\pi=(1-\gamma)\nabla_\theta\bar v_\pi.
$$

真正需要小心的是下一步。因为

$$
\bar v_\pi=d_\pi^Tv_\pi
$$

里不只是 $v_\pi$ 依赖 $\theta$，平稳分布 $d_\pi$ 也依赖 $\theta$。严格求导会有两项：

$$
\nabla_\theta\bar v_\pi
=
\sum_s\nabla_\theta d_\pi(s)v_\pi(s)
+
\sum_s d_\pi(s)\nabla_\theta v_\pi(s). \tag{9.20}
$$

当 $\gamma$ 接近 1 时，原书近似忽略第一项，保留第二项：

$$
\nabla_\theta\bar v_\pi
\approx
\sum_s d_\pi(s)\nabla_\theta v_\pi(s).
$$

现在把 Lemma 9.2 代进第二项：

$$
\nabla_\theta v_\pi(s)
=
\sum_{s'}\Pr_\pi(s'\mid s)
\sum_a\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
$$

代入：

$$
\sum_s d_\pi(s)\nabla_\theta v_\pi(s)
=
\sum_s d_\pi(s)
\sum_{s'}\Pr_\pi(s'\mid s)
\sum_a\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
$$

交换求和顺序：

$$
\sum_s d_\pi(s)\nabla_\theta v_\pi(s)
=
\sum_{s'}
\left[
\sum_s d_\pi(s)\Pr_\pi(s'\mid s)
\right]
\sum_a\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
$$

关键是括号：

$$
\sum_s d_\pi(s)\Pr_\pi(s'\mid s).
$$

由于

$$
\Pr_\pi(s'\mid s)
=
\sum_{k=0}^{\infty}\gamma^k[P_\pi^k]_{ss'},
$$

所以：

$$
\begin{aligned}
\sum_s d_\pi(s)\Pr_\pi(s'\mid s)
&=
\sum_s d_\pi(s)
\sum_{k=0}^{\infty}\gamma^k[P_\pi^k]_{ss'}\\
&=
\sum_{k=0}^{\infty}\gamma^k
\sum_s d_\pi(s)[P_\pi^k]_{ss'}.
\end{aligned}
$$

这里先做一个维度检查。设状态数为 $n=|\mathcal S|$，并采用本书的行向量转移约定：

$$
[P_\pi]_{ss'}
=
\Pr(S_{t+1}=s'\mid S_t=s,\pi).
$$

也就是说，$P_\pi$ 的“行”是当前状态 $s$，“列”是下一状态 $s'$。各对象维度是：

| 符号 | 维度 | 读法 |
|---|---:|---|
| $d_\pi$ | $n\times 1$ | 平稳分布列向量 |
| $d_\pi^T$ | $1\times n$ | 平稳分布行向量 |
| $P_\pi^k$ | $n\times n$ | 按策略走 $k$ 步的转移矩阵 |
| $d_\pi^TP_\pi^k$ | $1\times n$ | 从分布 $d_\pi$ 出发，走 $k$ 步后的状态分布 |

所以矩阵乘法的第 $s'$ 个分量就是：

$$
\big[d_\pi^TP_\pi^k\big]_{s'}
=
\sum_{s\in\mathcal S}
\underbrace{d_\pi(s)}_{\text{行向量第 }s\text{ 个分量}}
\underbrace{[P_\pi^k]_{ss'}}_{\text{矩阵第 }s\text{ 行、第 }s'\text{ 列}}.
$$

这正是上面出现的：

$$
\sum_s d_\pi(s)[P_\pi^k]_{ss'}.
$$

因为 $d_\pi$ 是平稳分布：

$$
d_\pi^TP_\pi=d_\pi^T,
$$

所以走 $k$ 步仍然有：

$$
d_\pi^TP_\pi^k=d_\pi^T.
$$

因此：

$$
\sum_s d_\pi(s)[P_\pi^k]_{ss'}=d_\pi(s').
$$

代回去：

$$
\sum_s d_\pi(s)\Pr_\pi(s'\mid s)
=
\sum_{k=0}^{\infty}\gamma^k d_\pi(s')
=
\frac{1}{1-\gamma}d_\pi(s').
$$

于是：

$$
\sum_s d_\pi(s)\nabla_\theta v_\pi(s)
=
\frac{1}{1-\gamma}
\sum_{s'} d_\pi(s')
\sum_a\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
$$

把哑变量 $s'$ 改名成 $s$：

$$
\sum_s d_\pi(s)\nabla_\theta v_\pi(s)
=
\frac{1}{1-\gamma}
\sum_s d_\pi(s)
\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

所以得到的是：

$$
\nabla_\theta\bar v_\pi
\approx
\frac{1}{1-\gamma}
\sum_s d_\pi(s)
\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

现在回到：

$$
\nabla_\theta\bar r_\pi
=
(1-\gamma)\nabla_\theta\bar v_\pi.
$$

把上一行的近似代入：

$$
\nabla_\theta\bar r_\pi
\approx
(1-\gamma)
\left[
\frac{1}{1-\gamma}
\sum_s d_\pi(s)
\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a)
\right].
$$

于是：

$$
\nabla_\theta\bar r_\pi
\approx
\sum_s d_\pi(s)
\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

也就是说，$(1-\gamma)$ 没丢，而是：

$$
(1-\gamma)\times\frac{1}{1-\gamma}=1.
$$

再用 log-trick：

$$
\nabla_\theta\bar r_\pi
\approx
\mathbb E_{S\sim d_\pi,\ A\sim\pi(S,\theta)}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)q_\pi(S,A)
\right].
$$

## 3. 速记

| 关系 | 是否精确 | 原因 |
|---|---|---|
| $\bar r_\pi=(1-\gamma)\bar v_\pi$ | 精确 | 来自 Bellman 方程和平稳分布等式 $d_\pi^TP_\pi=d_\pi^T$ |
| $\nabla_\theta\bar r_\pi=(1-\gamma)\nabla_\theta\bar v_\pi$ | 精确 | 对上一个精确关系求导 |
| $\nabla_\theta\bar r_\pi\approx\sum_s d_\pi(s)\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a)$ | 近似 | 忽略了 $\nabla_\theta d_\pi$ 项，$\gamma\to1$ 时更合理 |

⚠️ 易错点：Theorem 9.2 是固定初始分布 $d_0$ 下的严格形式；Theorem 9.3 是平稳分布 $d_\pi$ 下、折扣情形中的近似形式，$\gamma$ 越接近 1 越准。
