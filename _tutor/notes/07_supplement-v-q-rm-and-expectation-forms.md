# 第 7 章补充：$v_\pi$、$q_\pi$、RM 与期望/求和表达式

> 学习日期 2026-06-12 · 对应 7.1-7.2 · 重点：value definitions、Bellman expectation、Robbins-Monro

## 要解决的问题

7.1 和 7.2 里几个东西会反复切换：

- $v_\pi(s)$：给定策略 $\pi$ 时，一个状态有多好。
- $q_\pi(s,a)$：给定策略 $\pi$ 时，在状态 $s$ 先做动作 $a$ 有多好。
- Bellman equation 贝尔曼方程：把“长期回报”改写成“一步奖励 + 下一个位置的价值”。
- Robbins-Monro algorithm（RM）罗宾斯-门罗算法：把“求 Bellman 方程的解”变成“用带噪声样本一步步逼近解”。

这篇补充专门整理一条主线：

$$
\text{回报定义}
\Longrightarrow
\text{条件期望定义价值}
\Longrightarrow
\text{Bellman 期望式}
\Longrightarrow
\text{按概率求和展开}
\Longrightarrow
\text{用样本近似期望}
\Longrightarrow
\text{RM/TD/Sarsa 更新}.
$$

---

## 1. 先从 return 回报开始

从时刻 $t$ 开始，未来所有折扣奖励的总和叫 return 回报：

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2 R_{t+3}
+
\cdots.
$$

它也可以递归写成：

$$
G_t
=
R_{t+1}
+
\gamma G_{t+1}.
$$

读法：从现在开始的总回报 = 下一步奖励 + 折扣后的“从下一时刻开始的总回报”。

⚠️ $G_t$ 是随机变量。因为未来会走到哪里、拿到什么奖励，都可能有随机性。

---

## 2. $v_\pi(s)$：状态值是什么

state value 状态值定义为：

$$
v_\pi(s)
\doteq
\mathbb E_\pi[G_t\mid S_t=s].
$$

读法：如果当前在状态 $s$，之后一直按照策略 $\pi$ 行动，那么未来回报 $G_t$ 的期望是多少。

这里的下标 $\pi$ 表示两件事：

1. 动作怎么选，由策略 $\pi(a|s)$ 决定。
2. 期望是“在执行策略 $\pi$ 的前提下”取的。

把 $G_t=R_{t+1}+\gamma G_{t+1}$ 代入定义：

$$
\begin{aligned}
v_\pi(s)
&=
\mathbb E_\pi[G_t\mid S_t=s]\\
&=
\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s]\\
&=
\mathbb E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s].
\end{aligned}
$$

最后一步为什么能把 $G_{t+1}$ 换成 $v_\pi(S_{t+1})$？

因为到达下一个状态 $S_{t+1}$ 后，未来回报 $G_{t+1}$ 的条件期望就是：

$$
\mathbb E_\pi[G_{t+1}\mid S_{t+1}]
=
v_\pi(S_{t+1}).
$$

于是得到 state-value Bellman expectation equation 状态值贝尔曼期望方程：

$$
v_\pi(s)
=
\mathbb E_\pi[
R_{t+1}
+
\gamma v_\pi(S_{t+1})
\mid S_t=s
].
$$

---

## 3. $v_\pi(s)$ 的求和展开

如果把期望完全展开，就要枚举动作、奖励、下个状态。

给定当前状态 $s$，策略先选动作：

$$
a\sim\pi(\cdot|s).
$$

环境再给出奖励和下个状态：

$$
(r,s')\sim p(r,s'|s,a).
$$

因此：

$$
v_\pi(s)
=
\sum_a
\pi(a|s)
\sum_{r}
\sum_{s'}
p(r,s'|s,a)
\left[
r+\gamma v_\pi(s')
\right].
$$

读法：对所有可能动作、奖励、下个状态，把“一步奖励 + 下个状态价值”按发生概率加权平均。

如果把奖励期望先单独写出来，也常见到：

$$
v_\pi(s)
=
\sum_a\pi(a|s)
\left[
\sum_r r p(r|s,a)
+
\gamma\sum_{s'}p(s'|s,a)v_\pi(s')
\right].
$$

这两个式子本质一样，只是一个把 $r$ 和 $s'$ 放在联合概率 $p(r,s'|s,a)$ 里，一个把奖励期望和状态转移拆开写。

⚠️ 看到求和式时，不要被符号吓到。它只是条件期望的离散展开：

$$
\mathbb E[X]
=
\sum_x x\cdot p(x).
$$

---

## 4. $q_\pi(s,a)$：动作值是什么

action value 动作值定义为：

$$
q_\pi(s,a)
\doteq
\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

读法：如果当前在状态 $s$，并且第一步强制先做动作 $a$，之后再按照策略 $\pi$ 行动，那么未来回报的期望是多少。

它和 $v_\pi(s)$ 的区别是：

| 符号 | 条件里固定了什么 | 白话 |
|---|---|---|
| $v_\pi(s)$ | 只固定 $S_t=s$ | 在状态 $s$ 按策略 $\pi$ 行动，平均有多好 |
| $q_\pi(s,a)$ | 固定 $S_t=s,A_t=a$ | 在状态 $s$ 先做动作 $a$，之后按 $\pi$，平均有多好 |

把回报递归代入：

$$
\begin{aligned}
q_\pi(s,a)
&=
\mathbb E_\pi[G_t\mid S_t=s,A_t=a]\\
&=
\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s,A_t=a].
\end{aligned}
$$

到达 $S_{t+1}$ 后，后续会按策略 $\pi$ 选动作 $A_{t+1}$，所以：

$$
\mathbb E_\pi[G_{t+1}\mid S_{t+1}]
=
v_\pi(S_{t+1})
=
\sum_{a'}\pi(a'|S_{t+1})q_\pi(S_{t+1},a').
$$

因此动作值 Bellman 方程可以写成两种等价形式。

第一种，用 $v_\pi$ 写：

$$
q_\pi(s,a)
=
\mathbb E_\pi[
R_{t+1}
+
\gamma v_\pi(S_{t+1})
\mid
S_t=s,A_t=a
].
$$

第二种，显式保留下一个动作 $A_{t+1}$：

$$
q_\pi(s,a)
=
\mathbb E_\pi[
R_{t+1}
+
\gamma q_\pi(S_{t+1},A_{t+1})
\mid
S_t=s,A_t=a
].
$$

这就是 Sarsa 背后的方程。

---

## 5. $q_\pi(s,a)$ 的求和展开

先用 $v_\pi$ 版本展开：

$$
q_\pi(s,a)
=
\sum_r\sum_{s'}
p(r,s'|s,a)
\left[
r+\gamma v_\pi(s')
\right].
$$

再把 $v_\pi(s')$ 展开成动作值：

$$
v_\pi(s')
=
\sum_{a'}\pi(a'|s')q_\pi(s',a').
$$

代回去：

$$
q_\pi(s,a)
=
\sum_r\sum_{s'}
p(r,s'|s,a)
\left[
r
+
\gamma
\sum_{a'}\pi(a'|s')q_\pi(s',a')
\right].
$$

也可以把奖励项和未来价值项拆开：

$$
q_\pi(s,a)
=
\sum_r r p(r|s,a)
+
\gamma
\sum_{s'}p(s'|s,a)
\sum_{a'}\pi(a'|s')q_\pi(s',a').
$$

这就是书里 7.2 看到的形式。

如果定义：

$$
p(s',a'|s,a)
=
p(s'|s,a)\pi(a'|s'),
$$

那么未来价值项还可以写得更紧：

$$
\gamma
\sum_{s'}\sum_{a'}
q_\pi(s',a')p(s',a'|s,a).
$$

⚠️ 这里的 $a$ 和 $a'$ 不一样：

- $a$：当前已经固定的动作，条件里给了 $A_t=a$。
- $a'$：下个状态 $s'$ 下，策略 $\pi$ 可能选的下一个动作，需要求和平均。

---

## 6. $v_\pi$ 和 $q_\pi$ 的互相转换

从 $q_\pi$ 到 $v_\pi$：

$$
v_\pi(s)
=
\sum_a\pi(a|s)q_\pi(s,a).
$$

读法：状态 $s$ 的价值，就是在这个状态下按策略 $\pi$ 可能选择的各个动作价值的加权平均。

从 $v_\pi$ 到 $q_\pi$：

$$
q_\pi(s,a)
=
\mathbb E[
R_{t+1}
+
\gamma v_\pi(S_{t+1})
\mid S_t=s,A_t=a
].
$$

求和展开是：

$$
q_\pi(s,a)
=
\sum_r\sum_{s'}
p(r,s'|s,a)
\left[
r+\gamma v_\pi(s')
\right].
$$

所以：

$$
\boxed{
v_\pi(s)=\sum_a\pi(a|s)q_\pi(s,a)
}
$$

$$
\boxed{
q_\pi(s,a)=\mathbb E[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s,A_t=a]
}
$$

这两个框可以当作 7.2 的随身字典。

---

## 7. 普通 Sarsa 与 Expected Sarsa 到底差在哪

普通 Sarsa 的 target 是：

$$
R_{t+1}+\gamma q_t(S_{t+1},A_{t+1}).
$$

它真的采样一个 $A_{t+1}$。

Expected Sarsa 的 target 是：

$$
R_{t+1}
+
\gamma
\mathbb E[
q_t(S_{t+1},A)
\mid S_{t+1}
].
$$

它不采样下一个动作，而是在给定 $S_{t+1}$ 后，对所有动作按策略求平均：

$$
\mathbb E[
q_t(S_{t+1},A)
\mid S_{t+1}
]
=
\sum_a\pi_t(a|S_{t+1})q_t(S_{t+1},a).
$$

所以二者区别是：

| 算法             | 对 $A_{t+1}$ 怎么处理 | target                                              |     |
| -------------- | ---------------- | --------------------------------------------------- | --- |
| Sarsa          | 抽一个动作样本          | $R_{t+1}+\gamma q_t(S_{t+1},A_{t+1})$               |     |
| Expected Sarsa | 对动作分布求平均         | $R_{t+1}+\gamma\sum_a\pi_t(aS_{t+1})q_t(S_{t+1},a)$ |     |

它们学习的目标仍然是同一个 $q_\pi$。区别只是：

$$
\text{Sarsa：用一次样本估计期望}
$$

$$
\text{Expected Sarsa：提前把动作这层随机性精确平均掉}
$$

因此 Expected Sarsa 通常方差更低，但需要枚举动作。

---

## 8. Robbins-Monro algorithm：为什么 TD/Sarsa 能这样更新

RM 研究的是求根：

$$
g(w^*)=0.
$$

这里的 $g(w)$ 是一个很一般的函数。它的基础角色只有一个：

> 把“我们想找的答案 $w^*$”改写成“某个函数等于 0 的根”。

也就是说，只要某个问题能写成：

$$
\text{某个条件在 }w^*\text{ 处成立},
$$

我们就可以把这个条件包装成：

$$
g(w^*)=0.
$$

### $g(w)$ 的几种常见基础形式

**第一种：固定点方程的差值形式。**

如果目标满足：

$$
w^*=h(w^*),
$$

就可以定义：

$$
g(w)\doteq w-h(w).
$$

那么：

$$
g(w^*)=0
\Longleftrightarrow
w^*-h(w^*)=0
\Longleftrightarrow
w^*=h(w^*).
$$

TD 和 Sarsa 就属于这一类。因为 Bellman 方程本质上是：

$$
\text{价值}=\text{Bellman target 的期望}.
$$

所以它们很自然会变成：

$$
g(w)
=
\text{当前估计}
-
\text{目标的期望}.
$$

再用样本替代期望后，就得到：

$$
\tilde g(w)
=
\text{当前估计}
-
\text{样本 target}.
$$

这就是为什么 TD 里会出现：

$$
\tilde g(v_t(s_t))
=
v_t(s_t)
-
\left[
r_{t+1}+\gamma v_t(s_{t+1})
\right].
$$

**第二种：均值估计的差值形式。**

如果目标是估计一个期望：

$$
w^*=\mathbb E[X],
$$

也可以写成固定点差值：

$$
g(w)\doteq w-\mathbb E[X].
$$

因为：

$$
g(w^*)=0
\Longleftrightarrow
w^*=\mathbb E[X].
$$

但 $\mathbb E[X]$ 不知道，所以用样本 $x_k$ 构造：

$$
\tilde g(w_k)=w_k-x_k.
$$

这就是第 6 章从样本平均走到 Robbins-Monro 的基本例子。

**第三种：优化问题的梯度形式。**

如果目标是最小化一个函数：

$$
\min_w J(w),
$$

常见的一阶条件是：

$$
\nabla J(w^*)=0.
$$

这时可以定义：

$$
g(w)\doteq \nabla J(w).
$$

于是：

$$
g(w^*)=0
\Longleftrightarrow
\nabla J(w^*)=0.
$$

这类 $g(w)$ 不一定长得像“当前估计 - target”。它是梯度。SGD 随机梯度下降就是用样本梯度 $\tilde g(w_k)$ 去近似真实梯度 $g(w_k)$。

所以要记住：

> $g(w)$ 不一定都能或都必须写成差值。  
> 只有当原问题天然是“某个量 = 某个目标/期望目标”时，把 $g(w)$ 写成“当前量 - 目标量”才特别自然。TD 和 Sarsa 正好属于这一类。

如果能直接算 $g(w)$，普通方法可以写成：

$$
w_{k+1}=w_k-\alpha_k g(w_k).
$$

但强化学习里经常算不了真实期望，只能看到样本。于是 RM 允许我们用一个带噪声的估计：

$$
\tilde g(w_k)
=
g(w_k)+\text{noise}.
$$

更新变成：

$$
w_{k+1}
=
w_k-\alpha_k\tilde g(w_k).
$$

关键条件是：

$$
\mathbb E[\tilde g(w_k)\mid \mathcal H_k]
=
g(w_k).
$$

读法：单次样本可能有噪声，但平均来看方向是对的。

---

## 9. TD 如何对应 RM

状态值 Bellman 方程是：

$$
v_\pi(s)
=
\mathbb E[
R_{t+1}+\gamma v_\pi(S_{t+1})
\mid S_t=s
].
$$

把它改写成求根问题：

$$
g(v(s))
\doteq
v(s)
-
\mathbb E[
R_{t+1}+\gamma v(S_{t+1})
\mid S_t=s
].
$$

真正的 $v_\pi$ 满足：

$$
g(v_\pi(s))=0.
$$

但是我们不知道这个期望。采到一次样本：

$$
s_t=s,\quad r_{t+1},\quad s_{t+1},
$$

就用样本构造：

$$
\tilde g(v_t(s_t))
=
v_t(s_t)
-
\left[
r_{t+1}+\gamma v_t(s_{t+1})
\right].
$$

代入 RM：

$$
v_{t+1}(s_t)
=
v_t(s_t)
-
\alpha_t(s_t)
\tilde g(v_t(s_t)).
$$

得到 TD 更新：

$$
v_{t+1}(s_t)
=
v_t(s_t)
+
\alpha_t(s_t)
\left[
r_{t+1}
+
\gamma v_t(s_{t+1})
-
v_t(s_t)
\right].
$$

所以 TD 的本质是：

$$
\text{用单步样本近似 Bellman 期望，然后用 RM 做带噪声求根。}
$$

---

## 10. Sarsa 如何对应 RM

动作值 Bellman 方程是：

$$
q_\pi(s,a)
=
\mathbb E[
R_{t+1}
+
\gamma q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
].
$$

改写成求根：

$$
g(q(s,a))
\doteq
q(s,a)
-
\mathbb E[
R_{t+1}
+
\gamma q(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
].
$$

采到一次 Sarsa 样本：

$$
(s_t,a_t,r_{t+1},s_{t+1},a_{t+1}),
$$

用样本构造带噪声观测：

$$
\tilde g(q_t(s_t,a_t))
=
q_t(s_t,a_t)
-
\left[
r_{t+1}
+
\gamma q_t(s_{t+1},a_{t+1})
\right].
$$

代入 RM：

$$
q_{t+1}(s_t,a_t)
=
q_t(s_t,a_t)
-
\alpha_t(s_t,a_t)
\tilde g(q_t(s_t,a_t)).
$$

得到 Sarsa：

$$
q_{t+1}(s_t,a_t)
=
q_t(s_t,a_t)
+
\alpha_t(s_t,a_t)
\left[
r_{t+1}
+
\gamma q_t(s_{t+1},a_{t+1})
-
q_t(s_t,a_t)
\right].
$$

这和 TD 状态值更新完全同形，只是对象从 $s$ 变成了 $(s,a)$。

---

## 11. 一个小例子：从期望到求和

假设在状态 $s$ 下，策略有两个动作：

$$
\pi(a_1|s)=0.7,\qquad \pi(a_2|s)=0.3.
$$

动作值为：

$$
q_\pi(s,a_1)=10,\qquad q_\pi(s,a_2)=0.
$$

那么状态值是：

$$
v_\pi(s)
=
\mathbb E_{A\sim\pi(\cdot|s)}[q_\pi(s,A)].
$$

把期望写成求和：

$$
v_\pi(s)
=
\sum_a\pi(a|s)q_\pi(s,a).
$$

代入数字：

$$
v_\pi(s)
=
0.7\times10+0.3\times0
=
7.
$$

所以：

$$
\text{期望表达}
\quad
\mathbb E[q_\pi(s,A)]
$$

和

$$
\text{求和表达}
\quad
\sum_a\pi(a|s)q_\pi(s,a)
$$

说的是同一件事，只是一个抽象，一个展开。

---

## 12. 速记表

| 概念 | 公式 | 白话 |
|---|---|---|
| return | $G_t=R_{t+1}+\gamma G_{t+1}$ | 从现在开始的折扣总回报 |
| state value | $v_\pi(s)=\mathbb E_\pi[G_t|S_t=s]$ | 在状态 $s$ 按 $\pi$ 走的平均回报 |
| action value | $q_\pi(s,a)=\mathbb E_\pi[G_t|S_t=s,A_t=a]$ | 在 $s$ 先做 $a$，之后按 $\pi$ 走的平均回报 |
| $v$ from $q$ | $v_\pi(s)=\sum_a\pi(a|s)q_\pi(s,a)$ | 状态值是动作值的策略加权平均 |
| $q$ from $v$ | $q_\pi(s,a)=\mathbb E[R+\gamma v_\pi(S')|s,a]$ | 动作值是一步奖励加下个状态值 |
| TD target | $r_{t+1}+\gamma v_t(s_{t+1})$ | 状态值的一步样本目标 |
| Sarsa target | $r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})$ | 动作值的一步样本目标 |
| Expected Sarsa target | $r_{t+1}+\gamma\sum_a\pi_t(a|s_{t+1})q_t(s_{t+1},a)$ | 先把下一动作平均掉 |
| RM | $w_{k+1}=w_k-\alpha_k\tilde g(w_k)$ | 用带噪声样本求方程的根 |

## 13. 最容易混的三件事

**第一，$v_\pi$ 和 $q_\pi$ 都是给定策略 $\pi$ 的价值。**

它们不是最优价值。最优价值通常写作 $v_*$、$q_*$。

**第二，期望式和求和式不是两套理论。**

例如：

$$
\mathbb E[q_\pi(s,A)]
$$

展开后就是：

$$
\sum_a\pi(a|s)q_\pi(s,a).
$$

**第三，TD/Sarsa 的更新不是凭空发明的。**

它们都来自同一个套路：

$$
\text{Bellman 方程}
\Longrightarrow
\text{写成 }g(w)=0
\Longrightarrow
\text{用样本构造 }\tilde g(w)
\Longrightarrow
\text{RM 更新}.
$$
