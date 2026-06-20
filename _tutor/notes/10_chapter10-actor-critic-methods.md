# 第 10 章 Actor-Critic 方法（Actor-Critic Methods）- 10.1

> 原书 p224 · 学习日期 2026-06-20 · 当前涵盖 10.1

## 本章在全书的位置（先读这段）

第 8 章解决的是 value function approximation 价值函数近似：用参数 $w$ 表示 $\hat v(s,w)$ 或 $\hat q(s,a,w)$，让价值估计能推广到大状态空间。

第 9 章解决的是 policy gradient 策略梯度：用参数 $\theta$ 表示策略 $\pi(a\mid s,\theta)$，直接沿着

$$
\nabla_\theta J(\theta)
\approx
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)q_\pi(S,A)
\right]
$$

来更新策略。

第 10 章把这两条线合起来：

> actor 演员负责更新策略 $\pi(a\mid s,\theta)$；critic 评论家负责估计价值 $q_\pi(s,a)$ 或优势函数。actor 问“我该怎么改策略”，critic 回答“刚才这个动作到底值多少”。

这也是为什么 actor-critic 是自然出现的：第 9 章的策略梯度公式里本来就需要 $q_\pi(S,A)$，而第 8 章已经教过我们怎样用 TD 和函数近似估计动作价值。

---

## 10.1 The simplest actor-critic algorithm (QAC)（最简单的 Actor-Critic：QAC）

**要解决的问题**：REINFORCE 用一整段 Monte Carlo 回报估计 $q_\pi(s_t,a_t)$，方差很大。本节要说明：如果改用 TD learning 学出来的动作价值函数 $q(s,a,w)$ 来替代这个回报估计，就自然得到最简单的 actor-critic 算法。

### 1. 从第 9 章的策略梯度更新开始

第 9 章给出的策略梯度更新是：

$$
\begin{aligned}
\theta_{t+1}
&=
\theta_t+\alpha\nabla_\theta J(\theta_t)\\
&=
\theta_t+\alpha
\mathbb E_{S\sim\eta,\ A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)q_\pi(S,A)
\right]. \tag{10.1}
\end{aligned}
$$

其中 $\eta$ 是状态分布，可以根据具体目标理解成 $\rho_\pi$ 或 $d_\pi$。

真实梯度通常不知道，所以用样本梯度近似：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
q_t(s_t,a_t). \tag{10.2}
$$

这其实就是第 9 章的 (9.32)。所以 10.1 不是从零开始，而是在问：

> 这个 $q_t(s_t,a_t)$ 到底怎么来？

### 2. REINFORCE 和 actor-critic 的分叉点

第 9 章 REINFORCE 的做法是用 Monte Carlo return 蒙特卡洛回报：

$$
q_t(s_t,a_t)
=
\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k.
$$

这意味着要等 episode 结束，才能知道每一步后面的完整回报。

actor-critic 的想法是：不要等整局结束，改用一个由 TD learning 学出来的动作价值近似：

$$
q_t(s_t,a_t)
\approx
q(s_t,a_t,w_t).
$$

于是 (10.2) 变成：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
q(s_t,a_t,w_t).
$$

这就是 actor 更新。它仍然是策略梯度，只是把 $q_\pi$ 的估计交给了另一个模块。

两条路线可以并排看：

| 方法 | $q_t(s_t,a_t)$ 怎么估计 | 特点 |
|---|---|---|
| REINFORCE / Monte Carlo policy gradient | 用 episode 结束后的完整折扣回报 | 无偏但方差大，通常要等回合结束 |
| actor-critic | 用 TD 学出来的 $q(s_t,a_t,w_t)$ | 可在线更新，方差更低，但引入价值估计偏差 |

### 3. actor 和 critic 分别做什么？

actor-critic 这个名字正好对应两个参数化对象：

| 角色 | 参数 | 函数 | 负责什么 |
|---|---|---|---|
| actor 演员 | $\theta$ | $\pi(a\mid s,\theta)$ | 直接决定动作概率，更新策略 |
| critic 评论家 | $w$ | $q(s,a,w)$ | 估计动作价值，评价 actor 的动作好不好 |

actor 的更新式是：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
q(s_t,a_t,w_t).
$$

读法：critic 如果认为当前样本 $(s_t,a_t)$ 的动作价值高，actor 就更倾向于增加这类动作的概率；如果动作价值低，actor 的更新方向会相应变弱或反向。

critic 的任务是更新 $w$，让 $q(s,a,w)$ 更接近当前策略下的动作价值。

### 4. QAC：用 Sarsa 做 critic

最简单的 actor-critic 叫 Q actor-critic，简称 QAC。这里的 Q 表示 critic 估计的是 action value 动作价值：

$$
q(s,a,w).
$$

原书说 critic 使用的是第 8 章的 Sarsa with function approximation。第 8 章的 Sarsa 更新是：

$$
w_{t+1}
=
w_t
+
\alpha_w
\left[
r_{t+1}
+
\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t)
\right]
\nabla_w q(s_t,a_t,w_t).
$$

中括号里的量是 TD error 时间差分误差：

$$
\delta_t
=
r_{t+1}
+
\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t).
$$

所以 critic 更新也可以写成：

$$
w_{t+1}
=
w_t+\alpha_w\delta_t\nabla_w q(s_t,a_t,w_t).
$$

读法：如果当前估计 $q(s_t,a_t,w_t)$ 比 TD target

$$
r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)
$$

偏低，就把它往上推；如果偏高，就往下拉。

⚠️ 易错点：这里的 Sarsa 是 on-policy 的，因为 $a_{t+1}$ 是按当前 actor 策略 $\pi(a\mid s_{t+1},\theta_t)$ 实际采样出来的，不是取 $\arg\max_a q(s_{t+1},a,w_t)$。

### 5. Algorithm 10.1：最简单的 QAC 流程

**输入**：

- policy 策略函数 $\pi(a\mid s,\theta_0)$，初始参数 $\theta_0$
- action value 动作价值函数 $q(s,a,w_0)$，初始参数 $w_0$
- 学习率 $\alpha_\theta,\alpha_w>0$

每个时间步 $t$：

1. 按 actor 采样动作：

$$
a_t\sim\pi(\cdot\mid s_t,\theta_t).
$$

2. 执行动作，观察 $r_{t+1},s_{t+1}$。

3. 再按当前 actor 在下一状态采样：

$$
a_{t+1}\sim\pi(\cdot\mid s_{t+1},\theta_t).
$$

4. actor 更新策略参数：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
q(s_t,a_t,w_t).
$$

5. critic 用 Sarsa 更新价值参数：

$$
w_{t+1}
=
w_t
+
\alpha_w
\left[
r_{t+1}
+
\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t)
\right]
\nabla_w q(s_t,a_t,w_t).
$$

### 6. 一个最小数值例子：actor 为什么需要 critic

假设在某个状态 $s$，actor 采到了动作 $a$。策略梯度更新需要：

$$
\nabla_\theta\ln\pi(a\mid s,\theta_t)q_\pi(s,a).
$$

但是 $q_\pi(s,a)$ 不知道。QAC 用 critic 的估计代替它：

$$
q_\pi(s,a)\approx q(s,a,w_t).
$$

假设当前 critic 给出：

$$
q(s,a,w_t)=5.
$$

那么 actor 更新方向大致是：

$$
\theta_{t+1}
=
\theta_t
+
5\alpha_\theta\nabla_\theta\ln\pi(a\mid s,\theta_t).
$$

如果后来环境反馈说明 critic 太乐观了，比如：

$$
r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)=3,
$$

那么 TD error 是：

$$
\delta_t=3-5=-2.
$$

critic 会把 $q(s,a,w)$ 往下修。下一次 actor 再遇到类似样本时，就不会被一个过高的价值估计推得那么猛。

这就是 actor 和 critic 的配合关系：

$$
\text{critic 评价动作}
\Longrightarrow
\text{actor 按评价改策略}
\Longrightarrow
\text{新策略产生新样本}
\Longrightarrow
\text{critic 继续修正评价}.
$$

### 7. 和前两章的闭环

QAC 可以看成两条旧线的拼接：

| 来源 | 旧知识 | 在 QAC 里的角色 |
|---|---|---|
| 第 9 章 | policy gradient / REINFORCE 更新 $\theta$ | actor |
| 第 8 章 | Sarsa with function approximation 更新 $w$ | critic |

所以 QAC 的核心不是一个全新的公式，而是这个结构：

$$
\boxed{
\text{policy gradient 更新策略}
+
\text{TD learning 估计动作价值}
}
$$

这也解释了为什么第 10 章会成为第 8 章和第 9 章的交汇点。

⚠️ 易错点：actor-critic 不是“有 actor 就不需要价值”。恰好相反，它显式让 critic 学价值，只是这个价值不再用于贪心选动作，而是用于给 actor 的策略梯度提供评价信号。

---

## 我的疑问与解答

暂无。

---

## 脉络总结 / 要点速记

10.1：最简单的 actor-critic（QAC）来自对 REINFORCE 的替换：把 Monte Carlo 回报 $q_t(s_t,a_t)$ 换成 TD 学出来的 $q(s_t,a_t,w_t)$。actor 用策略梯度更新 $\theta$，critic 用 Sarsa 更新 $w$。
