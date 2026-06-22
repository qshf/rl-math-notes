# 第 10 章 Actor-Critic 方法（Actor-Critic Methods）- 10.1-10.6

> 原书 p224-245 · 学习日期 2026-06-20 · 涵盖 10.1-10.6（全章完）

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

⚠️ 记号说明：你可能会觉得这里少了一个帽子。第 8 章讲 function approximation 时，为了强调“这是对真实 $q_\pi(s,a)$ 的近似”，常写成

$$
\hat q(s,a,w).
$$

第 10 章原书在 actor-critic 里把 critic 的近似动作价值函数简写成

$$
q(s,a,w).
$$

所以这里的 $q(s,a,w)$ 不是 tabular true value 真值表，也不是已经等于真实的 $q_\pi(s,a)$；它就是 critic 用参数 $w$ 表示的近似函数。也可以按第 8 章记法理解成：

$$
q(s,a,w)\equiv \hat q(s,a,w)\approx q_\pi(s,a).
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

### 6. 一个最小数值例子：按 Algorithm 10.1 跑一轮

为了让上面的 5 个步骤真正落地，考虑一个极简例子。当前在状态 $s_1$，有两个动作 $a_1,a_2$。actor 当前策略是：

| 状态 | $\pi(a_1\mid s,\theta_t)$ | $\pi(a_2\mid s,\theta_t)$ |
|---|---:|---:|
| $s_1$ | 0.70 | 0.30 |

critic 当前对动作价值的估计是：

| 动作价值估计 | 数值 |
|---|---:|
| $q(s_1,a_1,w_t)$ | 2.0 |
| $q(s_1,a_2,w_t)$ | 0.5 |

再假设：

$$
\alpha_\theta=0.1,\qquad
\alpha_w=0.2,\qquad
\gamma=0.9.
$$

为了只看清流程，假设 actor 这一刻的 log-policy gradient 是一个一维数：

$$
\nabla_\theta\ln\pi(a_1\mid s_1,\theta_t)=0.4,
$$

critic 的函数梯度也先取成一维：

$$
\nabla_w q(s_1,a_1,w_t)=0.5.
$$

#### 第 1 步：actor 按当前策略采样动作

当前在 $s_1$，按策略抽动作：

$$
a_t\sim\pi(\cdot\mid s_1,\theta_t).
$$

假设这次抽到了 $a_t=a_1$。注意：这里不是选最大 $q$ 的动作，而是按 actor 的概率分布采样。

#### 第 2 步：执行动作，观察环境反馈

执行 $a_1$ 后，环境给出：

$$
r_{t+1}=1,\qquad s_{t+1}=s_2.
$$

#### 第 3 步：在下一状态继续按 actor 采样动作

假设在 $s_2$，actor 抽到：

$$
a_{t+1}=a_2.
$$

critic 对下一状态-动作的估计是：

$$
q(s_2,a_2,w_t)=3.0.
$$

#### 第 4 步：actor 用 critic 的当前评价更新策略

actor 更新用的是当前 critic 对 $(s_1,a_1)$ 的评价：

$$
q(s_1,a_1,w_t)=2.0.
$$

所以：

$$
\begin{aligned}
\theta_{t+1}
&=
\theta_t
+
\alpha_\theta
\nabla_\theta\ln\pi(a_1\mid s_1,\theta_t)
q(s_1,a_1,w_t)\\
&=
\theta_t
+
0.1\times 0.4\times 2.0\\
&=
\theta_t+0.08.
\end{aligned}
$$

读法：critic 认为 $(s_1,a_1)$ 的动作价值是正的，而且不小，所以 actor 沿着“提高这次动作概率”的方向走了一步。

#### 第 5 步：critic 用 Sarsa 的 TD error 更新价值估计

critic 的 TD target 是：

$$
r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)
=
1+0.9\times 3.0
=
3.7.
$$

当前 critic 对 $(s_1,a_1)$ 的估计是：

$$
q(s_1,a_1,w_t)=2.0.
$$

所以 TD error 是：

$$
\delta_t
=
3.7-2.0
=
1.7.
$$

critic 更新：

$$
\begin{aligned}
w_{t+1}
&=
w_t
+
\alpha_w\delta_t\nabla_w q(s_1,a_1,w_t)\\
&=
w_t
+
0.2\times1.7\times0.5\\
&=
w_t+0.17.
\end{aligned}
$$

读法：环境反馈和下一步价值说明，critic 原来把 $(s_1,a_1)$ 估低了，所以 $w$ 被往“提高 $q(s_1,a_1,w)$”的方向推。

把一轮流程合起来看：

| 步骤 | 用到什么 | 公式 / 为什么需要 | 更新谁 |
|---|---|---|---|
| 1. 采样 $a_t$ | actor 的 $\pi(a\mid s_t,\theta_t)$ | 得到当前动作样本 $(s_t,a_t)$ | 暂不更新 |
| 2. 观察 $r_{t+1},s_{t+1}$ | 环境反馈 | 得到 TD target 的即时奖励和下一状态 | 暂不更新 |
| 3. 采样 $a_{t+1}$ | actor 的 $\pi(a\mid s_{t+1},\theta_t)$ | 为 Sarsa target 准备 $q(s_{t+1},a_{t+1},w_t)$ | 暂不更新 |
| 4. actor 更新 | $q(s_t,a_t,w_t)$ | $\theta_{t+1}=\theta_t+\alpha_\theta\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)q(s_t,a_t,w_t)$ | 更新 $\theta$ |
| 5. critic 更新 | TD error $\delta_t$ | $\delta_t=r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)-q(s_t,a_t,w_t)$ | 更新 $w$ |

第 3 步之所以要采样 $a_{t+1}$，不是为了马上更新 actor，而是因为这里的 critic 用的是 Sarsa 形式。Sarsa 的 TD target 必须沿着当前策略实际采样到的下一步动作往前看：

$$
r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t).
$$

如果不采样 $a_{t+1}$，critic 就缺少 target 里的下一步动作价值项，也就无法计算：

$$
\delta_t
=
r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)-q(s_t,a_t,w_t).
$$

这个例子体现了 QAC 的核心：actor 负责按概率行动并更新策略，critic 负责根据 TD 误差修正价值评价。下一轮 actor 再更新时，就会用到更好的 critic 评价。

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

## 10.2 Advantage actor-critic (A2C)（优势 Actor-Critic）

**要解决的问题**：10.1 的 QAC 直接用 $q(s_t,a_t,w_t)$ 乘策略梯度，但绝对动作价值可能方差很大。本节要说明：把动作价值减去一个 baseline 基线，不改变策略梯度的期望，却能降低样本估计的方差；当 baseline 选成 $v_\pi(s)$ 时，就得到 advantage function 优势函数和 A2C。

### 1. 为什么可以减 baseline？

第 9 章的策略梯度核心形式是：

$$
\mathbb E_{S\sim\eta,\ A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)q_\pi(S,A)
\right].
$$

A2C 的第一步是说：可以从 $q_\pi(S,A)$ 里减去一个只依赖状态的函数 $b(S)$：

$$
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)q_\pi(S,A)
\right]
=
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)
\left(q_\pi(S,A)-b(S)\right)
\right]. \tag{10.3}
$$

这叫 baseline invariance 基线不变性：减去 baseline 后，期望梯度不变。

为什么？只要证明多减掉的那一项期望为 0：

$$
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)b(S)
\right]
=0.
$$

展开看：

$$
\begin{aligned}
&\mathbb E_{S\sim\eta,\ A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)b(S)
\right]\\
&=
\sum_s\eta(s)\sum_a
\pi(a\mid s,\theta_t)
\nabla_\theta\ln\pi(a\mid s,\theta_t)b(s).
\end{aligned}
$$

用 log-trick 反过来：

$$
\pi(a\mid s,\theta_t)
\nabla_\theta\ln\pi(a\mid s,\theta_t)
=
\nabla_\theta\pi(a\mid s,\theta_t).
$$

所以：

$$
\begin{aligned}
&\sum_s\eta(s)\sum_a
\pi(a\mid s,\theta_t)
\nabla_\theta\ln\pi(a\mid s,\theta_t)b(s)\\
&=
\sum_s\eta(s)b(s)
\sum_a\nabla_\theta\pi(a\mid s,\theta_t)\\
&=
\sum_s\eta(s)b(s)
\nabla_\theta
\sum_a\pi(a\mid s,\theta_t)\\
&=
\sum_s\eta(s)b(s)\nabla_\theta 1
=0.
\end{aligned}
$$

关键点是：对固定状态 $s$，baseline $b(s)$ 对所有动作都一样，所以它不会偏向某个动作；而所有动作概率加起来永远是 1，导数就是 0。

⚠️ 易错点：baseline 只能依赖状态 $S$，不能随动作 $A$ 任意变化。因为如果 $b=b(S,A)$，它就可能改变不同动作之间的相对更新权重，期望梯度不再保证不变。

### 2. baseline 有什么用：不改均值，降低方差

定义随机梯度样本：

$$
X(S,A)
\doteq
\nabla_\theta\ln\pi(A\mid S,\theta_t)
\left[
q_\pi(S,A)-b(S)
\right]. \tag{10.4}
$$

真实梯度是：

$$
\mathbb E[X(S,A)].
$$

baseline invariance 告诉我们：换不同的 $b(S)$，这个期望不变。但样本方差会变。

这件事非常像估计一个数：如果随机样本上下波动很大，单个样本就很不可靠；如果样本围绕均值波动很小，单个样本更接近真实期望。

| baseline 选择 | 期望梯度 | 样本方差 |
|---|---|---|
| $b(S)=0$ | 不变 | 可能很大 |
| 合适的 $b(S)$ | 不变 | 可以更小 |

REINFORCE 和 QAC 可以看成默认用 $b=0$。A2C 的改进，就是选一个更好的 baseline。

### 3. 最优 baseline 和实用 baseline

原书给出能最小化方差的 optimal baseline 最优基线：

$$
b^*(s)
=
\frac{
\mathbb E_{A\sim\pi}
\left[
\left\|
\nabla_\theta\ln\pi(A\mid s,\theta_t)
\right\|^2
q_\pi(s,A)
\right]
}{
\mathbb E_{A\sim\pi}
\left[
\left\|
\nabla_\theta\ln\pi(A\mid s,\theta_t)
\right\|^2
\right]
}. \tag{10.5}
$$

这个式子的直觉是：不是简单平均动作价值，而是按 $\|\nabla_\theta\ln\pi\|^2$ 给不同动作加权。哪个动作对策略参数更敏感，它的价值就对方差影响更大。

但这个最优 baseline 太复杂，实践中不方便用。于是原书说：如果去掉权重 $\|\nabla_\theta\ln\pi\|^2$，得到一个简洁的次优 baseline：

$$
b^\dagger(s)
=
\mathbb E_{A\sim\pi}
\left[
q_\pi(s,A)
\right]
=
v_\pi(s).
$$

这就是 state value 状态价值。

这里不矛盾。虽然 $v_\pi(s)$ 可以展开成含动作符号的式子，但那个动作只是“求和用的哑变量”，不是当前样本动作 $A$：

$$
v_\pi(s)
=
\sum_{a'\in\mathcal A}
\pi(a'\mid s)q_\pi(s,a').
$$

读法是：固定状态 $s$ 后，把这个状态下所有可能动作的 $q_\pi(s,a')$ 按策略概率加权平均。平均完成以后，结果就是一个数，只由 $s$ 决定，不再随“这次实际采到了哪个动作 $A$”改变。

对比一下就清楚了：

| 写法 | 是否允许做 baseline | 原因 |
|---|---|---|
| $b(s)=v_\pi(s)=\sum_{a'}\pi(a'\mid s)q_\pi(s,a')$ | 允许 | 对固定 $s$，它已经对所有动作求平均，是同一个数 |
| $b(s,A)=q_\pi(s,A)$ | 不允许随便当 baseline | 它会随当前采样动作 $A$ 变化，会改变动作之间的相对权重 |

所以 baseline 条件里的“不依赖动作”更准确地说是：不能依赖当前被采样出来的随机动作 $A$。公式内部用 $a'$ 把所有动作求和，不算违反这个条件。

读法：在状态 $s$ 下，所有动作价值按当前策略平均，就是这个状态本身的价值。用它做 baseline 等于问：

> 这次动作比当前状态下的平均水平好多少？

### 4. advantage function：动作比平均水平好多少

当 baseline 取成 $v_\pi(S)$ 时，策略梯度变成：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)
\left(
q_\pi(S,A)-v_\pi(S)
\right)
\right].
$$

定义 advantage function 优势函数：

$$
\delta_\pi(S,A)
\doteq
q_\pi(S,A)-v_\pi(S). \tag{10.7}
$$

它的含义是：动作 $A$ 相对于状态 $S$ 下平均动作表现的优势。

因为

$$
v_\pi(s)
=
\sum_{a'}\pi(a'\mid s)q_\pi(s,a'),
$$

所以 $v_\pi(s)$ 是当前策略下动作价值的平均值。

| 情况 | 含义 | actor 应该怎样 |
|---|---|---|
| $\delta_\pi(s,a)>0$ | 这个动作比平均动作更好 | 增强这个动作概率 |
| $\delta_\pi(s,a)<0$ | 这个动作比平均动作更差 | 降低这个动作概率 |
| $\delta_\pi(s,a)=0$ | 这个动作差不多就是平均水平 | 不需要强烈调整 |

这比 QAC 直接用 $q(s,a,w)$ 更合理：actor 真正关心的不是“动作价值绝对值是多少”，而是“这个动作相对同状态下其他动作是否更好”。

### 5. 从 advantage 到 A2C 更新式

理论版本是：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)
\delta_\pi(S,A)
\right].
$$

样本版本是：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
\delta_t(s_t,a_t). \tag{10.8}
$$

其中：

$$
\delta_t(s_t,a_t)
=
q_t(s_t,a_t)-v_t(s_t).
$$

如果 $q_t$ 和 $v_t$ 都用 Monte Carlo 估计，这叫 REINFORCE with a baseline。

如果 $q_t$ 和 $v_t$ 都用 TD learning 估计，就是 advantage actor-critic，简称 A2C。

### 6. 为什么 A2C 可以只学 $v(s,w)$？

直接估计 advantage 似乎需要两个网络：

$$
q_t(s_t,a_t)-v_t(s_t).
$$

一个估计 $q_\pi(s,a)$，一个估计 $v_\pi(s)$。

但 A2C 常用 TD error 直接近似 advantage：

$$
q_t(s_t,a_t)-v_t(s_t)
\approx
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t).
$$

为什么合理？因为根据动作价值定义：

$$
q_\pi(s_t,a_t)
=
\mathbb E
\left[
R_{t+1}+\gamma v_\pi(S_{t+1})
\mid
S_t=s_t,A_t=a_t
\right].
$$

所以：

$$
q_\pi(s_t,a_t)-v_\pi(s_t)
=
\mathbb E
\left[
R_{t+1}
+
\gamma v_\pi(S_{t+1})
-
v_\pi(S_t)
\mid
S_t=s_t,A_t=a_t
\right].
$$

也就是说，单步 TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
$$

正是在用一条样本估计 advantage。

这有一个重要好处：A2C 只需要一个 critic 网络 $v(s,w)$，不用同时维护 $q(s,a,w)$ 和 $v(s,w)$ 两个价值网络。

### 7. Algorithm 10.2：A2C / TD Actor-Critic

**输入**：

- policy 函数 $\pi(a\mid s,\theta_0)$
- value 函数 $v(s,w_0)$
- 学习率 $\alpha_\theta,\alpha_w>0$

每个时间步 $t$：

1. 按当前策略采样动作：

$$
a_t\sim\pi(\cdot\mid s_t,\theta_t).
$$

2. 执行动作，观察：

$$
r_{t+1},\ s_{t+1}.
$$

3. 计算 advantage / TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t).
$$

4. actor 更新策略：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\delta_t
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t).
$$

5. critic 更新状态价值：

$$
w_{t+1}
=
w_t
+
\alpha_w
\delta_t
\nabla_w v(s_t,w_t).
$$

和 QAC 对比：

| 对比点 | QAC | A2C |
|---|---|---|
| critic 学什么 | $q(s,a,w)$ | $v(s,w)$ |
| actor 乘什么信号 | 动作价值 $q(s_t,a_t,w_t)$ | advantage / TD error $\delta_t$ |
| 是否要采样 $a_{t+1}$ | 需要，用于 Sarsa target | 不需要，只用 $v(s_{t+1},w_t)$ |
| 更新直觉 | 绝对动作价值越大越增强 | 比平均水平好才增强 |

⚠️ 易错点：A2C 的 $\delta_t$ 在这里有两层意义。对 critic 来说，它是 TD error，用来修正 $v(s,w)$；对 actor 来说，它是 advantage 的样本估计，用来判断当前动作相对平均表现好不好。

### 8. 一个最小数值例子：为什么 advantage 更合理

假设在同一个状态 $s$，critic 估计：

$$
v(s,w_t)=10.
$$

现在 actor 采到动作 $a$，环境给出：

$$
r_{t+1}=1,\qquad v(s_{t+1},w_t)=12,\qquad \gamma=0.9.
$$

TD error 是：

$$
\delta_t
=
1+0.9\times12-10
=
1.8.
$$

虽然当前状态价值已经很高，$v(s,w_t)=10$，但这个动作带来的结果比当前平均水平还要好，所以 $\delta_t>0$，actor 会增强这个动作。

再看另一个动作 $a'$。假设采到它后：

$$
r_{t+1}=1,\qquad v(s_{t+1},w_t)=9.
$$

则：

$$
\delta_t
=
1+0.9\times9-10
=
-0.9.
$$

虽然仍然拿到了正奖励 $1$，但相对于状态 $s$ 的平均水平 $10$ 来说，这个动作结果不好，所以 actor 会降低它的概率。

这就是 advantage 的核心直觉：

> 不看“绝对回报是不是正”，而看“比这个状态下的平均表现好不好”。

### 9. 和 10.1 的闭环

10.1 的 QAC 说：

$$
\text{actor 更新用 }q(s_t,a_t,w_t).
$$

10.2 的 A2C 改成：

$$
\text{actor 更新用 }\delta_t
=
r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w_t).
$$

所以 A2C 的进步是：

$$
\text{用相对优势代替绝对价值}
\quad+\quad
\text{用一个 }v(s,w)\text{ critic 同时服务 actor 和 critic}.
$$

另外，A2C 的策略 $\pi(a\mid s,\theta)$ 本身就是随机的，所以它可以直接用于探索，不需要像某些 value-based 控制方法那样额外加 $\varepsilon$-greedy。

---

## 10.3 Off-policy actor-critic（离策略 Actor-Critic）

**要解决的问题**：前面的 REINFORCE、QAC、A2C 都是 on-policy 同策略方法：用当前策略 $\pi$ 采样，也更新同一个 $\pi$。本节要回答：如果样本不是由目标策略 $\pi$ 产生，而是由另一个 behavior policy 行为策略 $\beta$ 产生，还能不能用这些样本来更新 $\pi$？

### 1. 为什么前面的 actor-critic 是 on-policy？

A2C 的理论梯度形状是：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{S\sim\eta,\ A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta_t)
\left(q_\pi(S,A)-v_\pi(S)\right)
\right].
$$

注意动作分布是：

$$
A\sim\pi.
$$

也就是说，如果我们想用样本近似这个期望，动作样本应该按当前目标策略 $\pi$ 生成。于是：

| 角色 | on-policy 情形 |
|---|---|
| 采样数据的策略 | $\pi$ |
| 正在优化的目标策略 | $\pi$ |

两者是同一个，所以叫 on-policy。

off-policy 离策略情形则是：

| 角色 | off-policy 情形 |
|---|---|
| 采样数据的行为策略 | $\beta$ |
| 正在优化的目标策略 | $\pi$ |

问题来了：样本动作来自 $\beta$，但梯度期望想要的是 $\pi$ 下的动作分布。中间的分布不一致，就需要 importance sampling 重要性采样来校正。

### 2. importance sampling：用 $p_1$ 的样本估计 $p_0$ 的期望

先离开强化学习，看一个普通期望。目标是：

$$
\mathbb E_{X\sim p_0}[X].
$$

如果样本 $x_i$ 本来就是按 $p_0$ 采的，那么直接平均：

$$
\frac{1}{n}\sum_{i=1}^n x_i
$$

就可以估计这个期望。

但现在样本不是从 $p_0$ 来的，而是从另一个分布 $p_1$ 来的。直接平均会估计成：

$$
\mathbb E_{X\sim p_1}[X],
$$

不是我们想要的 $\mathbb E_{X\sim p_0}[X]$。

importance sampling 的技巧是乘上一个权重：

$$
\frac{p_0(x)}{p_1(x)}.
$$

推导很短：

$$
\begin{aligned}
\mathbb E_{X\sim p_0}[X]
&=
\sum_x p_0(x)x\\
&=
\sum_x p_1(x)
\frac{p_0(x)}{p_1(x)}x\\
&=
\mathbb E_{X\sim p_1}
\left[
\frac{p_0(X)}{p_1(X)}X
\right]. \tag{10.9}
\end{aligned}
$$

所以用 $p_1$ 的样本估计 $p_0$ 的期望时，应当算加权平均：

$$
\frac{1}{n}\sum_{i=1}^n
\frac{p_0(x_i)}{p_1(x_i)}x_i. \tag{10.10}
$$

其中

$$
\frac{p_0(x_i)}{p_1(x_i)}
$$

叫 importance weight 重要性权重。

直觉：

| 情况 | 权重 | 含义 |
|---|---:|---|
| $p_0(x)>p_1(x)$ | $>1$ | 这个样本在目标分布中更常见，要放大 |
| $p_0(x)<p_1(x)$ | $<1$ | 这个样本在目标分布中过度出现，要缩小 |
| $p_0(x)=p_1(x)$ | $=1$ | 两个分布一致，普通平均即可 |

⚠️ 易错点：必须满足 support coverage 支撑覆盖：如果 $p_0(x)>0$，就必须有 $p_1(x)>0$。否则目标分布里可能出现的样本，行为分布永远采不到，再怎么加权也估不出来。

### 3. 一个最小数值例子：为什么要加权

原书用 $X\in\{+1,-1\}$ 举例。

目标分布 $p_0$ 是：

| $X$ | $p_0(X)$ |
|---|---:|
| $+1$ | 0.5 |
| $-1$ | 0.5 |

所以真实目标期望是：

$$
\mathbb E_{p_0}[X]
=
1\times0.5+(-1)\times0.5
=0.
$$

但样本来自行为分布 $p_1$：

| $X$ | $p_1(X)$ |
|---|---:|
| $+1$ | 0.8 |
| $-1$ | 0.2 |

如果直接平均，长期会得到：

$$
\mathbb E_{p_1}[X]
=
1\times0.8+(-1)\times0.2
=0.6.
$$

这明显偏离目标期望 0。

用 importance weight：

| $X$ | 权重 $\frac{p_0(X)}{p_1(X)}$ | 加权后的贡献 |
|---|---:|---:|
| $+1$ | $0.5/0.8=0.625$ | $0.625\times(+1)$ |
| $-1$ | $0.5/0.2=2.5$ | $2.5\times(-1)$ |

直觉上，$+1$ 在行为分布里被采得太多，所以权重要压小；$-1$ 被采得太少，所以权重要放大。

### 4. 回到 off-policy policy gradient

在 off-policy actor-critic 中：

| 记号 | 含义 |
|---|---|
| $\beta(a\mid s)$ | behavior policy 行为策略，负责生成样本 |
| $\pi(a\mid s,\theta)$ | target policy 目标策略，我们真正想优化 |
| $d_\beta$ | 行为策略 $\beta$ 下的平稳状态分布 |
| $\rho$ | 从 $d_\beta$ 出发、按目标策略 $\pi$ 未来折扣访问到的状态分布 |

本节选择的目标函数是：

$$
J(\theta)
=
\sum_s d_\beta(s)v_\pi(s)
=
\mathbb E_{S\sim d_\beta}[v_\pi(S)].
$$

读法：起点状态按行为策略 $\beta$ 的长期分布来加权，但状态价值看的是目标策略 $\pi$ 的价值。

Theorem 10.1 给出 off-policy policy gradient：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{S\sim\rho,\ A\sim\beta}
\left[
\underbrace{
\frac{\pi(A\mid S,\theta)}
{\beta(A\mid S)}
}_{\text{importance weight}}
\nabla_\theta\ln\pi(A\mid S,\theta)
q_\pi(S,A)
\right]. \tag{10.11}
$$

和 on-policy 版本相比：

$$
\mathbb E_{S,\ A\sim\pi}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)q_\pi(S,A)
\right],
$$

差别只有两个：

| 对比点 | on-policy | off-policy |
|---|---|---|
| 动作样本来自 | $A\sim\pi$ | $A\sim\beta$ |
| 是否需要重要性权重 | 不需要 | 需要 $\frac{\pi(A\mid S,\theta)}{\beta(A\mid S)}$ |

这个比值的作用就是：把行为策略 $\beta$ 采出来的动作样本，校正成目标策略 $\pi$ 下的期望。

### 5. 状态分布 $\rho$ 是什么？

Theorem 10.1 里的状态分布是：

$$
\rho(s)
\doteq
\sum_{s'\in\mathcal S}
d_\beta(s')
\Pr_\pi(s\mid s').
$$

读法：先按 $d_\beta$ 选择起点 $s'$，再看如果以后按目标策略 $\pi$ 走，未来折扣访问到 $s$ 的总权重。

这和第 9 章 Theorem 9.2 的 $\rho_\pi$ 很像：

| 分布 | 起点分布 | 未来按谁走 |
|---|---|---|
| $\rho_\pi$ | $d_0$ | $\pi$ |
| $\rho$ | $d_\beta$ | $\pi$ |

所以 off-policy 并不是说状态和动作都完全来自 $\beta$ 后就结束了。理论梯度里，动作采样用 $\beta$，再用 $\pi/\beta$ 校正；状态分布则是“从 $\beta$ 的长期状态分布出发，按 $\pi$ 的未来影响展开”。

### 6. off-policy actor-critic 的更新式

和 10.2 一样，可以加入 baseline：

$$
q_\pi(S,A)-v_\pi(S).
$$

于是样本版 actor 更新是：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\frac{\pi(a_t\mid s_t,\theta_t)}
{\beta(a_t\mid s_t)}
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
\left(q_t(s_t,a_t)-v_t(s_t)\right).
$$

再用 TD error 近似 advantage：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t).
$$

得到：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\frac{\pi(a_t\mid s_t,\theta_t)}
{\beta(a_t\mid s_t)}
\delta_t
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t).
$$

critic 也用同样的 importance weight：

$$
w_{t+1}
=
w_t
+
\alpha_w
\frac{\pi(a_t\mid s_t,\theta_t)}
{\beta(a_t\mid s_t)}
\delta_t
\nabla_w v(s_t,w_t).
$$

⚠️ 易错点：重要性采样不只修正 actor。原书强调，critic 也从 on-policy 变成 off-policy，因此 critic 的 TD 更新也乘上同一个 importance weight。

### 7. Algorithm 10.3：Off-policy actor-critic

**输入**：

- 已给定的行为策略 $\beta(a\mid s)$
- 目标策略 $\pi(a\mid s,\theta_0)$
- value 函数 $v(s,w_0)$
- 学习率 $\alpha_\theta,\alpha_w>0$

每个时间步：

1. 用行为策略采样动作：

$$
a_t\sim\beta(\cdot\mid s_t).
$$

2. 执行动作，观察：

$$
r_{t+1},s_{t+1}.
$$

3. 计算 TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t).
$$

4. 计算 importance weight：

$$
c_t
=
\frac{\pi(a_t\mid s_t,\theta_t)}
{\beta(a_t\mid s_t)}.
$$

5. actor 更新：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta c_t\delta_t
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t).
$$

6. critic 更新：

$$
w_{t+1}
=
w_t
+
\alpha_w c_t\delta_t
\nabla_w v(s_t,w_t).
$$

### 8. 一个最小数值例子：重要性权重怎么改变更新力度

假设在状态 $s$，行为策略和目标策略对动作 $a$ 的概率是：

$$
\beta(a\mid s)=0.20,\qquad
\pi(a\mid s,\theta_t)=0.60.
$$

那么 importance weight 是：

$$
c_t
=
\frac{0.60}{0.20}
=3.
$$

如果这一步 TD error 是：

$$
\delta_t=1.5,
$$

那么 actor 实际乘上的信号是：

$$
c_t\delta_t=3\times1.5=4.5.
$$

直觉：这个动作在目标策略 $\pi$ 里比在行为策略 $\beta$ 里更常出现。既然我们用的是 $\beta$ 的样本，而 $\beta$ 采到这个动作偏少，那么一旦采到了，就要放大它的贡献。

反过来，如果：

$$
\beta(a\mid s)=0.60,\qquad
\pi(a\mid s,\theta_t)=0.20,
$$

则：

$$
c_t=\frac{0.20}{0.60}=\frac13.
$$

同样的 $\delta_t=1.5$，实际信号变成：

$$
c_t\delta_t=0.5.
$$

直觉：这个动作在行为策略里被采得太多，但目标策略并不那么常选它，所以要降低这类样本对目标策略更新的影响。

### 9. 和 10.2 的闭环

A2C 是：

$$
\theta_{t+1}
=
\theta_t+\alpha_\theta\delta_t\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t).
$$

Off-policy actor-critic 只是多了一个校正项：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\underbrace{
\frac{\pi(a_t\mid s_t,\theta_t)}
{\beta(a_t\mid s_t)}
}_{\text{校正 }\beta\text{ 和 }\pi\text{ 的差异}}
\delta_t
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t).
$$

所以这一节的核心可以记成：

$$
\boxed{
\text{off-policy}
=
\text{on-policy actor-critic}
+
\text{importance weight } \frac{\pi}{\beta}
}
$$

---

## 10.4 Deterministic actor-critic（确定性 Actor-Critic）

**要解决的问题**：前面几节的 actor 都是 stochastic policy 随机策略 $\pi(a\mid s,\theta)$，它输出的是“每个动作的概率”。本节换成 deterministic policy 确定性策略 $\mu(s,\theta)$，它直接输出一个动作。这样特别适合连续动作空间，也自然引出 deterministic policy gradient 确定性策略梯度。

### 1. 从随机策略到确定性策略：actor 到底输出什么？

前面 10.1-10.3 的 actor 是：

$$
\pi(a\mid s,\theta).
$$

给定状态 $s$，它输出的是一个动作分布。例如：

$$
\pi(a_1\mid s,\theta)=0.7,\qquad
\pi(a_2\mid s,\theta)=0.3.
$$

所以 stochastic actor 的更新逻辑是：

> 这次采到了动作 $a_t$，critic 觉得它好不好？如果好，就提高它的概率；如果不好，就降低它的概率。

确定性策略改成：

$$
a=\mu(s,\theta).
$$

给定状态 $s$，它不再输出概率，而是直接输出动作本身。例如连续控制里：

$$
\mu(s,\theta)=1.37
$$

就表示 actor 直接选择动作 $a=1.37$。

两者对比一下：

| 策略类型 | 记号 | actor 输出 | actor 更新的直觉 |
|---|---|---|---|
| stochastic policy 随机策略 | $\pi(a\mid s,\theta)$ | 每个动作的概率 | 调整“采到某动作的概率” |
| deterministic policy 确定性策略 | $\mu(s,\theta)$ | 一个具体动作 | 直接把输出动作往更好的方向推 |

⚠️ 易错点：确定性策略不是“不需要探索”。它只是 target policy 目标策略 $\mu$ 是确定性的。实际收集数据时，仍然可以用带噪声的 behavior policy 行为策略 $\beta$ 来探索。

### 2. 确定性策略梯度长什么样？

随机策略梯度的核心形状是：

$$
\nabla_\theta J(\theta)
\approx
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)q_\pi(S,A)
\right].
$$

这里有两个关键对象：

- $\nabla_\theta\ln\pi(A\mid S,\theta)$：告诉 actor 怎样改变“动作概率”。
- $q_\pi(S,A)$：critic 评价这个动作好不好。

确定性策略没有 $\ln\pi(A\mid S,\theta)$，因为它不输出概率。它输出的是动作：

$$
A=\mu(S,\theta).
$$

所以更新方向要换成 chain rule 链式法则：

$$
\theta
\longrightarrow
\mu(S,\theta)
\longrightarrow
q_\mu(S,\mu(S,\theta)).
$$

也就是：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{S\sim\eta}
\left[
\nabla_\theta\mu(S)
\left(\nabla_a q_\mu(S,a)\right)\big|_{a=\mu(S)}
\right]
}
\tag{10.14}
$$

读法：

- $\nabla_a q_\mu(S,a)\big|_{a=\mu(S)}$：critic 告诉我们，在当前动作附近，动作 $a$ 往哪里改能让 $q$ 变大。
- $\nabla_\theta\mu(S)$：actor 告诉我们，参数 $\theta$ 怎么改会让输出动作 $\mu(S,\theta)$ 往那个方向动。

所以 deterministic actor 的一句话版本是：

$$
\boxed{
\text{critic 对动作求梯度，actor 用链式法则把这个动作梯度传回 }\theta
}
$$

⚠️ 记号小心：原书写

$$
\left(\nabla_a q_\mu(S,a)\right)\big|_{a=\mu(S)}
$$

而不写成 $\nabla_a q_\mu(S,\mu(S))$。原因是前者明确表示“先把 $q_\mu(S,a)$ 当作 $a$ 的函数求导，再代入 $a=\mu(S)$”。后者容易让人误以为 $q_\mu(S,\mu(S))$ 已经不是关于 $a$ 的函数了。

### 3. 两个梯度公式从哪里来？正文只看主线

随机策略梯度里的

$$
\mathbb E_{S,A}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)q_\pi(S,A)
\right]
$$

来自第 9 章的 policy gradient theorem。先有求和形式：

$$
\sum_s\eta(s)\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a),
$$

再用 log-trick：

$$
\nabla_\theta\pi(a\mid s,\theta)
=
\pi(a\mid s,\theta)\nabla_\theta\ln\pi(a\mid s,\theta),
$$

于是它变成“按 $S\sim\eta,\ A\sim\pi$ 采样”的期望。

确定性策略梯度不是把上式直接代入 $A=\mu(S)$。它从另一个起点来：

$$
v_\mu(s)=q_\mu(s,\mu(s,\theta)).
$$

为了看清求导路径，也可以把 $q_\mu$ 对策略参数的依赖显式写成：

$$
v_\mu(s)=Q(s,\mu(s,\theta);\theta),
\qquad
Q(s,a;\theta)\doteq q_{\mu_\theta}(s,a).
$$

对这个复合函数求导，会出现当前状态的局部 actor 方向：

$$
u(s)
=
\nabla_\theta\mu(s)\cdot
\left(\nabla_a q_\mu(s,a)\right)\big|_{a=\mu(s)}.
$$

这里正文只写 actor 真正用于更新的局部方向 $u(s)$。完整推导里还有一项 $\nabla_\theta q_\mu(s,a)$，它表示“真实 $q_\mu$ 会因为未来策略改变而改变”；这项会通过 Bellman 递归展开到未来状态，最后被整理进状态分布 $\eta$ 里。

再把未来状态影响用 Bellman 递归展开，最后得到：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{S\sim\eta}
\left[
u(S)
\right]
=
\mathbb E_{S\sim\eta}
\left[
\nabla_\theta\mu(S)
\left(\nabla_a q_\mu(S,a)\right)\big|_{a=\mu(S)}
\right].
$$

完整推导，包括随机策略公式的来源、log-trick 怎么变成期望、以及确定性策略里 $A=\mu(s)$ 与 $u(s)$ 的关系，放在补充笔记：

> [[10_supplement-policy-gradient-formula-origins|补充：随机策略梯度与确定性策略梯度公式来源]]

这也是确定性策略适合连续动作空间的原因之一。随机策略在连续动作空间里常常要处理概率密度、采样和高方差问题；确定性策略每个状态只输出一个动作，critic 再告诉这个动作该往哪边挪。

### 4. 两个目标下的状态分布 $\eta$

Theorem 10.2 先把结论写成统一形式：

$$
\nabla_\theta J(\theta)
=
\sum_{s\in\mathcal S}
\eta(s)
\nabla_\theta\mu(s)
\left(\nabla_a q_\mu(s,a)\right)\big|_{a=\mu(s)}.
$$

这里 $\eta$ 不是一个新的固定符号，而是“看你优化哪个目标”。

| 目标 | $J(\theta)$ | 状态分布 $\eta$ | 对应定理 |
|---|---|---|---|
| 折扣平均价值 | $J(\theta)=d_0^Tv_\mu$ | $\rho_\mu(s)=\sum_{s'}d_0(s')\Pr_\mu(s\mid s')$ | Theorem 10.3 |
| 平均奖励 | $J(\theta)=\bar r_\mu$ | $d_\mu(s)$ | Theorem 10.4 |

折扣情形下，$\rho_\mu$ 的意思和第 9 章很像：从初始分布 $d_0$ 出发，在策略 $\mu$ 下，按折扣累计访问状态的总权重。

$$
\rho_\mu(s)
=
\sum_{s'}d_0(s')\Pr_\mu(s\mid s'),
\qquad
\Pr_\mu(s\mid s')
=
\sum_{k=0}^{\infty}\gamma^k[P_\mu^k]_{s's}.
$$

平均奖励情形下，用的是 $\mu$ 诱导的平稳分布：

$$
d_\mu^TP_\mu=d_\mu^T.
$$

所以可以把 Theorem 10.2 记成：

$$
\boxed{
\text{确定性策略梯度的局部形状不变，差别只在状态按什么分布加权}
}
$$

### 5. Lemma 10.1 在说什么？

原书为了推导 Theorem 10.3，先给出 Lemma 10.1：

$$
\nabla_\theta v_\mu(s)
=
\sum_{s'\in\mathcal S}
\Pr_\mu(s'\mid s)
\nabla_\theta\mu(s')
\left(\nabla_a q_\mu(s',a)\right)\big|_{a=\mu(s')}.
\tag{10.16}
$$

这条式子的直觉是：

> 从当前状态 $s$ 出发，$v_\mu(s)$ 会受到未来所有可能访问到的状态 $s'$ 的影响。每个未来状态 $s'$ 上，actor 都可能因为 $\theta$ 改变动作 $\mu(s')$，从而改变价值。

所以右边有三层含义：

| 项 | 含义 |
|---|---|
| $\Pr_\mu(s'\mid s)$ | 从 $s$ 出发，将来折扣意义下访问到 $s'$ 的总权重 |
| $\nabla_\theta\mu(s')$ | 参数 $\theta$ 对未来状态 $s'$ 的动作输出有什么影响 |
| $\nabla_a q_\mu(s',a)\big\vert_{a=\mu(s')}$ | 在未来状态 $s'$，动作往哪边变会提高 critic 价值 |

这和第 9 章你之前问过的“为什么 $\nabla v$ 会看到未来状态”是同一个结构。当前价值 $v_\mu(s)$ 不只由当前动作决定，还由策略在未来状态上的动作决定。

### 6. Algorithm 10.4：确定性 actor-critic 怎么跑？

**输入**：

- behavior policy 行为策略 $\beta(a\mid s)$，负责探索并和环境交互
- deterministic target policy 确定性目标策略 $\mu(s,\theta_0)$
- critic 动作价值函数 $q(s,a,w_0)$
- 学习率 $\alpha_\theta,\alpha_w>0$

每个时间步：

1. 用行为策略采样动作并和环境交互：

$$
a_t\sim\beta(\cdot\mid s_t),
\qquad
\text{observe } r_{t+1},s_{t+1}.
$$

2. critic 计算 TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma q(s_{t+1},\mu(s_{t+1},\theta_t),w_t)
-
q(s_t,a_t,w_t).
$$

3. actor 用确定性策略梯度更新：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\nabla_\theta\mu(s_t,\theta_t)
\left(\nabla_a q(s_t,a,w_t)\right)\big|_{a=\mu(s_t)}.
$$

4. critic 更新：

$$
w_{t+1}
=
w_t
+
\alpha_w\delta_t\nabla_w q(s_t,a_t,w_t).
$$

和前面算法并排看时，最好不要只看 actor 信号。actor-critic 每一步其实都在做同一件事：

$$
\boxed{
\text{先构造 critic 的 TD error }\delta_t
\quad\Longrightarrow\quad
\text{再分别更新 actor 的 }\theta\text{ 和 critic 的 }w
}
$$

先看 TD error 是怎么来的：

| 算法                  | 行为动作怎么来                                                                               | TD error $\delta_t$                                                  |
| ------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| QAC                 | $a_t\sim\pi(\cdot\mid s_t,\theta_t)$，再采样 $a_{t+1}\sim\pi(\cdot\mid s_{t+1},\theta_t)$ | $r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)-q(s_t,a_t,w_t)$               |
| A2C                 | $a_t\sim\pi(\cdot\mid s_t,\theta_t)$                                                  | $r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w_t)$                           |
| off-policy A2C / AC | $a_t\sim\beta(\cdot\mid s_t)$                                                         | $r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w_t)$                           |
| deterministic AC    | $a_t\sim\beta(\cdot\mid s_t)$，但下一步用目标动作 $\mu(s_{t+1},\theta_t)$                       | $r_{t+1}+\gamma q(s_{t+1},\mu(s_{t+1},\theta_t),w_t)-q(s_t,a_t,w_t)$ |

再看 $\theta_{t+1}$ 和 $w_{t+1}$ 分别怎么更新：

| 算法 | actor 更新 $\theta_{t+1}$ | critic 更新 $w_{t+1}$ | 直观区别 |
|---|---|---|---|
| QAC | $\theta_t+\alpha_\theta\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)q(s_t,a_t,w_t)$ | $w_t+\alpha_w\delta_t\nabla_w q(s_t,a_t,w_t)$ | actor 用“绝对动作价值”增强采样动作概率 |
| A2C | $\theta_t+\alpha_\theta\delta_t\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)$ | $w_t+\alpha_w\delta_t\nabla_w v(s_t,w_t)$ | actor 用 TD error 近似 advantage，只奖励“比预期好”的动作 |
| off-policy A2C / AC | $\theta_t+\alpha_\theta c_t\delta_t\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)$ | $w_t+\alpha_w c_t\delta_t\nabla_w v(s_t,w_t)$ | 和 A2C 同形，但用 $c_t=\frac{\pi(a_t\mid s_t,\theta_t)}{\beta(a_t\mid s_t)}$ 校正样本来自 $\beta$ |
| deterministic AC | $\theta_t+\alpha_\theta\nabla_\theta\mu(s_t,\theta_t)\nabla_a q(s_t,a,w_t)\big\vert_{a=\mu(s_t)}$ | $w_t+\alpha_w\delta_t\nabla_w q(s_t,a_t,w_t)$ | actor 不调概率，直接把动作输出往 critic 认为更好的方向移动 |

这样看会更直观：

- $\delta_t$ 是 critic 的“当前估计错了多少”。
- $w_{t+1}$ 总是用 $\delta_t$ 修 critic，让价值估计更准。
- $\theta_{t+1}$ 是 actor 的策略更新；QAC/A2C/off-policy A2C 改动作概率，deterministic AC 改动作本身。
- off-policy A2C 的 $\delta_t$ 和 A2C 一样，真正新增的是 $c_t$：它不改变 TD target 的形状，而是改变这条样本对 actor 和 critic 更新的权重。

### 7. 为什么这里 off-policy 却没有 importance sampling？

10.3 的 off-policy actor-critic 要乘：

$$
\frac{\pi(a_t\mid s_t,\theta_t)}{\beta(a_t\mid s_t)}.
$$

你很自然会问：10.4 明明也用了行为策略 $\beta$，为什么 Algorithm 10.4 没有这个比值？

原因是两节修正的对象不一样。

在 10.3 里，目标策略 $\pi$ 是随机策略。我们要估计的是：

$$
\mathbb E_{A\sim\pi(\cdot\mid S)}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)\cdots
\right].
$$

但样本动作 $a_t$ 来自 $\beta$，所以必须用 $\pi/\beta$ 把“动作分布不一致”校正回来。

在 10.4 里，actor 的梯度没有 $A\sim\mu$ 这一层采样：

$$
\nabla_\theta\mu(s_t)
\left(\nabla_aq(s_t,a,w_t)\right)\big|_{a=\mu(s_t)}.
$$

给定 $s_t$ 后，目标动作就是 $\mu(s_t)$，不是从 $\beta$ 样本动作 $a_t$ 上做 log-policy 更新。因此 actor 不需要动作层面的 importance weight。

critic 这边也要分清两个动作：

| 符号 | 从哪里来 | 用来做什么 |
|---|---|---|
| $a_t$ | 行为策略 $\beta$ | 真正与环境交互，产生 $r_{t+1},s_{t+1}$ |
| $\tilde a_{t+1}=\mu(s_{t+1},\theta_t)$ | 目标策略 $\mu$ | 只放进 TD target，评价目标策略下一步会做什么 |

critic 所需样本是：

$$
(s_t,a_t,r_{t+1},s_{t+1},\tilde a_{t+1}),
\qquad
\tilde a_{t+1}=\mu(s_{t+1},\theta_t).
$$

这里 $\tilde a_{t+1}$ 不拿去和环境交互，只是用来计算：

$$
q(s_{t+1},\mu(s_{t+1},\theta_t),w_t).
$$

所以它是在用行为策略的数据，学习目标策略 $\mu$ 的价值，这就是 off-policy；但它不像 10.3 那样需要对“被采样动作来自哪个概率分布”做 $\pi/\beta$ 校正。

⚠️ 易错点：$a_t$ 和 $\mu(s_t)$ 不一定相同。critic 的当前项是 $q(s_t,a_t,w_t)$，因为环境奖励来自行为动作 $a_t$；actor 的梯度项在 $a=\mu(s_t)$ 处求 $\nabla_a q$，因为 actor 要改的是目标策略输出的动作。

### 8. 一个最小数值例子：行为动作探索，目标动作做 bootstrap

考虑一维连续动作、两个状态 $s_0,s_1$。这个例子故意把两个动作分开：

| 动作 | 数值 | 来自哪里 | 用在哪里 |
|---|---:|---|---|
| 行为动作 $a_t$ | $0.8$ | $a_t\sim\beta(\cdot\mid s_0)$ | 真正和环境交互，产生 $r_{t+1},s_{t+1}$ |
| 当前目标动作 $\mu(s_0,\theta_t)$ | $1.5$ | target policy $\mu$ | actor 在这里看动作梯度 |
| 下一步目标动作 $\mu(s_1,\theta_t)$ | $2.0$ | target policy $\mu$ | critic 的 TD target 用它 bootstrap |

设 actor 很简单，只有一个参数，但不同状态有不同偏置：

$$
\mu(s_0,\theta)=\theta,\qquad
\mu(s_1,\theta)=\theta+0.5.
$$

当前：

$$
\theta_t=1.5,\qquad
\gamma=0.9,\qquad
\alpha_\theta=0.1,\qquad
\alpha_w=0.2.
$$

行为策略为了探索，在 $s_0$ 采到了一个不等于目标动作的动作：

$$
a_t=0.8\sim\beta(\cdot\mid s_0).
$$

执行它以后，环境给出：

$$
r_{t+1}=1,\qquad s_{t+1}=s_1.
$$

critic 当前给出的三个数值是：

$$
q(s_0,a_t,w_t)=q(s_0,0.8,w_t)=1.0,
\qquad
q(s_1,\mu(s_1,\theta_t),w_t)=q(s_1,2.0,w_t)=4.0.
$$

注意第二个价值不是 $q(s_1,a_{t+1},w_t)$，因为下一步没有再从 $\beta$ 采一个动作来做 Sarsa 式 bootstrap；它直接用目标策略的动作：

$$
\tilde a_{t+1}=\mu(s_1,\theta_t)=2.0.
$$

因此 deterministic AC 的 TD error 是：

$$
\begin{aligned}
\delta_t
&=
r_{t+1}
+
\gamma q(s_{t+1},\mu(s_{t+1},\theta_t),w_t)
-
q(s_t,a_t,w_t)\\
&=
1+0.9\times 4.0-1.0\\
&=
3.6.
\end{aligned}
$$

读法：行为动作 $a_t=0.8$ 产生了这条真实经验；但 critic 在评价“目标策略下一步会怎样”时，看的不是行为策略下一步可能采什么，而是 $\mu(s_1,\theta_t)=2.0$。

如果为了最小化计算，设 critic 对当前样本的参数梯度为：

$$
\nabla_w q(s_0,0.8,w_t)=1,
$$

那么 critic 更新就是：

$$
w_{t+1}
=
w_t+\alpha_w\delta_t\nabla_wq(s_0,0.8,w_t)
=
w_t+0.2\times3.6\times1
=
w_t+0.72.
$$

critic 这一步只回答一个问题：

> 对刚才真实发生的这条经验 $(s_0,0.8,r=1,s_1)$，我原来给 $q(s_0,0.8)$ 的估计准不准？

原来 critic 认为：

$$
q(s_0,0.8,w_t)=1.0.
$$

但这条经验告诉它：执行 $0.8$ 以后，立刻拿到奖励 $1$，并且到了 $s_1$。如果从 $s_1$ 开始按目标策略 $\mu$ 继续走，下一步目标动作是 $\mu(s_1)=2.0$，critic 对这个后续价值的估计是 $4.0$。所以“执行 $0.8$ 这件事”的新目标应该是：

$$
\underbrace{1}_{\text{立即奖励}}
+
\underbrace{0.9\times4.0}_{\text{到 }s_1\text{ 后，按目标策略继续的估计价值}}
=4.6.
$$

也就是说，critic 原来把 $q(s_0,0.8)$ 估成 $1.0$，现在发现它应该更接近 $4.6$。因此 $\delta_t=4.6-1.0=3.6>0$，critic 就把 $q(s_0,0.8)$ 往上修。

这里要特别分清：critic 更新 $q(s_0,0.8)$，不是因为 actor 想输出 $0.8$；只是因为 $0.8$ 是这次由行为策略 $\beta$ 真正拿去和环境交互的动作，所以我们有它的真实后果。

现在再看 actor。actor 回答的是另一个问题：

> 如果目标策略自己在 $s_0$ 选动作，它现在会输出 $1.5$；那这个输出应该往左调还是往右调？

actor 不沿着行为动作 $a_t=0.8$ 更新，也不试图“提高 $0.8$ 这个动作的概率”。原因是 deterministic actor 根本不维护“动作概率表”；它只维护一个函数 $\mu(s,\theta)$。当前在 $s_0$，这个函数输出的是：

$$
\mu(s_0,\theta_t)=1.5
$$

所以 actor 要改的是 $\mu(s_0,\theta)$ 这个输出值，而不是刚才 $\beta$ 偶然采到的 $0.8$。

怎么知道 $1.5$ 该往哪边改？让 critic 在 $a=1.5$ 附近看一眼“动作价值曲线”的斜率。设在 $s_0$ 附近，critic 对动作的局部形状可以写成：

$$
q(s_0,a,w_t)=-(a-2)^2+5.
$$

这个局部 critic 的意思是：在 $s_0$，动作越接近 $a=2$，价值越高。

因为在 $s_0$：

$$
\nabla_\theta\mu(s_0,\theta)=1,
$$

且：

$$
\nabla_aq(s,a,w)
=
-2(a-2).
$$

在当前目标动作 $a=\mu(s_0,\theta_t)=1.5$ 处：

$$
\nabla_aq(s_0,a,w_t)\big|_{a=1.5}
=
-2(1.5-2)
=1.
$$

actor 更新：

$$
\begin{aligned}
\theta_{t+1}
&=
\theta_t
+
\alpha_\theta
\nabla_\theta\mu(s_0,\theta_t)
\nabla_aq(s_0,a,w_t)\big|_{a=\mu(s_0)}\\
&=
1.5+0.1\times1\times1\\
&=
1.6.
\end{aligned}
$$

读法：在 $a=1.5$ 这里，$\nabla_a q=1>0$，说明动作稍微变大，critic 估计的价值会上升。所以 actor 把目标策略输出从 $1.5$ 推到 $1.6$。

这就是这一步最容易混的地方：

| 问题 | 用哪个动作 | 为什么 |
|---|---|---|
| critic 要修正哪个 $q$ 值？ | $a_t=0.8$ | 因为这是行为策略真实执行的动作，我们观察到了它导致的 $r,s_1$ |
| TD target 的下一步动作是谁？ | $\mu(s_1)=2.0$ | 因为 critic 要评价目标策略 $\mu$，所以 bootstrap 时假设下一步按 $\mu$ 走 |
| actor 要移动哪个动作？ | $\mu(s_0)=1.5$ | 因为 actor 的参数控制的是目标策略输出，不是行为策略采样出来的 $0.8$ |

一句话：$0.8$ 用来产生经验、训练 critic；$2.0$ 用来构造 critic 的下一步目标；$1.5$ 才是 actor 当前真正要移动的策略输出。

把整步串起来就是：

| 部分 | 用的动作 | 数值结果 | 含义 |
|---|---|---:|---|
| 环境交互 | $a_t=0.8\sim\beta$ | 得到 $r=1,s_1$ | 行为策略负责探索 |
| critic TD target | $\mu(s_1)=2.0$ | $1+0.9\times4.0=4.6$ | 下一步按目标策略评价 |
| critic TD error | 当前项 $q(s_0,0.8)=1.0$ | $\delta_t=3.6$ | 当前 critic 低估了这条经验 |
| actor 梯度 | $\mu(s_0)=1.5$ | $\theta:1.5\to1.6$ | actor 直接移动目标动作 |

所以这张图里的 deterministic AC 思想是：行为策略 $\beta$ 负责产生真实样本；critic 的 TD error 用“行为动作的当前项 + 目标动作的下一步 bootstrap”；actor 则完全不调动作概率，而是在 $\mu(s_t)$ 处沿着 $\nabla_a q$ 直接移动目标动作。

### 9. 和前面三节的闭环

10.1 到 10.4 可以看成 actor 更新形态一步步变化：

| 小节 | actor 是什么 | critic 给什么信号 | 更新关键词 |
|---|---|---|---|
| 10.1 QAC | 随机策略 $\pi$ | $q(s_t,a_t,w_t)$ | 增强高价值动作概率 |
| 10.2 A2C | 随机策略 $\pi$ | advantage / TD error $\delta_t$ | 相对平均水平好才增强 |
| 10.3 Off-policy AC | 随机策略 $\pi$，数据来自 $\beta$ | $c_t\delta_t$ | 用 $\pi/\beta$ 校正行为策略 |
| 10.4 Deterministic AC | 确定性策略 $\mu$ | $\nabla_a q(s,a,w)$ | 直接移动动作输出 |

这一节的核心可以压缩成：

$$
\boxed{
\text{stochastic actor 调动作概率；deterministic actor 调动作本身}
}
$$

---

## 10.5 Summary（本章小结）

**要解决的问题**：本节不是引入新算法，而是把 10.1-10.4 的 actor-critic 主线收束起来：第 9 章只有 actor 的策略梯度还缺一个价值估计器，第 10 章逐步把 critic 加进来，并从 on-policy、advantage、off-policy 一直走到 deterministic policy gradient。

### 1. 本章真正合并了哪两条线？

第 8 章讲的是 critic 的能力：

$$
q_\pi(s,a)
\approx
q(s,a,w),
\qquad
v_\pi(s)
\approx
v(s,w).
$$

第 9 章讲的是 actor 的能力：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
q_t(s_t,a_t).
$$

第 10 章把二者合在一起：

$$
\boxed{
\text{actor 负责改策略}
\quad+\quad
\text{critic 负责给 actor 一个价值信号}
}
$$

读法：actor 不再单靠完整回报来判断“刚才动作好不好”，而是让 critic 在线学习一个价值函数，再把这个价值函数变成策略更新的信号。

这也是 actor-critic 的核心妥协：

| 方法 | 好处 | 代价 |
|---|---|---|
| REINFORCE | Monte Carlo 回报通常是无偏估计 | 方差大，常要等 episode 结束 |
| actor-critic | TD 信号可在线更新，方差通常更低 | critic 近似会引入偏差 |

所以 actor-critic 不是“比 REINFORCE 多一个模块”这么简单；它是把策略梯度中的价值估计问题，交给 TD learning 和函数近似来处理。

### 2. 10.1：QAC，把 Monte Carlo 回报换成 TD critic

10.1 的 QAC，Q actor-critic，做的替换很直接：

$$
q_t(s_t,a_t)
\quad\Longleftarrow\quad
q(s_t,a_t,w_t).
$$

于是 actor 更新为：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t)
q(s_t,a_t,w_t).
$$

critic 则用 Sarsa with function approximation 学动作价值：

$$
\delta_t
=
r_{t+1}
+
\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t),
$$

$$
w_{t+1}
=
w_t
+
\alpha_w\delta_t\nabla_wq(s_t,a_t,w_t).
$$

读法：actor 问“这个动作值得增加概率吗”，critic 用 $q(s_t,a_t,w_t)$ 回答；critic 自己则用 TD error 不断修正。

⚠️ 易错点：QAC 里的 $q(s,a,w)$ 是近似函数，不是真实 $q_\pi(s,a)$。它能指导 actor，但它本身也在学习，所以 actor 和 critic 是互相追着动的两个系统。

### 3. 10.2：A2C，把“绝对好”改成“比预期更好”

QAC 用的是动作价值 $q_\pi(S,A)$。但策略梯度有一个重要性质：加减一个只依赖状态的 baseline 基线，不改变梯度期望。

也就是：

$$
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)b(S)
\right]
=0.
$$

所以可以把 $q_\pi(S,A)$ 换成：

$$
q_\pi(S,A)-b(S).
$$

最自然的选择是：

$$
b(S)=v_\pi(S),
$$

这时得到 advantage function 优势函数：

$$
a_\pi(S,A)
=
q_\pi(S,A)-v_\pi(S).
$$

读法：A2C 不再问“这个动作的总价值高不高”，而是问“这个动作相对于当前状态的平均水平，是更好还是更差”。

实际算法中，A2C 用 TD error 近似 advantage：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t).
$$

actor 更新为：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\delta_t
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t).
$$

这里 $\delta_t>0$ 表示“实际看到的结果比 critic 对 $s_t$ 的预期更好”，于是增加 $a_t$ 的概率；$\delta_t<0$ 表示“比预期差”，于是降低这类动作的概率。

### 4. 10.3：Off-policy AC，样本来自 $\beta$，目标优化 $\pi$

前两节默认数据来自当前 actor 自己，也就是 on-policy。10.3 放宽这一点：行为策略 behavior policy $\beta$ 负责采样，目标策略 target policy $\pi$ 负责被优化。

问题是：策略梯度本来需要的是

$$
\mathbb E_{A\sim\pi(\cdot\mid S)}
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)\cdots
\right],
$$

但现在动作来自

$$
A\sim\beta(\cdot\mid S).
$$

因此要用 importance sampling 重要性采样校正：

$$
c_t
=
\frac{\pi(a_t\mid s_t,\theta_t)}
{\beta(a_t\mid s_t)}.
$$

off-policy actor-critic 的更新形状就变成：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
c_t\delta_t
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t),
$$

$$
w_{t+1}
=
w_t
+
\alpha_w
c_t\delta_t
\nabla_wv(s_t,w_t).
$$

读法：$c_t$ 不是新的奖励，也不是新的 advantage；它只是告诉我们“这条由 $\beta$ 采来的样本，在 $\pi$ 的世界里应该占多大权重”。

⚠️ 易错点：如果 $\beta(a_t\mid s_t)$ 很小而 $\pi(a_t\mid s_t)$ 不小，$c_t$ 会很大，更新方差也会变大。这是 off-policy policy gradient 经常需要小心处理的地方。

### 5. 10.4：Deterministic AC，从调概率变成调动作本身

10.1-10.3 都使用 stochastic policy 随机策略：

$$
\pi(a\mid s,\theta).
$$

actor 的更新对象是动作概率，所以会出现：

$$
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t).
$$

10.4 改成 deterministic policy 确定性策略：

$$
a=\mu(s,\theta).
$$

这时 actor 不再问“怎样提高某个动作的概率”，而是问“怎样把当前输出动作往更高价值方向移动”。因此梯度变成链式法则：

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\nabla_\theta\mu(S,\theta)
\left(\nabla_a q(S,a,w)\right)\big|_{a=\mu(S,\theta)}
\right].
$$

对应更新为：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\nabla_\theta\mu(s_t,\theta_t)
\left(\nabla_a q(s_t,a,w_t)\right)\big|_{a=\mu(s_t,\theta_t)}.
$$

读法：critic 给出“动作往哪个方向变，价值会上升”，actor 再通过 $\nabla_\theta\mu$ 把这个动作空间里的方向传回参数空间。

这一步尤其适合 continuous action space 连续动作空间。因为在连续动作中，对所有动作枚举概率或者求 $\arg\max_a q(s,a)$ 通常很困难；而 deterministic actor 可以直接输出一个动作。

### 6. 四种算法放在一张表里

本章四节不是四个孤立算法，而是一条逐步替换的链：

| 小节 | 算法 | actor 类型 | critic 学什么 | actor 用什么信号 |
|---|---|---|---|---|
| 10.1 | QAC | 随机策略 $\pi(a\mid s,\theta)$ | $q(s,a,w)$ | 动作价值 $q(s_t,a_t,w_t)$ |
| 10.2 | A2C | 随机策略 $\pi(a\mid s,\theta)$ | $v(s,w)$ | TD error / advantage $\delta_t$ |
| 10.3 | Off-policy AC | 目标策略 $\pi$，行为策略 $\beta$ | $v(s,w)$ | $c_t\delta_t$ |
| 10.4 | Deterministic AC | 确定性策略 $\mu(s,\theta)$ | $q(s,a,w)$ | 动作梯度 $\nabla_aq(s,a,w)$ |

如果只记一句话，可以这样记：

$$
\boxed{
\text{QAC 用价值，A2C 用优势，off-policy AC 加重要性权重，deterministic AC 直接沿动作价值梯度移动动作。}
}
$$

### 7. 和现代强化学习算法的关系

原书最后提醒：policy gradient 和 actor-critic 是现代强化学习里非常核心的基础。许多更高级算法都可以看成在本章基础上继续加稳定化技巧、约束或新的建模视角。

比如：

| 算法 / 方向 | 可以从本章哪条线理解 |
|---|---|
| PPO | 从 stochastic policy gradient / actor-critic 出发，限制策略每次更新幅度 |
| TRPO | 也是限制策略更新幅度，但用 trust region 的优化思想 |
| SAC | actor-critic 加上 entropy 最大化，让策略既追求回报又保持探索 |
| TD3 | deterministic actor-critic / DDPG 路线上的改进，重点缓解 critic 过估计 |
| multi-agent RL | 把单智能体的 actor-critic 扩展到多个 agent |
| model-based RL | 用经验样本学习环境模型，再借模型辅助决策 |
| distributional RL | 不只估计期望回报，而是估计回报分布 |

这些内容本书不展开，但第 10 章已经给了读它们的最小骨架：看到 actor，就问它怎样参数化策略；看到 critic，就问它估计 $v$、$q$ 还是 advantage；看到 off-policy，就问行为策略和目标策略如何校正；看到 deterministic，就问动作梯度如何传回 actor 参数。

### 8. 章末闭环

从第 8 章到第 10 章，主线可以压缩成三步：

$$
\boxed{
\text{第 8 章：学会近似价值}
\quad\Longrightarrow\quad
\text{第 9 章：学会直接优化策略}
\quad\Longrightarrow\quad
\text{第 10 章：用价值估计辅助策略优化}
}
$$

到这里，我们已经有了深度强化学习很多算法的共同语言：

- policy 策略可以是表格、随机函数、确定性函数。
- value 价值可以是表格，也可以是函数近似。
- update 更新可以来自 Monte Carlo return、TD error、advantage、importance sampling 或 action gradient。
- actor-critic 的关键不是名字，而是 actor 和 critic 两个学习过程如何互相提供信号、互相影响稳定性。

⚠️ 最后一个易错点：critic 并不是“裁判给分后就结束”。critic 的估计会影响 actor 的方向；actor 的变化又会改变 critic 要估计的目标分布。因此 actor-critic 的难点常常不是某一个公式，而是两个近似学习器同时更新时的稳定性。

---

## 10.6 Q&A（问答）

**要解决的问题**：本节用四个问题把第 10 章最容易混淆的概念收口：actor-critic 到底是不是 policy gradient，baseline 为什么能加，importance sampling 是否只属于策略梯度，以及 deterministic policy gradient 为什么天然可以 off-policy。

### Q1. Actor-critic 和 policy gradient methods 是什么关系？

原书答案的核心是：

$$
\boxed{
\text{actor-critic methods are actually policy gradient methods}
}
$$

也就是说，actor-critic 本质上仍然是 policy gradient methods 策略梯度方法。有时二者甚至会被交替使用。

为什么？因为 actor 的更新仍然来自策略梯度公式：

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)
q_\pi(S,A)
\right].
$$

这个式子里真正麻烦的是 $q_\pi(S,A)$：我们不知道真实 action value 动作价值，所以必须估计它。

不同方法的分叉点就在这里：

| 方法 | 怎么估计 $q_\pi(S,A)$ 或相关价值信号 |
|---|---|
| REINFORCE | 用 Monte Carlo return 蒙特卡洛回报 |
| QAC | 用 TD learning 学到的 $q(s,a,w)$ |
| A2C | 用 TD error $\delta_t$ 近似 advantage |
| deterministic AC | 用 critic 的 $q(s,a,w)$ 提供动作梯度 |

所以 actor-critic 不是另一类和 policy gradient 平行的方法，而是 policy gradient 的一种实现结构：

$$
\boxed{
\text{policy gradient 给目标方向；critic 用 TD + function approximation 提供价值估计。}
}
$$

名字里的 actor-critic 强调的是算法结构：

- actor 演员：更新策略参数 $\theta$。
- critic 评论家：更新价值参数 $w$。

读法：policy gradient 是“要沿什么数学方向更新策略”；actor-critic 是“用一个 actor 和一个 critic 怎样在算法上实现这个方向”。

⚠️ 易错点：不是所有 policy gradient 都叫 actor-critic。只有当动作价值、状态价值或优势信号由 TD learning + value function approximation 这类 critic 模块在线估计时，才通常称为 actor-critic。

### Q2. 为什么要给 actor-critic 引入额外 baseline？

baseline 基线最重要的性质是：只要它不依赖动作 $A$，把它从动作价值里减掉，不会改变策略梯度的期望。

也就是：

$$
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)b(S)
\right]
=0.
$$

因此：

$$
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)q_\pi(S,A)
\right]
=
\mathbb E
\left[
\nabla_\theta\ln\pi(A\mid S,\theta)
\left(q_\pi(S,A)-b(S)\right)
\right].
$$

这说明 baseline 不改变“平均更新方向”。那为什么要加？因为它可以降低 estimation variance 估计方差。

最常用的 baseline 是：

$$
b(S)=v_\pi(S).
$$

于是得到 advantage 优势函数：

$$
a_\pi(S,A)
=
q_\pi(S,A)-v_\pi(S).
$$

它的读法很关键：

$$
\boxed{
\text{advantage 不是问这个动作好不好，而是问它比当前状态的平均动作好多少。}
}
$$

一个最小直觉例子：

| 状态 $S$ | 动作 $A$ | $q_\pi(S,A)$ | $v_\pi(S)$ | $a_\pi(S,A)$ | actor 应该怎么理解 |
|---|---:|---:|---:|---:|---|
| 同一个状态 | $a_1$ | 100 | 95 | 5 | 比平均好，应该提高概率 |
| 同一个状态 | $a_2$ | 90 | 95 | -5 | 比平均差，应该降低概率 |

如果只看 $q_\pi(S,A)$，两个动作的价值都是正的，容易都被“奖励”；但看 advantage，就能分出谁比当前状态下的正常水平更好。

所以 A2C 的 actor 更新写成：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_\theta
\delta_t
\nabla_\theta\ln\pi(a_t\mid s_t,\theta_t),
$$

其中：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
$$

被用来近似 advantage。

⚠️ 易错点：baseline 不是为了改变最优策略，也不是为了加入新的偏好；它主要是为了让梯度估计更稳定、方差更小。

### Q3. Importance sampling 只能用在 policy-based algorithms 吗？

答案是否定的。importance sampling 重要性采样是一个通用的期望估计技巧，不只属于 policy gradient。

它解决的问题是：

$$
\text{目标期望属于分布 }p,
\quad
\text{但样本来自分布 }q.
$$

如果要估计：

$$
\mathbb E_{X\sim p}[f(X)],
$$

但手里只有：

$$
X\sim q,
$$

则可以写成：

$$
\mathbb E_{X\sim p}[f(X)]
=
\mathbb E_{X\sim q}
\left[
\frac{p(X)}{q(X)}f(X)
\right].
$$

这就是 importance sampling 的基本形状。

强化学习里到处都是 expectation 期望，所以它自然能用在很多地方：

| 场景 | 被估计的东西为什么是期望 |
|---|---|
| state value 状态价值 $v_\pi(s)$ | 它是从状态 $s$ 出发的 expected return |
| action value 动作价值 $q_\pi(s,a)$ | 它是从状态动作 $(s,a)$ 出发的 expected return |
| policy gradient 策略梯度 | 真实梯度本身写成对状态、动作的期望 |

因此 importance sampling 可以用于 value-based methods 价值型方法，也可以用于 policy-based methods 策略型方法。

10.3 里它的作用是把来自 behavior policy 行为策略 $\beta$ 的动作样本，校正成 target policy 目标策略 $\pi$ 下的期望：

$$
c_t
=
\frac{\pi(a_t\mid s_t,\theta_t)}
{\beta(a_t\mid s_t)}.
$$

原书还特别指出：Algorithm 10.3 的 value-based component，也就是 critic 更新部分，也用了这个思想。

⚠️ 易错点：importance sampling 的本质不是“策略梯度专用修正项”，而是“换分布估计期望”的通用工具。只要有“目标分布”和“采样分布”不一致，就可能需要它。

### Q4. 为什么 deterministic policy gradient method 是 off-policy？

随机策略的 actor 更新含有动作随机变量：

$$
\nabla_\theta\ln\pi(A\mid S,\theta),
\qquad
A\sim\pi(\cdot\mid S).
$$

因此如果动作其实来自 $\beta$，就会出现动作分布不一致，需要 importance sampling。

但 deterministic policy gradient 确定性策略梯度不一样。确定性 actor 是：

$$
a=\mu(s,\theta).
$$

给定状态 $s$ 后，目标动作不是随机采样出来的，而是直接算出来的。对应梯度为：

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\nabla_\theta\mu(S,\theta)
\left(\nabla_aq(S,a,w)\right)\big|_{a=\mu(S,\theta)}
\right].
$$

注意这里没有：

$$
A\sim\mu(\cdot\mid S),
$$

也没有：

$$
\nabla_\theta\ln\pi(A\mid S,\theta).
$$

因此估计 deterministic policy gradient 时，不需要让样本动作来自目标策略 $\mu$。实际与环境交互的动作可以来自任意 behavior policy 行为策略，比如：

$$
a_t\sim\beta(\cdot\mid s_t),
$$

或者工程里常见的：

$$
a_t=\mu(s_t,\theta_t)+\text{exploration noise}.
$$

actor 更新时真正用的是：

$$
\nabla_\theta\mu(s_t,\theta_t)
\left(\nabla_aq(s_t,a,w_t)\right)\big|_{a=\mu(s_t,\theta_t)}.
$$

读法：行为策略 $\beta$ 负责“带我去收集状态和奖励样本”；目标策略 $\mu$ 负责“在这些状态上，我应该把动作输出往哪个方向改”。

所以 deterministic policy gradient 可以 off-policy 的原因是：

$$
\boxed{
\text{它的真实梯度不含动作随机变量；给定状态后，目标动作由 }\mu(s,\theta)\text{ 直接确定。}
}
$$

⚠️ 易错点：这不表示 deterministic actor-critic 完全不需要探索。它仍然需要行为策略或噪声去探索环境；只是 actor 梯度本身不要求动作样本来自目标策略。

### 本节闭环

第 10 章最后这组 Q&A 可以整理成四句话：

- actor-critic 是 policy gradient 的一种结构化实现，因为 actor 仍然沿策略梯度更新。
- baseline 不改变策略梯度期望，但能降低方差，于是得到 A2C。
- importance sampling 是通用的换分布估计期望技巧，value-based 和 policy-based 方法都可以用。
- deterministic policy gradient 的梯度不含动作随机变量，所以它可以使用 off-policy 数据。

这也把本章的核心骨架彻底闭上了：actor 负责优化策略，critic 负责提供价值信号；baseline、importance sampling、deterministic gradient 都是在解决“这个价值信号怎样更稳定、更通用、更适合连续动作空间”的问题。

---

## 我的疑问与解答

暂无。

---

## 脉络总结 / 要点速记

10.1：最简单的 actor-critic（QAC）来自对 REINFORCE 的替换：把 Monte Carlo 回报 $q_t(s_t,a_t)$ 换成 TD 学出来的 $q(s_t,a_t,w_t)$。actor 用策略梯度更新 $\theta$，critic 用 Sarsa 更新 $w$。

10.2：A2C 引入 baseline。减去只依赖状态的 $b(S)$ 不改变策略梯度期望，但能降低方差。实用选择是 $b(S)=v_\pi(S)$，于是 actor 用 advantage $q_\pi(S,A)-v_\pi(S)$ 更新；实际算法里用 TD error $\delta_t=r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w_t)$ 来近似 advantage。

10.3：off-policy actor-critic 允许样本来自行为策略 $\beta$，但目标仍是优化 $\pi$。核心工具是 importance sampling，把更新乘上 $\frac{\pi(a_t\mid s_t,\theta_t)}{\beta(a_t\mid s_t)}$，用来校正“样本来自 $\beta$、目标期望属于 $\pi$”之间的分布差异。

10.4：deterministic actor-critic 把 actor 从随机策略 $\pi(a\mid s,\theta)$ 改成确定性策略 $\mu(s,\theta)$。随机 actor 调整动作概率；确定性 actor 用 $\nabla_\theta\mu(s)\nabla_a q(s,a,w)$ 直接移动动作输出。它天然适合连续动作空间，也常用行为策略 $\beta$ 加噪声探索、目标策略 $\mu$ 做价值评估。

10.5：本章总结把四条路线串起来：QAC 用 TD critic 替换 REINFORCE 的 Monte Carlo 回报；A2C 用 baseline / advantage 降低方差；off-policy actor-critic 用 importance sampling 处理样本来自 $\beta$ 而目标是 $\pi$ 的差异；deterministic actor-critic 把策略从“动作概率”改成“直接输出动作”，用 $\nabla_a q$ 指导动作移动。第 10 章因此把第 8 章的价值函数近似和第 9 章的策略梯度真正合到了一起。

10.6：Q&A 收束四个易混点：actor-critic 本质上仍是 policy gradient，只是用 TD critic 估计价值信号；baseline 不改变梯度期望但能降方差；importance sampling 是通用的换分布估计期望技巧，不只属于策略梯度；deterministic policy gradient 的梯度不含动作随机变量，所以能用 off-policy 数据学习。
