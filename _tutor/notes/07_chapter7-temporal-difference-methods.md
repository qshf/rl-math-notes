# 第 7 章 时序差分方法（Temporal-Difference Methods）- 7.1-7.7（全章完）

> 原书 p136-160 · 学习日期 2026-06-14 · 当前涵盖 7.1-7.7（全章完）

## 本章在全书的位置（先读这段）

第 5 章的 Monte Carlo methods 蒙特卡洛方法已经把 model-free 无模型学习带了出来：不知道环境模型也没关系，只要能采样，就可以用样本回报 $G_t$ 估计价值。

第 6 章的 stochastic approximation 随机近似又补上了数学语言：当目标量是某个方程的根，或者某个期望条件定义的值，我们可以用带噪声的样本一步步逼近它。

第 7 章把这两条线合在一起，进入 temporal-difference learning 时序差分学习：

> TD 的核心想法：不等完整 episode 结束，也不等完整回报 $G_t$ 出来；每看到一步经验 $(s_t,r_{t+1},s_{t+1})$，就用“一步奖励 + 下个状态的当前价值估计”来修正当前状态的价值。

这比 MC 更在线，也比第 4 章的动态规划更现实：它不需要已知 $p(s'|s,a)$ 和 $p(r|s,a)$，只需要经验样本。

---

## 7.1 TD learning of state values（状态值的 TD 学习）

**要解决的问题**：给定一个 policy 策略 $\pi$，我们想在不知道环境模型的情况下，仅靠按 $\pi$ 生成的经验样本，在线估计每个 state value 状态值 $v_\pi(s)$。

本节的 TD learning 特指最经典的一步状态值估计算法。广义上，本章的 Sarsa、$n$-step Sarsa、Q-learning 都属于 TD learning；但 7.1 先讲最基础的 state-value prediction 状态值预测。

### 1. TD 状态值更新算法

假设经验轨迹来自策略 $\pi$：

$$
(s_0,r_1,s_1,\ldots,s_t,r_{t+1},s_{t+1},\ldots).
$$

TD 对刚访问的状态 $s_t$ 做更新：

$$
v _ {t + 1} \left(s _ {t}\right)
=
v _ {t} \left(s _ {t}\right)
- \alpha_ {t} \left(s _ {t}\right)
\left[
v _ {t} \left(s _ {t}\right)
-
\left(r _ {t + 1} + \gamma v _ {t} \left(s _ {t + 1}\right)\right)
\right]. \tag {7.1}
$$

没有访问到的状态保持不变：

$$
v _ {t + 1} (s) = v _ {t} (s), \quad \text{for all } s \neq s _ {t}. \tag {7.2}
$$

读法：在时刻 $t$，我们只更新当前刚看到的状态 $s_t$；其他状态的估计值不动。

把式 (7.1) 改写成更直观的“朝目标移动”形式：

$$
v_{t+1}(s_t)
=
v_t(s_t)
+\alpha_t(s_t)
\left[
\underbrace{r_{t+1}+\gamma v_t(s_{t+1})}_{\text{TD target 目标}}
-
\underbrace{v_t(s_t)}_{\text{当前估计}}
\right].
$$

读法：新估计 = 旧估计 + 步长 $\times$“目标值和旧估计的差”。

这里的目标值不是完整回报 $G_t$，而是：

$$
\bar v_t \doteq r_{t+1}+\gamma v_t(s_{t+1}).
$$

这就是 one-step TD target 一步 TD 目标。

⚠️ **式 (7.2) 虽然常被省略，但数学上不能忘。** TD 每次只改一个被访问状态；如果你以为所有 $s$ 都同步更新，就会把它误读成第 4 章的动态规划 backup。

### 2. 为什么这个算法来自 Bellman equation

状态值定义是：

$$
v _ {\pi} (s)
=
\mathbb {E} \left[
R _ {t + 1} + \gamma G _ {t + 1}
\mid S _ {t} = s
\right], \quad s \in \mathcal {S}. \tag {7.3}
$$

由于从 $S_{t+1}$ 开始的未来回报期望就是 $v_\pi(S_{t+1})$，所以它可以写成 Bellman expectation equation 贝尔曼期望方程：

$$
v _ {\pi} (s)
=
\mathbb {E} \left[
R _ {t + 1}
+ \gamma v _ {\pi} \left(S _ {t + 1}\right)
\mid S _ {t} = s
\right], \quad s \in \mathcal {S}. \tag {7.4}
$$

拆开读：

$$
\underbrace{v_\pi(s)}_{\text{当前状态真价值}}
=
\mathbb E\left[
\underbrace{R_{t+1}}_{\text{下一步奖励}}
+
\gamma
\underbrace{v_\pi(S_{t+1})}_{\text{下个状态真价值}}
\mid S_t=s
\right].
$$

如果模型已知，我们可以像第 2-4 章一样直接算这个期望。现在模型未知，所以无法直接算右边的期望；但我们能采到一个样本：

$$
r_{t+1},\quad s_{t+1}.
$$

于是用样本量：

$$
r_{t+1}+\gamma v_t(s_{t+1})
$$

去近似期望里的：

$$
R_{t+1}+\gamma v_\pi(S_{t+1}).
$$

这一步就是 TD 的灵魂：用一次真实转移样本，加上当前对下个状态价值的估计，组成一个带噪声的 Bellman target。

### 3. Box 7.1：TD 是 Robbins-Monro 的具体化

第 6 章 Robbins-Monro algorithm 的形状是：要求解

$$
g(w^*)=0,
$$

但只能看到带噪声的 $\tilde g(w)$，于是迭代：

$$
w_{k+1}=w_k-\alpha_k\tilde g(w_k).
$$

现在 Bellman equation 也可以写成“求根问题”。对当前状态 $s_t$，定义：

$$
g (v _ {\pi} (s _ {t}))
\doteq
v _ {\pi} (s _ {t})
-
\mathbb {E} \big [
R _ {t + 1}
+ \gamma v _ {\pi} (S _ {t + 1})
\mid S _ {t} = s _ {t}
\big ].
$$

Bellman equation 成立就等价于：

$$
g(v_\pi(s_t))=0.
$$

但我们不知道这个期望，只能用样本构造 noisy observation 带噪观测：

$$
\tilde {g} \left(v _ {\pi} \left(s _ {t}\right)\right)
=
v _ {\pi} \left(s _ {t}\right)
-
\left[
r _ {t + 1}
+ \gamma v _ {\pi} \left(s _ {t + 1}\right)
\right].
$$

把它看成“真函数 + 噪声”：

$$
\tilde g
=
\underbrace{
v_\pi(s_t)
-
\mathbb E[
R_{t+1}+\gamma v_\pi(S_{t+1})
\mid S_t=s_t]
}_{g(v_\pi(s_t))}
+
\underbrace{
\mathbb E[
R_{t+1}+\gamma v_\pi(S_{t+1})
\mid S_t=s_t]
-
[r_{t+1}+\gamma v_\pi(s_{t+1})]
}_{\eta}.
$$

读法：$\tilde g$ 是用单个样本替代期望后得到的随机版本；$\eta$ 就是这一次采样带来的噪声。

按 Robbins-Monro 更新：

$$
v _ {t + 1} (s _ {t})
=
v _ {t} (s _ {t})
-
\alpha_ {t} (s _ {t})
\left(
v _ {t} (s _ {t})
-
\left[
r _ {t + 1}
+ \gamma v _ {\pi} \left(s _ {t + 1}\right)
\right]
\right). \tag {7.5}
$$

式 (7.5) 里还有一个理想量 $v_\pi(s_{t+1})$。真正的 TD 算法把它换成当前估计 $v_t(s_{t+1})$：

$$
v_\pi(s_{t+1})
\quad\leadsto\quad
v_t(s_{t+1}).
$$

于是得到式 (7.1)。

⚠️ **这里有一个很关键的“大胆替换”。** Robbins-Monro 推导中右边本来用了真值 $v_\pi(s_{t+1})$；TD 实际算法用估计值 $v_t(s_{t+1})$。这叫 bootstrapping 自举：用当前估计去更新当前估计。这个替换是否仍然收敛，就是后面 Theorem 7.1 要回答的问题。

### 4. TD target 和 TD error

原书把式 (7.1) 标成：

$$
\underbrace {v _ {t + 1} \left(s _ {t}\right)} _ {\text {new estimate}}
=
\underbrace {v _ {t} \left(s _ {t}\right)} _ {\text {current estimate}}
- \alpha_ {t} \left(s _ {t}\right)
\left[
\overbrace {
v _ {t} \left(s _ {t}\right)
-
\left(
\underbrace {
r _ {t + 1}
+ \gamma v _ {t} \left(s _ {t + 1}\right)
} _ {\text {TD target } \bar {v} _ {t}}
\right)
} ^ {\text {TD error } \delta_ {t}}
\right]. \tag {7.6}
$$

TD target 定义为：

$$
\bar {v} _ {t} \doteq r _ {t + 1} + \gamma v _ {t} (s _ {t + 1}).
$$

TD error 时序差分误差定义为：

$$
\delta_ {t}
\doteq
v_t(s_t)-\bar v_t
=
v _ {t} (s _ {t})
-
\left(
r _ {t + 1}
+ \gamma v _ {t} (s _ {t + 1})
\right).
$$

很多教材把 TD error 写成相反号：

$$
\delta_t^{\text{common}}
=
r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t).
$$

本书这里采用的是“当前估计 - TD target”的符号，所以更新式写成减去 $\alpha_t\delta_t$。两种写法等价，关键是别混符号。

读法：如果 $v_t(s_t)$ 比 TD target 大，说明当前估计偏高，更新会降低它；如果 $v_t(s_t)$ 比 TD target 小，更新会提高它。

### 5. 为什么 $\bar v_t$ 叫 target

从式 (7.6) 出发：

$$
v_{t+1}(s_t)
=
v_t(s_t)-\alpha_t(s_t)[v_t(s_t)-\bar v_t].
$$

两边减去 $\bar v_t$：

$$
v _ {t + 1} (s _ {t}) - \bar {v} _ {t}
=
\left[
v _ {t} (s _ {t}) - \bar {v} _ {t}
\right]
-
\alpha_ {t} (s _ {t})
\left[
v _ {t} (s _ {t}) - \bar {v} _ {t}
\right].
$$

提公因子：

$$
v _ {t + 1} (s _ {t}) - \bar {v} _ {t}
=
\left[
1-\alpha_t(s_t)
\right]
\left[
v _ {t} (s _ {t}) - \bar {v} _ {t}
\right].
$$

取绝对值：

$$
\left|
v _ {t + 1} (s _ {t}) - \bar {v} _ {t}
\right|
=
\left|
1-\alpha_t(s_t)
\right|
\left|
v _ {t} (s _ {t}) - \bar {v} _ {t}
\right|.
$$

当 $0<\alpha_t(s_t)<1$ 时：

$$
\left|
v _ {t + 1} (s _ {t}) - \bar {v} _ {t}
\right|
<
\left|
v _ {t} (s _ {t}) - \bar {v} _ {t}
\right|.
$$

读法：更新后，$v(s_t)$ 离 $\bar v_t$ 更近了。所以 $\bar v_t$ 确实是算法在这一步想靠近的 target 目标。

### 6. 一个两状态小例子

设只有两个状态 $A,B$，折扣因子：

$$
\gamma=0.9.
$$

当前估计为：

$$
v_0(A)=0,\qquad v_0(B)=2.
$$

学习率：

$$
\alpha=0.5.
$$

现在采到一条转移：

$$
s_t=A,\qquad r_{t+1}=1,\qquad s_{t+1}=B.
$$

TD target 是：

$$
\bar v_t
=
r_{t+1}+\gamma v_t(B)
=
1+0.9\times2
=
2.8.
$$

本书符号下的 TD error 是：

$$
\delta_t
=
v_t(A)-\bar v_t
=
0-2.8
=
-2.8.
$$

更新 $A$：

$$
v_{t+1}(A)
=
v_t(A)-\alpha\delta_t
=
0-0.5\times(-2.8)
=
1.4.
$$

$B$ 没被访问，所以：

$$
v_{t+1}(B)=v_t(B)=2.
$$

| 步骤 | 访问状态 | 奖励 | 下个状态 | TD target $\bar v_t$ | 更新前 $v(A)$ | 更新后 $v(A)$ | $v(B)$ |
|---:|---|---:|---|---:|---:|---:|---:|
| 0 | A | 1 | B | 2.8 | 0 | 1.4 | 2 |

这个小例子能看出 TD 和 MC 的差别：TD 没有等整条轨迹结束，也没有真的看到从 $A$ 出发的完整回报；它只看一步奖励 $1$，再借用当前对 $B$ 的估计 $2$，就立刻把 $A$ 往 $2.8$ 的方向推了一半。

### 7. TD error 的两层含义

第一层，temporal-difference 时序差分：它比较的是两个时间步之间的估计关系：

$$
v_t(s_t)
\quad\text{vs.}\quad
r_{t+1}+\gamma v_t(s_{t+1}).
$$

前者是“现在对当前状态的估计”，后者是“看到下一步后，对当前状态应有价值的一步估计”。

第二层，估计误差：如果 $v_t=v_\pi$ 已经完全准确，那么 TD error 的条件期望为 0：

$$
\begin{aligned}
\mathbb {E} [ \delta_ {t} \mid S _ {t} = s _ {t} ]
&=
\mathbb {E} \big [
v _ {\pi} (S _ {t})
-
(R _ {t + 1} + \gamma v _ {\pi} (S _ {t + 1}))
\mid S _ {t} = s _ {t}
\big ] \\
&=
v _ {\pi} \left(s _ {t}\right)
-
\mathbb {E} \left[
R _ {t + 1}
+ \gamma v _ {\pi} \left(S _ {t + 1}\right)
\mid S _ {t} = s _ {t}
\right] \\
&=0.
\end{aligned}
$$

最后一步正是 Bellman expectation equation。

所以 TD error 不只是“相邻两步不一致”，更重要的是：它告诉我们当前估计距离 Bellman equation 的固定点还有多远。样本里的新信息 innovation 创新，就是通过 TD error 进入更新的。

### 8. TD 和 MC 的核心区别

| 维度 | TD learning | MC learning |
|---|---|---|
| 更新时机 | online 在线，每来一步样本就能更新 | offline 离线，通常要等 episode 完整结束 |
| 任务类型 | 可处理 episodic 和 continuing tasks | 主要处理有限 episode 任务 |
| 是否自举 | bootstrapping 自举，用 $v_t(s_{t+1})$ 更新 $v_t(s_t)$ | non-bootstrapping 非自举，直接用真实采样回报 |
| 初始值 | 需要初始猜测，因为更新依赖当前估计 | 不那么依赖初始价值估计 |
| 方差 | 通常较低，因为只用一步随机量 | 通常较高，因为完整回报包含很多随机量 |
| 偏差 | 可能有 bootstrapping bias 自举偏差 | 回报样本本身更直接，但方差大 |

一句话记忆：

$$
\text{MC：等完整回报，用真实长样本。}
$$

$$
\text{TD：不等结束，用一步样本 + 当前估计。}
$$

⚠️ **不要说 TD 一定比 MC 好。** TD 的优势是在线、方差低、能处理 continuing tasks；代价是它用了自举估计，早期如果 $v_t$ 很差，target 也会被带偏。

### 9. Theorem 7.1：TD 收敛定理

原书定理说：给定策略 $\pi$，使用式 (7.1) 的 TD 算法，如果对每个状态 $s\in S$ 都有

$$
\sum_t \alpha_t(s)=\infty,
\qquad
\sum_t \alpha_t^2(s)<\infty,
$$

那么：

$$
v_t(s)\to v_\pi(s)
\quad\text{almost surely as }t\to\infty,
\quad \forall s\in S.
$$

读法：只要每个状态被足够多次访问，且学习率满足第 6 章 Robbins-Monro/Dvoretzky 风格的条件，TD 的状态值估计会以几乎必然收敛的意义逼近真值。

这两个步长条件的直觉是：

$$
\sum_t\alpha_t(s)=\infty
$$

表示总学习量不能太少，否则还没走到答案附近，步长就没了。

$$
\sum_t\alpha_t^2(s)<\infty
$$

表示噪声影响要能被逐渐压下去，否则随机波动会一直累积。

一个典型例子是：

$$
\alpha_t(s)=\frac{1}{N_t(s)},
$$

其中 $N_t(s)$ 是状态 $s$ 到时刻 $t$ 被访问的次数。

⚠️ **这些条件必须对每个状态成立。** 因为只有被访问时 $\alpha_t(s)>0$；没访问时可以看成 $\alpha_t(s)=0$。如果某个状态几乎不被访问，那么它的价值不可能靠样本学准。

实践中常用 小的常数学习率 $\alpha$。这时严格的

$$
\sum_t\alpha_t^2(s)<\infty
$$

不成立，所以不会得到同样的 almost sure convergence 几乎必然收敛结论；但常数步长适合非平稳问题或策略持续变化的问题，后面 Sarsa 里会更自然。

为什么常数学习率不满足这个条件？假设某个状态 $s$ 被反复访问，而且每次访问都用同一个常数步长：

$$
\alpha_t(s)=0.1.
$$

那么平方后仍然是常数：

$$
\alpha_t^2(s)=0.01.
$$

把它从 $t=1$ 一直加到无穷：

$$
\sum_{t=1}^{\infty}\alpha_t^2(s)
=
0.01+0.01+0.01+\cdots
=
\infty.
$$

所以它不可能小于无穷，也就是不满足

$$
\sum_t\alpha_t^2(s)<\infty.
$$

对比一下递减步长：

$$
\alpha_t(s)=\frac{1}{t}.
$$

这时：

$$
\sum_{t=1}^{\infty}\alpha_t(s)
=
1+\frac12+\frac13+\cdots
=
\infty,
$$

但：

$$
\sum_{t=1}^{\infty}\alpha_t^2(s)
=
1+\frac14+\frac19+\cdots
<\infty.
$$

这正好满足随机近似喜欢的两个条件：第一条保证还会一直学习，第二条保证噪声的累计影响会被压住。

直觉上，常数学习率像是“永远保持同样敏感”：新样本来了，总会用固定幅度修正，所以最后会在真值附近持续抖动。递减学习率像是“越到后面越稳”：早期大胆学，后期小心修正，因此更适合证明几乎必然收敛到一个固定真值。

### 10. Box 7.2：收敛证明的骨架

证明的目的不是重新发明一个新定理，而是把 TD 更新整理成第 6 章 Theorem 6.3 能处理的 stochastic process 随机过程。

先别急着看公式。Box 7.2 其实只有三步：

1. **把“价值估计”改写成“估计误差”。** 不直接看 $v_t(s)$，而是看它离真值还有多远：

$$
\Delta_t(s)=v_t(s)-v_\pi(s).
$$

如果能证明 $\Delta_t(s)\to0$，那就等于证明 $v_t(s)\to v_\pi(s)$。

2. **把 TD 更新改写成误差递推。** 也就是把

$$
v_{t+1}(s)=\cdots
$$

改造成

$$
\Delta_{t+1}(s)
=
(1-\alpha_t(s))\Delta_t(s)
+
\alpha_t(s)\eta_t(s).
$$

这句话的白话读法是：下一步误差 = 旧误差留下大部分 + 这一步随机样本带来的扰动。

3. **证明这个扰动总体上会把误差压小。** $\eta_t(s)$ 单次可能正、可能负、可能很吵；但取条件期望后，它的大小最多是

$$
\gamma\|\Delta_t\|_\infty.
$$

因为 $\gamma<1$，所以平均方向是收缩的。再配合学习率条件和有界方差，就能调用第 6 章的收敛定理。

所以这段证明的主线不是“算出一个新算法”，而是：

$$
\text{TD 更新}
\quad\Longrightarrow\quad
\text{误差递推}
\quad\Longrightarrow\quad
\text{套用随机近似收敛定理}.
$$

对任意状态 $s$，定义估计误差：

$$
\Delta_t(s)\doteq v_t(s)-v_\pi(s).
$$

这里有两个很容易混的符号：

| 符号 | 含义 | 是否随机/随时间变 |
|---|---|---|
| $s$ | 证明里“任意挑出来检查”的一个状态 | 先固定住，用来分析这一格的误差 |
| $s_t$ | 第 $t$ 步轨迹里实际访问到的状态 $S_t$ 的样本值 | 随时间和采样轨迹变化 |

所以证明的问法是：**我任意拿一个状态 $s$ 来看，在第 $t$ 步它是不是刚好等于实际访问状态 $s_t$？**

- 如果 $s=s_t$，说明这个被检查的状态正好被访问到了，TD 会更新它；
- 如果 $s\neq s_t$，说明这个被检查的状态这一步没被访问，TD 不更新它。

为什么要这么绕？因为收敛定理要证明的是所有状态的误差向量都收敛：

$$
\Delta_t
=
\big(\Delta_t(s_1),\Delta_t(s_2),\ldots\big),
$$

不是只证明当前访问到的那个 $s_t$。所以要对任意 $s$ 都写出它在这一步的变化。

如果 $s=s_t$，由 TD 更新：

$$
v _ {t + 1} (s)
=
v _ {t} (s)
-
\alpha_ {t} (s)
\Big (
v _ {t} (s)
-
(r _ {t + 1} + \gamma v _ {t} (s _ {t + 1}))
\Big ).
$$

这行可以等价读成：

$$
\text{当我检查的这个状态 }s\text{ 恰好就是 }s_t\text{ 时，它的更新式就是 TD 更新式。}
$$

所以它不是把 $s$ 和 $s_t$ 写乱了，而是在说一个带条件的分段规则：

$$
v_{t+1}(s)=
\begin{cases}
v_t(s)-\alpha_t(s)\left[v_t(s)-(r_{t+1}+\gamma v_t(s_{t+1}))\right],
& s=s_t,\\
v_t(s),
& s\neq s_t.
\end{cases}
$$

如果只看访问到的那个状态，当然也可以写成：

$$
v_{t+1}(s_t)
=
v_t(s_t)
-
\alpha_t(s_t)
\left[
v_t(s_t)
-
(r_{t+1}+\gamma v_t(s_{t+1}))
\right].
$$

但证明要同时描述所有状态 $s$ 的误差变化，所以原书保留 $s$，再在右边标注条件 $s=s_t$。

两边减去 $v_\pi(s)$：

$$
\Delta_ {t + 1} (s)
=
(1-\alpha_t(s))\Delta_t(s)
+
\alpha_t(s)
\underbrace{
\left[
r_{t+1}+\gamma v_t(s_{t+1})-v_\pi(s)
\right]
}_{\eta_t(s)}.
$$

如果 $s\neq s_t$，价值不更新；这也可以写成同样形式，只是 $\alpha_t(s)=0$。

这里的 $\alpha_t(s)=0$ 是一个“有效学习率”的记号约定：第 $t$ 步没有访问状态 $s$，所以这一步对 $s$ 的更新幅度就是 0。它不是说我们给每个没访问状态真的手动设置了一个超参数，而是为了把“访问时更新、不访问时不更新”写成同一个公式。

具体看，如果统一公式是：

$$
v_{t+1}(s)
=
v_t(s)
-
\alpha_t(s)
\left[
v_t(s)
-
(r_{t+1}+\gamma v_t(s_{t+1}))
\right],
$$

当 $s\neq s_t$ 时，令这一步的有效步长为：

$$
\alpha_t(s)=0,
$$

代进去：

$$
v_{t+1}(s)
=
v_t(s)
-
0\cdot
\left[
v_t(s)
-
(r_{t+1}+\gamma v_t(s_{t+1}))
\right]
=
v_t(s).
$$

这就和原来的“不访问则不更新”完全一样。

可以把它想成：

$$
\alpha_t(s)
=
\begin{cases}
\text{正常学习率}, & s=s_t,\\
0, & s\neq s_t.
\end{cases}
$$

这样写的好处是，后面证明收敛时不必每一行都分两种情况，可以统一写成一个误差递推式。

因此统一得到：

$$
\Delta_ {t + 1} (s)
=
(1-\alpha_t(s))\Delta_t(s)
+
\alpha_t(s)\eta_t(s).
$$

这正是第 6 章随机近似收敛定理要的形状：

$$
\text{下一步误差}
=
\text{旧误差保留一部分}
+
\text{步长}\times\text{噪声/扰动项}.
$$

接下来要验证 Theorem 6.3 的三个条件。

**条件 1：学习率条件。**

这就是 Theorem 7.1 直接假设的：

$$
\sum_t \alpha_t(s)=\infty,
\qquad
\sum_t \alpha_t^2(s)<\infty.
$$

**条件 2：平均扰动是压缩的。**

要证明：

$$
\left\|
\mathbb E[\eta_t(s)\mid\mathcal H_t]
\right\|_\infty
\le
\gamma
\|\Delta_t\|_\infty.
$$

当 $s=s_t$ 时：

$$
\eta_t(s)
=
r_{t+1}+\gamma v_t(s_{t+1})-v_\pi(s_t).
$$

对给定 $s_t$ 取期望：

$$
\mathbb E[\eta_t(s)]
=
\mathbb E[
r_{t+1}+\gamma v_t(s_{t+1})
\mid s_t]
-
v_\pi(s_t).
$$

用 Bellman equation：

$$
v_\pi(s_t)
=
\mathbb E[
r_{t+1}+\gamma v_\pi(s_{t+1})
\mid s_t],
$$

两式相减：

$$
\mathbb E[\eta_t(s)]
=
\gamma
\mathbb E[
v_t(s_{t+1})-v_\pi(s_{t+1})
\mid s_t].
$$

展开成转移概率：

$$
\mathbb E[\eta_t(s)]
=
\gamma
\sum_{s'\in\mathcal S}
p(s'|s_t)
\left[
v_t(s')-v_\pi(s')
\right].
$$

取绝对值并放大到最大误差：

$$
\begin{aligned}
\left|
\mathbb E[\eta_t(s)]
\right|
&\le
\gamma
\sum_{s'\in\mathcal S}
p(s'|s_t)
\max_{s'\in\mathcal S}
\left|
v_t(s')-v_\pi(s')
\right| \\
&=
\gamma
\|\Delta_t\|_\infty.
\end{aligned}
$$

关键就是 $\gamma<1$，它让 Bellman 结构具有 contraction 压缩性。

**条件 3：扰动方差有界。**

当 $s=s_t$ 时：

$$
\eta_t(s)
=
r_{t+1}+\gamma v_t(s_{t+1})-v_\pi(s_t).
$$

如果奖励有界，状态数有限，价值估计也在合理条件下有界，那么这个随机扰动的方差可以被控制。当 $s\neq s_t$ 时，$\eta_t(s)=0$，方差更显然为 0。

于是第 6 章定理可用，得到 TD 收敛。

#### 一个最小例子：看懂 $\Delta$ 和 $\eta$ 在干嘛

还是用两个状态 $A,B$。假设真值是：

$$
v_\pi(A)=10,\qquad v_\pi(B)=4.
$$

当前估计是：

$$
v_t(A)=7,\qquad v_t(B)=6.
$$

那么误差是：

$$
\Delta_t(A)=v_t(A)-v_\pi(A)=7-10=-3,
$$

$$
\Delta_t(B)=v_t(B)-v_\pi(B)=6-4=2.
$$

这说明：$A$ 被低估了 3，$B$ 被高估了 2。

现在第 $t$ 步实际访问到：

$$
s_t=A,\qquad r_{t+1}=1,\qquad s_{t+1}=B,\qquad \gamma=0.9,\qquad \alpha=0.5.
$$

TD 更新 $A$：

$$
\bar v_t
=
1+0.9v_t(B)
=
1+0.9\times6
=
6.4.
$$

$$
v_{t+1}(A)
=
7+0.5(6.4-7)
=
6.7.
$$

所以新的误差是：

$$
\Delta_{t+1}(A)
=
v_{t+1}(A)-v_\pi(A)
=
6.7-10
=
-3.3.
$$

这一次样本反而让 $A$ 的误差从 $-3$ 变成 $-3.3$，更差了一点。为什么收敛证明还成立？因为 TD 不要求每一个样本都让误差变小；它只要求在期望意义上，Bellman 结构会把误差按 $\gamma<1$ 的比例压缩。单次样本会抖，但长期平均方向是对的。

这里的扰动项是：

$$
\eta_t(A)
=
r_{t+1}+\gamma v_t(B)-v_\pi(A)
=
1+0.9\times6-10
=
-3.6.
$$

误差递推给出：

$$
\Delta_{t+1}(A)
=
(1-\alpha)\Delta_t(A)+\alpha\eta_t(A)
=
0.5\times(-3)+0.5\times(-3.6)
=
-3.3.
$$

和直接算出来一样。

这个例子说明 Box 7.2 的公式不是额外发明了什么东西，只是把 TD 更新从“价值的变化”翻译成“误差的变化”。这样一翻译，第 6 章的随机近似定理就能接上。

### 11. 本节闭环

7.1 的核心是把三件事接起来：

| 来源 | 在 7.1 中变成什么 |
|---|---|
| 第 2 章 Bellman expectation equation | TD 要求解的方程 |
| 第 5 章 model-free sampling | 用样本 $(s_t,r_{t+1},s_{t+1})$ 替代环境模型 |
| 第 6 章 stochastic approximation | 解释 TD 为什么是“带噪声求 Bellman 方程的根” |

这一节只做 policy evaluation 策略评估：给定 $\pi$，估计 $v_\pi$。它还不能直接求最优策略。下一节 7.2 会转向 action value 动作值，并引入 Sarsa；一旦学的是 $q_\pi(s,a)$，就可以接上 policy improvement 策略改进，从而开始学习更好的策略。

---

## 7.2 TD learning of action values: Sarsa（动作值的 TD 学习：Sarsa）

**要解决的问题**：7.1 的 TD 只能估计 state value 状态值 $v_\pi(s)$；如果我们想边学习边改进策略，就更需要直接估计 action value 动作值 $q_\pi(s,a)$，因为有了每个动作的价值，才能在状态 $s$ 下判断“哪个动作更值得选”。

这一节的主角是 Sarsa。它可以先作为 policy evaluation 策略评估算法：给定一个策略 $\pi$，估计 $q_\pi(s,a)$；也可以和 $\epsilon$-greedy policy improvement 策略改进合在一起，用来学习更好的策略。

### 1. 从 TD 到 Sarsa：把 $v(s)$ 换成 $q(s,a)$

7.1 的 TD 状态值更新是：

$$
v_{t+1}(s_t)
=
v_t(s_t)
+\alpha_t(s_t)
\left[
r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)
\right].
$$

Sarsa 做的事情非常像，只是把“当前状态”换成“当前状态-动作对”：

$$
q _ {t + 1} \left(s _ {t}, a _ {t}\right)
=
q _ {t} \left(s _ {t}, a _ {t}\right)
- \alpha_ {t} \left(s _ {t}, a _ {t}\right)
\left[
q _ {t} \left(s _ {t}, a _ {t}\right)
-
\left(r _ {t + 1} + \gamma q _ {t} \left(s _ {t + 1}, a _ {t + 1}\right)\right)
\right].
\tag{7.12}
$$

等价地写成“朝目标移动”的形式：

$$
q _ {t + 1} \left(s _ {t}, a _ {t}\right)
=
q _ {t} \left(s _ {t}, a _ {t}\right)
+
\alpha_ {t} \left(s _ {t}, a _ {t}\right)
\left[
\underbrace{
r _ {t + 1} + \gamma q _ {t} \left(s _ {t + 1}, a _ {t + 1}\right)
}_{\text{Sarsa TD target 目标}}
-
\underbrace{
q _ {t} \left(s _ {t}, a _ {t}\right)
}_{\text{当前估计}}
\right].
$$

读法：这一步访问了 $(s_t,a_t)$，于是只更新这个动作值；目标是“一步奖励 $r_{t+1}$ + 下一个状态实际选出的下一个动作 $a_{t+1}$ 的折扣价值”。

其他没访问到的状态-动作对保持不变：

$$
q _ {t + 1} (s, a) = q _ {t} (s, a),
\quad
\text{for all }(s,a)\neq(s_t,a_t).
$$

⚠️ **Sarsa 多出来的关键变量是 $a_{t+1}$。** 它不是只看下个状态 $s_{t+1}$，而是还要看策略在 $s_{t+1}$ 下实际采样出的动作 $a_{t+1}$。这会让它成为 on-policy 同策略算法：它评估并改进的，正是当前正在用来采样的那个策略。

### 2. 为什么叫 Sarsa

每次更新需要一串五元组：

$$
\left(s_t,a_t,r_{t+1},s_{t+1},a_{t+1}\right).
$$

按英文首字母就是：

$$
\text{State-Action-Reward-State-Action}
\quad\Rightarrow\quad
\text{Sarsa}.
$$

这个名字其实很朴素：它直接把算法每一步要吃进去的数据样本写在脸上。

### 3. Sarsa 在数学上求解什么方程

Sarsa 是一个 stochastic approximation 随机近似算法，用来求给定策略 $\pi$ 的 action-value Bellman equation 动作值贝尔曼方程：

$$
q _ {\pi} (s, a)
=
\mathbb {E}
\left[
R + \gamma q _ {\pi} \left(S ^ {\prime}, A ^ {\prime}\right)
\mid s, a
\right],
\quad
\text{for all }(s,a).
\tag{7.13}
$$

拆开读：

$$
\underbrace{q_\pi(s,a)}_{\text{从 }s\text{ 执行动作 }a\text{ 的真价值}}
=
\mathbb E\left[
\underbrace{R}_{\text{即时奖励}}
+
\gamma
\underbrace{q_\pi(S',A')}_{\text{下个状态按 }\pi\text{ 选动作后的动作值}}
\mid s,a
\right].
$$

这里 $A'$ 是到达 $S'$ 之后，再按同一个策略 $\pi$ 采样出来的动作。因此 Bellman 方程里既有环境随机性 $S',R$，也有策略随机性 $A'$。

Box 7.3 从第 2.8.2 节的动作值 Bellman 方程出发：

$$
\begin{aligned}
q _ {\pi} (s, a)
&=
\sum_ {r} r p (r | s, a)
+
\gamma
\sum_ {s ^ {\prime}} \sum_ {a ^ {\prime}}
q _ {\pi} \left(s ^ {\prime}, a ^ {\prime}\right)
p \left(s ^ {\prime} | s, a\right)
\pi \left(a ^ {\prime} | s ^ {\prime}\right).
\tag{7.14}
\end{aligned}
$$



读法：先拿到奖励的期望，再枚举所有可能的下个状态 $s'$ 和下个动作 $a'$，把 $q_\pi(s',a')$ 按发生概率加权平均。

关键分解是：

$$
p(s',a'|s,a)
=
p(s'|s,a)p(a'|s',s,a)
=
p(s'|s,a)\pi(a'|s').
$$

第二个等号用的是 Markov 性质和策略定义：到了 $s'$ 之后，下一动作只由当前策略在 $s'$ 的分布决定，不再额外依赖旧的 $s,a$。

代回去：

$$
q _ {\pi} (s, a)
=
\sum_ {r} r p (r | s, a)
+
\gamma
\sum_ {s ^ {\prime}} \sum_ {a ^ {\prime}}
q _ {\pi} (s ^ {\prime}, a ^ {\prime})
p (s ^ {\prime}, a ^ {\prime} | s, a).
$$

这正是期望形式：

$$
q_\pi(s,a)
=
\mathbb E[
R+\gamma q_\pi(S',A')
\mid s,a].
$$

所以 Sarsa 的单样本目标：

$$
r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})
$$

就是在用一次经验样本近似这个 Bellman 期望。

### 4. 一个最小手算例子：更新一个动作值

设 $\gamma=0.9$，学习率 $\alpha=0.5$。当前有两个状态 $A,B$，每个状态两个动作：$\text{Left},\text{Right}$。当前估计为：

| 状态-动作 | 当前 $q_t$ |
|---|---:|
| $q_t(A,\text{Right})$ | 0 |
| $q_t(B,\text{Left})$ | 4 |
| $q_t(B,\text{Right})$ | 1 |

现在采到一步 Sarsa 样本：

$$
s_t=A,\quad
a_t=\text{Right},\quad
r_{t+1}=2,\quad
s_{t+1}=B,\quad
a_{t+1}=\text{Left}.
$$

Sarsa TD target 是：

$$
r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})
=
2+0.9q_t(B,\text{Left})
=
2+0.9\times4
=
5.6.
$$

于是更新：

$$
\begin{aligned}
q_{t+1}(A,\text{Right})
&=
q_t(A,\text{Right})
+
\alpha
\left[
5.6-q_t(A,\text{Right})
\right]\\
&=
0+0.5(5.6-0)
=2.8.
\end{aligned}
$$

没访问到的动作值不变：

$$
q_{t+1}(B,\text{Left})=4,\qquad
q_{t+1}(B,\text{Right})=1.
$$

| 步骤 | 样本 | TD target | 更新前 $q(A,\text{Right})$ | 更新后 $q(A,\text{Right})$ |
|---:|---|---:|---:|---:|
| 0 | $(A,\text{Right},2,B,\text{Left})$ | 5.6 | 0 | 2.8 |

⚠️ **如果下一个实际动作变成 $\text{Right}$，target 会变。** 假如 $a_{t+1}=\text{Right}$，则 target 是：

$$
2+0.9q_t(B,\text{Right})
=
2+0.9\times1
=2.9.
$$

这正是 Sarsa 的 on-policy 味道：它学到的价值会反映当前策略实际会怎么走，而不是只幻想下个状态一定选最大动作。

### 5. 收敛条件：每个状态-动作对都要充分访问

Theorem 7.2 说：给定策略 $\pi$，如果 Sarsa 更新式 (7.12) 中的学习率对所有 $(s,a)$ 都满足：

$$
\sum_t\alpha_t(s,a)=\infty,
\qquad
\sum_t\alpha_t^2(s,a)<\infty,
$$

那么：

$$
q_t(s,a)\to q_\pi(s,a),
\quad
\text{almost surely as }t\to\infty,
\quad
\forall(s,a).
$$

这和 7.1 的 TD 收敛定理几乎一模一样，只是对象从 $s$ 换成了 $(s,a)$。

⚠️ **这次“充分访问”的要求更强。** 7.1 要求每个状态 $s$ 被访问足够多；Sarsa 要求每个状态-动作对 $(s,a)$ 被访问足够多。因为如果某个动作在某个状态下几乎没被试过，你就没有样本去估计它的 $q_\pi(s,a)$。

这也是为什么后面做控制时要用 $\epsilon$-greedy：它让非贪心动作也有一定概率被试到，不至于太早把探索关死。

### 6. 用 Sarsa 学习最优策略：评估和改进交替进行

到这里，Sarsa 还只是“给定 $\pi$，估计 $q_\pi$”。为了学习更优策略，原书把 Sarsa 和 policy improvement 策略改进结合起来，也仍然称为 Sarsa。

Algorithm 7.1 的每一步有两个核心动作：

1. **更新刚访问的动作值。**

$$
q _ {t + 1} \left(s _ {t}, a _ {t}\right)
=
q _ {t} \left(s _ {t}, a _ {t}\right)
- \alpha_ {t} \left(s _ {t}, a _ {t}\right)
\left[
q _ {t} \left(s _ {t}, a _ {t}\right)
-
\left(r _ {t + 1} + \gamma q _ {t} \left(s _ {t + 1}, a _ {t + 1}\right)\right)
\right].
$$

2. **把当前状态 $s_t$ 的策略更新成对 $q_{t+1}$ 的 $\epsilon$-greedy policy。**

这里很容易误解：$q_{t+1}$ 不是突然知道了所有状态-动作对的真实价值。上一步只更新了一个格子：

$$
q_t(s_t,a_t)\to q_{t+1}(s_t,a_t).
$$

其他格子只是沿用旧估计：

$$
q_{t+1}(s,a)=q_t(s,a),
\quad
\text{for all }(s,a)\neq(s_t,a_t).
$$

所以 $\epsilon$-greedy policy 用的不是“真实动作值”，而是当前手里的估计表 $q_{t+1}$。在状态 $s_t$ 下，它会比较这一行所有候选动作的当前估计：

$$
q_{t+1}(s_t,a),
\quad a\in\mathcal A(s_t).
$$

谁最大，就把谁当作 greedy action 贪心动作。

若 $a$ 是当前最大动作：

$$
\pi_ {t + 1} (a | s _ {t})
=
1 - \frac {\epsilon}{| \mathcal {A} (s _ {t}) |}
\left(| \mathcal {A} (s _ {t}) | - 1\right),
$$

否则：

$$
\pi_ {t + 1} (a | s _ {t})
=
\frac {\epsilon}{| \mathcal {A} (s _ {t}) |}.
$$

这和第 5 章 MC $\epsilon$-greedy 控制的思想相同：大部分时间选当前最好的动作，小概率探索其他动作。

举个小例子。假设当前状态 $s_t$ 有三个动作，更新完一个格子后，估计表这一行是：

| 动作 | 当前估计 |
|---|---:|
| $a_1$ | $q_{t+1}(s_t,a_1)=6$ |
| $a_2$ | $q_{t+1}(s_t,a_2)=5$ |
| $a_3$ | $q_{t+1}(s_t,a_3)=1$ |

那么 $a_1$ 是当前最大动作。若 $\epsilon=0.1$，则策略会大概率选 $a_1$，小概率探索 $a_2,a_3$。

实际实现时，也常常不显式存一张 $\pi$ 表，而是在每次需要选择动作时，直接根据当前 $q_t(s,\cdot)$ 做一次 $\epsilon$-greedy 采样。这等价于说：策略一直被最新的 $q$ 隐式更新。

注意这不是“先把当前策略评估到完全准确，再统一改进策略”。它是每走一步就更新一点 $q$，再马上更新一点策略，然后立刻用新策略继续采样：

$$
\text{采样}
\to
\text{更新 }q
\to
\text{更新 }\pi
\to
\text{继续采样}.
$$

这就是 generalized policy iteration 广义策略迭代：policy evaluation 和 policy improvement 交织在一起、彼此推动。

### 7. Figure 7.2：路径学习例子

![原书图 7.2 左：Sarsa 学到的网格路径策略](../../images/7db7964fb8f11692b5d7e2fcda3a270d307aa642d6c571c3ef173d29c8d727df.jpg)

![原书图 7.2 右：每个 episode 的总奖励和长度](../../images/9f6207fd531d7ea7e66eb5d89925b21561c50d1f0046be04f33031b3e069e598.jpg)

> **原书图 7.2**：所有 episode 从左上角开始，到蓝色 target 状态结束；黄色格子是 forbidden states。左图显示 Sarsa 学到的最终策略，右图显示训练过程中每个 episode 的总奖励和长度变化。

这个例子的奖励设置是：

$$
r_{\mathrm{target}}=0,\qquad
r_{\mathrm{forbidden}}=r_{\mathrm{boundary}}=-10,\qquad
r_{\mathrm{other}}=-1.
$$

所以智能体的目标不是“拿到正奖励”，而是尽快到达终点，并尽量避免撞墙或进入 forbidden states。因为普通移动每步都是 $-1$，episode 越短，总惩罚越小。

右上图的 total rewards 总奖励逐渐上升，意思是：早期乱走时经常撞边界、踩 forbidden states、绕远路，所以总奖励很负；后期路径变短、错误变少，总奖励接近 0。

右下图的 episode length 逐渐下降，意思是：策略越来越会从起点走到目标。偶尔出现突然变长或总奖励暴跌，是因为 $\epsilon$-greedy 仍然保留探索概率，智能体有时会故意或随机选到非最优动作。

⚠️ **图中的学习目标不是求所有状态的全局最优策略。** 原书特别提醒：这个任务每个 episode 都从固定起点出发，只需要找到从起点到目标的好路径。因此很多远离路径的状态可能没有充分探索，它们的策略不一定最优。

### 8. Expected Sarsa：把下个动作的随机性求平均

普通 Sarsa 的 target 是：

$$
r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}).
$$

这里 $a_{t+1}$ 是实际采样出来的动作，所以 target 会受到下一动作采样随机性的影响。

Expected Sarsa 把这一步改成对策略下所有动作求期望：

$$
q _ {t + 1} (s _ {t}, a _ {t})
=
q _ {t} (s _ {t}, a _ {t})
- \alpha_ {t} (s _ {t}, a _ {t})
\Big[
q _ {t} (s _ {t}, a _ {t})
-
\left(
r _ {t + 1}
+
\gamma
\mathbb {E} [ q _ {t} (s _ {t + 1}, A) ]
\right)
\Big].
$$

其中：

$$
\mathbb {E} [ q _ {t} (s _ {t + 1}, A) ]
=
\sum_{a'} \pi_t(a'|s_{t+1})q_t(s_{t+1},a')
\doteq
v_t(s_{t+1}).
$$

读法：到了 $s_{t+1}$ 后，不再只看一次实际抽中的 $a_{t+1}$，而是把当前策略在下一状态可能选的所有候选动作按概率加权平均。

普通 Sarsa 与 Expected Sarsa 的差别可以压缩成一行：

> 记号说明：原书这里常用 $a$ 作为求和里的 dummy variable 哑变量。为了避免把它误读成当前动作 $a_t$，本笔记统一把“下一状态里的候选动作”写成 $a'$。

| 算法             | TD target                                           |
| -------------- | --------------------------------------------------- |
| Sarsa          | $r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})$               |
| Expected Sarsa | $r_{t+1}+\gamma\sum_{a'}\pi_t(a'|s_{t+1})q_t(s_{t+1},a')$ |

这里三个动作符号的角色不同：

| 符号 | 含义 |
|---|---|
| $a_t$ | 当前时刻已经执行的动作 |
| $a_{t+1}$ | 下一时刻按策略实际采样出来的动作 |
| $a'$ | 在下一状态 $s_{t+1}$ 里用于求和枚举的候选动作 |

Expected Sarsa 的好处是方差更低：它不再额外采样一个随机动作 $a_{t+1}$，而是对所有候选 $a'$ 按 $\pi_t(a'|s_{t+1})$ 求平均。代价是要对动作集合求和，动作很多时计算会稍贵。

这里原书接着写式 (7.15)，目的不是突然引入一个新方程，而是要回答一个自然问题：

> Expected Sarsa 把普通 Sarsa 里的 $q_t(s_{t+1},a_{t+1})$ 换成了对下一状态所有候选动作的期望 $\sum_{a'}\pi_t(a'|s_{t+1})q_t(s_{t+1},a')$。那这个“对下一动作先求平均”的 target，还是在求同一个 $q_\pi$ 吗？还是已经变成另一个目标了？

式 (7.15) 就是在证明：**Expected Sarsa 仍然是在求同一个动作值 Bellman 方程，只是把“先采样 $a_{t+1}$ 再更新”改成了“在 $s_{t+1}$ 处先按策略把所有动作平均掉”。**

先把普通 Sarsa 对应的 Bellman 方程写出来：

$$
q_\pi(s,a)
=
\mathbb E[
R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})
\mid
S_t=s,A_t=a
].
$$

这里的期望同时包含三种随机性：奖励 $R_{t+1}$、下个状态 $S_{t+1}$、以及到达下个状态后策略抽出的动作 $A_{t+1}$。

Expected Sarsa 想做的是：不要等 $A_{t+1}$ 抽样出来，而是在给定 $S_{t+1}$ 后，先把 $A_{t+1}$ 这层随机性平均掉。因此就得到一个嵌套期望。

式 (7.15) 看起来有点绕：

$$
q _ {\pi} (s, a)
=
\mathbb {E}
\Big[
R _ {t + 1}
+
\gamma
\mathbb {E}
[q _ {\pi} (S _ {t + 1}, A _ {t + 1})
\mid S _ {t + 1}]
\mid
S _ {t} = s, A _ {t} = a
\Big].
\tag{7.15}
$$

但内部期望其实就是：

$$
\mathbb {E}
\left[
q _ {\pi} \left(S _ {t + 1}, A _ {t + 1}\right)
\mid S _ {t + 1}
\right]
=
\sum_{a'}q_\pi(S_{t+1},a')\pi(a'|S_{t+1})
=
v_\pi(S_{t+1}).
$$

代回去就变成：

$$
q_\pi(s,a)
=
\mathbb E[
R_{t+1}
+
\gamma v_\pi(S_{t+1})
\mid
S_t=s,A_t=a
],
$$

也就是熟悉的动作值 Bellman 方程。

### 9. 本节闭环

7.2 把 7.1 的 TD 思想从状态值推广到动作值：

$$
v_t(s_t)
\quad\leadsto\quad
q_t(s_t,a_t).
$$

一旦学的是 $q(s,a)$，就能在每个状态下比较动作，并接上 $\epsilon$-greedy 策略改进。Sarsa 因此从 policy evaluation 走向了 control 控制。

本节要抓住三句话：

| 句子 | 含义 |
|---|---|
| Sarsa 的样本是 $(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})$ | 更新依赖下一步实际动作 |
| Sarsa 是 on-policy | 它用当前策略采样，也评估当前策略 |
| Sarsa + $\epsilon$-greedy = 广义策略迭代 | 每步交替做一点评估和一点改进 |

接下来 7.3 会继续推广：不是只看一步 TD target，而是看 $n$-step Sarsa，把 MC 的长回报和 TD 的一步自举连成一条连续谱。

---

## 7.3 TD learning of action values: $n$-step Sarsa（动作值的 $n$ 步 Sarsa）

**要解决的问题**：7.2 的 Sarsa 只看一步奖励就开始 bootstrapping 自举；MC learning 蒙特卡洛学习则等到 episode 结束，用完整回报更新。7.3 要把这两端连起来：能不能先看 $n$ 步真实奖励，再用第 $n$ 步后的 $q$ 值自举？

答案就是 $n$-step Sarsa。它把 one-step Sarsa 和 MC 看成同一个家族的两个极端。

### 1. 从动作值定义重新出发

动作值定义是：

$$
q _ {\pi} (s, a)
=
\mathbb {E} [ G _ {t} | S _ {t} = s, A _ {t} = a ].
\tag{7.16}
$$

其中 return 回报为：

$$
G _ {t}
=
R _ {t + 1}
+ \gamma R _ {t + 2}
+ \gamma^ {2} R _ {t + 3}
+ \cdots .
$$

7.3 的关键观察是：这个未来回报可以在不同位置“截断”，截断之后用当前价值估计接上尾巴。

一步形式：

$$
G _ {t} ^ {(1)}
=
R _ {t + 1}
+ \gamma q _ {\pi} (S _ {t + 1}, A _ {t + 1}).
$$

两步形式：

$$
G _ {t} ^ {(2)}
=
R _ {t + 1}
+ \gamma R _ {t + 2}
+ \gamma^ {2} q _ {\pi} (S _ {t + 2}, A _ {t + 2}).
$$

$n$ 步形式：

$$
G _ {t} ^ {(n)}
=
R _ {t + 1}
+ \gamma R _ {t + 2}
+ \cdots
+ \gamma^{n-1}R_{t+n}
+ \gamma^ {n} q _ {\pi} (S _ {t + n}, A _ {t + n}).
$$

无穷步形式，也就是完整 MC 回报：

$$
G _ {t} ^ {(\infty)}
=
R _ {t + 1}
+ \gamma R _ {t + 2}
+ \gamma ^2 R _ {t + 3}
+ \cdots.
$$

读法：$n$-step return 先真实看 $n$ 步奖励，然后在第 $n$ 步位置用 $q_\pi(S_{t+n},A_{t+n})$ 接上后续价值。

⚠️ **更直观地说，$n$ 是“真实往前看几步再自举”。** $n=1$ 时几乎立刻自举；$n$ 很大时越来越像 MC；$n=\infty$ 时完全不自举，直接用完整回报。

### 2. $n=1$：退化成普通 Sarsa

当 $n=1$：

$$
q _ {\pi} (s, a)
=
\mathbb {E}
\left[
R _ {t + 1}
+ \gamma q _ {\pi} (S _ {t + 1}, A _ {t + 1})
\mid s, a
\right].
$$

用样本替代期望，就得到 7.2 的 Sarsa 更新：

$$
q _ {t + 1} (s _ {t}, a _ {t})
=
q _ {t} (s _ {t}, a _ {t})
+ \alpha_ {t} (s _ {t}, a _ {t})
\left[
r _ {t + 1}
+ \gamma q _ {t} (s _ {t + 1}, a _ {t + 1})
- q_t(s_t,a_t)
\right].
$$

也就是说，普通 Sarsa 只是 $n$-step Sarsa 在 $n=1$ 时的特例。

### 3. $n=\infty$：退化成 MC learning

当 $n=\infty$，target 不再包含 $q_t$，而是完整采样回报：

$$
G_t
=
r_{t+1}
+\gamma r_{t+2}
+\gamma^2 r_{t+3}
+\cdots.
$$

这时用完整回报更新：

$$
q(s_t,a_t)
\leftarrow
G_t.
$$

原书写成：

$$
q _ {t + 1} (s _ {t}, a _ {t})
=
g _ {t}
\doteq
r _ {t + 1}
+ \gamma r _ {t + 2}
+ \gamma^ {2} r _ {t + 3}
+ \ldots .
$$

这就是 MC learning 的思想：等 episode 结束，拿到从 $(s_t,a_t)$ 出发的完整折扣回报，再用它估计动作值。

⚠️ **MC 的 target 不自举。** 它不用 $q_t(s_{t+n},a_{t+n})$ 接尾巴，所以初始 $q$ 值不会直接进入 target；但完整回报包含很多随机奖励，方差通常更大。

### 4. 一般 $n$：$n$-step Sarsa 更新式

一般情况下：

$$
q _ {\pi} (s, a)
=
\mathbb {E}
\left[
R _ {t + 1}
+ \gamma R _ {t + 2}
+ \dots
+ \gamma^{n-1}R_{t+n}
+ \gamma^ {n} q _ {\pi} (S _ {t + n}, A _ {t + n})
\mid s, a
\right].
$$

用一次采样轨迹替代期望，得到 $n$-step target：

$$
\bar q_t^{(n)}
\doteq
r _ {t + 1}
+ \gamma r _ {t + 2}
+ \dots
+ \gamma^{n-1}r_{t+n}
+ \gamma^ {n} q _ {t} (s _ {t + n}, a _ {t + n}).
$$

于是更新式为：

$$
q _ {t + 1} \left(s _ {t}, a _ {t}\right)
=
q _ {t} \left(s _ {t}, a _ {t}\right)
+ \alpha_ {t} \left(s _ {t}, a _ {t}\right)
\left[
\bar q_t^{(n)}
- q _ {t} \left(s _ {t}, a _ {t}\right)
\right].
\tag{7.17}
$$

把 target 展开就是：

$$
\begin{aligned}
q _ {t + 1} \left(s _ {t}, a _ {t}\right)
&=
q _ {t} \left(s _ {t}, a _ {t}\right)\\
&\quad
- \alpha_ {t} \left(s _ {t}, a _ {t}\right)
\Big[
q _ {t} \left(s _ {t}, a _ {t}\right)
- \left(
r _ {t + 1}
+ \gamma r _ {t + 2}
+ \dots
+ \gamma^ {n} q _ {t} \left(s _ {t + n}, a _ {t + n}\right)
\right)
\Big].
\end{aligned}
$$

读法：当前动作值向 $n$-step target 靠近。这个 target 的前半段是真实采到的奖励，最后一项是第 $n$ 步后的自举估计。

### 5. 为什么实现时要等到 $t+n$

one-step Sarsa 在时刻 $t+1$ 拿到：

$$
(r_{t+1},s_{t+1},a_{t+1})
$$

就能更新 $(s_t,a_t)$。

但 $n$-step Sarsa 要用：

$$
(s_t,a_t,r_{t+1},s_{t+1},a_{t+1},\ldots,r_{t+n},s_{t+n},a_{t+n}).
$$

在时刻 $t$，这些未来样本还没有发生。因此必须等到时刻 $t+n$，才能回头更新 $q(s_t,a_t)$。

原书把实现时的更新写成：

$$
\begin{aligned}
q _ {t + n} \left(s _ {t}, a _ {t}\right)
&=
q _ {t + n - 1} \left(s _ {t}, a _ {t}\right)\\
&\quad
- \alpha_ {t + n - 1} (s _ {t}, a _ {t})
\Big[
q _ {t + n - 1} (s _ {t}, a _ {t})\\
&\qquad
- \big (
r _ {t + 1}
+ \gamma r _ {t + 2}
+ \dots
+ \gamma^ {n} q _ {t + n - 1} (s _ {t + n}, a _ {t + n})
\big )
\Big].
\end{aligned}
$$

这只是把“理论上用 $t$ 写的更新”改成“实际等到 $t+n$ 才能执行的更新”。

⚠️ **不要误会成只在每隔 $n$ 步才更新一次。** 实际在线实现时，每来一个新时间步，都可以更新“$n$ 步之前”的那一对 $(s_{t-n},a_{t-n})$。只是每个样本要等够 $n$ 步信息。

### 6. 一个 3-step Sarsa 手算例子

设：

$$
\gamma=0.9,\qquad \alpha=0.5.
$$

当前要更新：

$$
q_t(s_t,a_t)=2.
$$

之后三步采到的奖励是：

$$
r_{t+1}=1,\qquad r_{t+2}=0,\qquad r_{t+3}=2.
$$

第 3 步后的动作值估计是：

$$
q_t(s_{t+3},a_{t+3})=5.
$$

3-step target 为：

$$
\begin{aligned}
\bar q_t^{(3)}
&=
r_{t+1}
+\gamma r_{t+2}
+\gamma^2 r_{t+3}
+\gamma^3 q_t(s_{t+3},a_{t+3})\\
&=
1
+0.9\times0
+0.9^2\times2
+0.9^3\times5\\
&=
1+0+1.62+3.645\\
&=
6.265.
\end{aligned}
$$

更新：

$$
\begin{aligned}
q_{t+1}(s_t,a_t)
&=
2
+0.5(6.265-2)\\
&=
4.1325.
\end{aligned}
$$

表格看一眼：

| 项 | 数值 | 作用 |
|---|---:|---|
| $r_{t+1}$ | 1 | 第 1 步真实奖励 |
| $\gamma r_{t+2}$ | 0 | 第 2 步真实奖励折扣后 |
| $\gamma^2 r_{t+3}$ | 1.62 | 第 3 步真实奖励折扣后 |
| $\gamma^3q_t(s_{t+3},a_{t+3})$ | 3.645 | 第 3 步后的自举尾巴 |
| target $\bar q_t^{(3)}$ | 6.265 | 更新要靠近的目标 |
| 新 $q$ | 4.1325 | 从 2 朝 6.265 走一半 |

这个例子里，前 3 步是“真实看到的”，第 3 步之后的未来还没有展开，所以用 $q_t(s_{t+3},a_{t+3})$ 概括。

### 7. bias-variance tradeoff：$n$ 控制偏差和方差

$n$-step Sarsa 的表现介于 one-step Sarsa 和 MC 之间：

| $n$ | 更像谁 | target 特点 | 方差 | 偏差 |
|---:|---|---|---|---|
| $1$ | Sarsa | 一步奖励 + 很早自举 | 低 | 较高 |
| 中等 $n$ | 多步 TD | 多看几步真实奖励，再自举 | 中等 | 中等 |
| $\infty$ | MC | 完整真实回报，不自举 | 高 | 较低 |

为什么小 $n$ 偏差较大？因为 target 很早就依赖当前估计 $q_t$，如果当前估计不准，target 也会被带偏。

为什么大 $n$ 方差较大？因为 target 包含更多真实随机奖励，随机路径越长，波动越大。

一句话：

$$
n\text{ 小：稳，但更依赖当前估计。}
$$

$$
n\text{ 大：更真实，但更吵。}
$$

### 8. 本节闭环

7.3 的核心是把 TD 和 MC 接成一条连续谱：

$$
\text{one-step Sarsa}
\quad
\subset
\quad
n\text{-step Sarsa}
\quad
\subset
\quad
\text{MC learning}.
$$

更准确地说：

$$
n=1
\Longrightarrow
\text{Sarsa},
\qquad
n=\infty
\Longrightarrow
\text{MC}.
$$

本节的 $n$-step Sarsa 仍然只是 policy evaluation 策略评估：给定策略 $\pi$，估计 $q_\pi$。如果要学习最优策略，还要像 7.2 一样，把它和 $\epsilon$-greedy policy improvement 策略改进结合起来。

下一节 7.4 会讲 Q-learning。它会把 Sarsa target 里的“下一步实际动作 $a_{t+1}$”换成“下个状态的最大动作价值”，从 on-policy 走向 off-policy。

---

## 7.4 TD learning of optimal action values: Q-learning（最优动作值的 TD 学习：Q-learning）

**要解决的问题**：Sarsa 学的是给定策略 $\pi$ 的 $q_\pi(s,a)$；Q-learning 直接学习 optimal action value 最优动作值 $q_*(s,a)$，从而得到最优策略。

这一节最容易混的是动作符号，所以先把记号摆平：

| 符号 | 含义 |
|---|---|
| $a_t$ | 当前时刻在 $s_t$ 实际执行的动作 |
| $a_{t+1}$ | 下一时刻在 $s_{t+1}$ 按行为策略实际采样出来的动作 |
| $a'$ | 在下一状态 $s_{t+1}$ 或 $s'$ 中用于枚举/取最大值的候选动作 |
| $\pi_b$ | behavior policy 行为策略，用来生成经验样本 |
| $\pi_T$ | target policy 目标策略，算法真正想学到的策略 |

> 原书在很多 $\max_a$ 里直接使用 $a$ 作为 dummy variable 哑变量。为了避免把它误读成当前动作 $a_t$，本笔记在“下一状态的候选动作”处统一写成 $a'$。

### 1. 从 Sarsa 到 Q-learning：target 变在哪里

Sarsa 的 TD target 是：

$$
r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}).
$$

它使用的是行为策略实际采样出来的下一动作 $a_{t+1}$。

Q-learning 的 TD target 是：

$$
r_{t+1}
+\gamma
\max_{a'\in\mathcal A(s_{t+1})}
q_t(s_{t+1},a').
$$

它不需要知道行为策略在 $s_{t+1}$ 实际采样了哪个 $a_{t+1}$；它只需要知道下一个状态 $s_{t+1}$，然后在这个状态的所有候选动作 $a'$ 中取最大值。

| 算法 | 每步样本需要什么 | TD target | 学什么 |
|---|---|---|---|
| Sarsa | $(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})$ | $r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})$ | 当前策略的 $q_\pi$ |
| Q-learning | $(s_t,a_t,r_{t+1},s_{t+1})$ | $r_{t+1}+\gamma\max_{a'}q_t(s_{t+1},a')$ | 最优动作值 $q_*$ |

这就是 Q-learning 能 off-policy 离策略学习的核心：生成经验的策略可以负责探索，而 target 里的未来动作按贪心方式处理。

### 2. Q-learning 更新式

原书的 Q-learning 更新是：

$$
q _ {t + 1} (s _ {t}, a _ {t})
=
q _ {t} (s _ {t}, a _ {t})
- \alpha_ {t} (s _ {t}, a _ {t})
\left[
q _ {t} (s _ {t}, a _ {t})
-
\left(
r _ {t + 1}
+ \gamma
\max _ {a' \in \mathcal {A} (s _ {t + 1})}
q _ {t} (s _ {t + 1}, a')
\right)
\right].
\tag{7.18}
$$

等价地写成“朝 target 移动”：

$$
q _ {t + 1} (s _ {t}, a _ {t})
=
q _ {t} (s _ {t}, a _ {t})
+
\alpha_ {t} (s _ {t}, a _ {t})
\left[
\underbrace{
r _ {t + 1}
+ \gamma
\max_{a'} q _ {t} (s _ {t + 1}, a')
}_{\text{Q-learning TD target}}
-
\underbrace{
q _ {t} (s _ {t}, a _ {t})
}_{\text{当前估计}}
\right].
$$

其他没访问到的状态-动作对不变：

$$
q _ {t + 1} (s, a)
=
q _ {t} (s, a),
\quad
\text{for all }(s,a)\neq(s_t,a_t).
$$

### 3. 一个最小例子：$a'$ 不是 $a_{t+1}$

假设某一步采到：

$$
s_t=A,\qquad a_t=\text{Right},\qquad r_{t+1}=1,\qquad s_{t+1}=B.
$$

在 $B$ 有两个候选动作：

| 候选动作 $a'$ | 当前估计 $q_t(B,a')$ |
|---|---:|
| Left | 2 |
| Right | 8 |

假设行为策略实际在 $B$ 采样出：

$$
a_{t+1}=\text{Left}.
$$

Sarsa 使用真实采样动作：

$$
r_{t+1}+\gamma q_t(B,\text{Left})
=
1+0.9\times2
=2.8.
$$

Q-learning 不使用这个真实采样动作，而是枚举候选动作并取最大：

$$
r_{t+1}+\gamma\max_{a'}q_t(B,a')
=
1+0.9\max\{2,8\}
=8.2.
$$

同一条经验 $(A,\text{Right},1,B)$，Sarsa 和 Q-learning 的 target 可以完全不同。

一句话：

$$
a_{t+1}\text{ 是行为策略实际抽到的动作；}
\qquad
a'\text{ 是下个状态里用于比较的候选动作。}
$$

### 4. Q-learning 在数学上求解什么

Q-learning 求解的是 action-value Bellman optimality equation 动作值贝尔曼最优方程：

$$
q (s, a)
=
\mathbb {E}
\left[
R _ {t + 1}
+ \gamma
\max _ {a'\in\mathcal A(S_{t+1})}
q \left(S _ {t + 1}, a'\right)
\mid
S _ {t} = s, A _ {t} = a
\right].
\tag{7.19}
$$

如果 $q$ 已经是最优动作值 $q_*$，对应的最优状态值是：

$$
v_*(s)
=
\max_{a\in\mathcal A(s)}q_*(s,a).
$$

因此式 (7.19) 也可以写成：

$$
q_*(s,a)
=
\mathbb E
\left[
R_{t+1}
+\gamma v_*(S_{t+1})
\mid
S_t=s,A_t=a
\right],
$$

其中：

$$
v_*(S_{t+1})
=
\max_{a'\in\mathcal A(S_{t+1})}
q_*(S_{t+1},a').
$$

读法：当前在 $s$ 先做动作 $a$，这个动作的最优价值等于“一步奖励 + 下个状态的最优价值”的期望。

### 5. Box 7.5：为什么式 (7.19) 是 Bellman optimality equation

从式 (7.19) 出发，把期望展开：

$$
q(s,a)
=
\sum_r p(r|s,a)r
+\gamma
\sum_{s'}p(s'|s,a)
\max_{a'\in\mathcal A(s')}
q(s',a').
$$

读法：当前动作 $a$ 的价值 = 当前动作带来的奖励期望 + 下个状态最优动作价值的折扣期望。

对当前状态 $s$ 的所有当前动作 $a$ 取最大值：

$$
\max_{a\in\mathcal A(s)}q(s,a)
=
\max_{a\in\mathcal A(s)}
\left[
\sum_r p(r|s,a)r
+\gamma
\sum_{s'}p(s'|s,a)
\max_{a'\in\mathcal A(s')}
q(s',a')
\right].
$$

这里外层 $a$ 是当前状态 $s$ 下被比较的动作，它会影响 $p(r|s,a)$ 和 $p(s'|s,a)$；内层 $a'$ 是下个状态 $s'$ 下用于取最大值的候选动作。

定义：

$$
v(s)\doteq\max_{a\in\mathcal A(s)}q(s,a).
$$

则上式变成：

$$
v(s)
=
\max_{a\in\mathcal A(s)}
\left[
\sum_r p(r|s,a)r
+\gamma
\sum_{s'}p(s'|s,a)v(s')
\right].
$$

这就是第 3 章讲过的 state-value Bellman optimality equation 状态值贝尔曼最优方程。

也可以把“选最大动作”写成“在所有策略里选最好的动作分布”：

$$
v(s)
=
\max_\pi
\sum_{a\in\mathcal A(s)}
\pi(a|s)
\left[
\sum_r p(r|s,a)r
+\gamma
\sum_{s'}p(s'|s,a)v(s')
\right].
$$

所以式 (7.19) 确实是在动作值层面表达 Bellman optimality equation。

### 6. behavior policy 与 target policy

7.4 引入两个重要概念：

| 策略 | English | 含义 |
|---|---|---|
| 行为策略 | behavior policy $\pi_b$ | 用来和环境交互、生成经验样本的策略 |
| 目标策略 | target policy $\pi_T$ | 算法真正想评估或学到的策略 |

如果：

$$
\pi_b=\pi_T,
$$

就是 on-policy learning 同策略学习。

如果：

$$
\pi_b\neq\pi_T,
$$

就是 off-policy learning 离策略学习。

### 7. Sarsa 为什么是 on-policy

Sarsa 每次需要：

$$
(s_t,a_t,r_{t+1},s_{t+1},a_{t+1}).
$$

样本生成链条是：

$$
s_t
\xrightarrow{\pi_b}
a_t
\xrightarrow{\mathrm{model}}
r_{t+1},s_{t+1}
\xrightarrow{\pi_b}
a_{t+1}.
$$

Sarsa 的 target 是：

$$
r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}).
$$

这里 $a_{t+1}$ 是行为策略 $\pi_b$ 生成的，所以 Sarsa 评估的就是行为策略自己。如果它一边用 $\epsilon$-greedy 采样，一边改进同一个 $\epsilon$-greedy 策略，那么：

$$
\pi_b=\pi_T.
$$

所以 Sarsa 是 on-policy。

### 8. Q-learning 为什么是 off-policy

Q-learning 每次只需要：

$$
(s_t,a_t,r_{t+1},s_{t+1}).
$$

样本生成链条是：

$$
s_t
\xrightarrow{\pi_b}
a_t
\xrightarrow{\mathrm{model}}
r_{t+1},s_{t+1}.
$$

行为策略 $\pi_b$ 负责生成当前动作 $a_t$，从而得到经验样本。但 Q-learning 的 target 是：

$$
r_{t+1}
+\gamma
\max_{a'}q_t(s_{t+1},a').
$$

它不需要 $\pi_b$ 在 $s_{t+1}$ 实际选出的 $a_{t+1}$。它隐含的目标策略是 greedy policy 贪心策略：

$$
\pi_T(s)
=
\arg\max_a q_t(s,a).
$$

因此：

$$
\pi_b\text{ 可以负责探索采样，}
\qquad
\pi_T\text{ 可以负责朝最优策略收敛。}
$$

这两者不必相同，所以 Q-learning 是 off-policy。

⚠️ **off-policy 不等于不探索。** Q-learning 可以用任意行为策略生成数据，但行为策略仍然要充分探索，否则没访问过的状态-动作对学不准。

### 9. on-policy/off-policy 和 online/offline 不要混

这两个维度不同：

| 概念 | 判断问题 | 例子 |
|---|---|---|
| on-policy | $\pi_b=\pi_T$ 吗？ | Sarsa、MC control |
| off-policy | $\pi_b\neq\pi_T$ 也能学吗？ | Q-learning |
| online | 每拿到样本就更新吗？ | TD、Sarsa、在线 Q-learning |
| offline | 等样本收集完再处理吗？ | MC、离线 Q-learning |

Q-learning 是 off-policy，所以它可以 online，也可以 offline。

### 10. Algorithm 7.2 与 7.3：两种实现

**Algorithm 7.2：on-policy version。**

这个版本用当前 $\epsilon$-greedy 策略采样，也把策略更新成 $\epsilon$-greedy。它看起来像 Sarsa，但更新 target 仍然是：

$$
r_{t+1}
+\gamma
\max_{a'}q_t(s_{t+1},a').
$$

所以它和 Sarsa 的关键差别不是“采样时是否探索”，而是“更新 target 是否使用实际下一动作 $a_{t+1}$”。

**Algorithm 7.3：off-policy version。**

这个版本把两种策略明确分开：

| 策略 | 作用 |
|---|---|
| $\pi_b$ | 生成 episode，可以很探索，比如 uniform policy |
| $\pi_T$ | 根据 $q$ 贪心更新，不负责探索 |

对 $\pi_b$ 生成的每一步经验：

$$
(s_t,a_t,r_{t+1},s_{t+1})
$$

用 Q-learning target 更新 $q(s_t,a_t)$，然后把目标策略更新成：

$$
\pi_{T,t+1}(a|s_t)=1
\quad
\text{if }a=\arg\max_a q_{t+1}(s_t,a),
$$

$$
\pi_{T,t+1}(a|s_t)=0
\quad
\text{otherwise}.
$$

目标策略可以是 greedy，而不是 $\epsilon$-greedy，因为它不负责生成样本；探索交给 $\pi_b$。

### 11. Figure 7.3：Q-learning 的路径学习例子

![原书图 7.3 左：Q-learning 学到的路径策略](../../images/e7c5e45b54a32f4b23f4526e5e71a77481921f7b2a774bdfb506c8625a98809b.jpg)

![原书图 7.3 右上：每个 episode 的总奖励](../../images/3586c83890855f9a9124b41487b44bcac36352ec5660dfb1f4d0fa4d690610bc.jpg)

![原书图 7.3 右下：每个 episode 的长度](../../images/9a9a4d8fa0a1d93d83d593012f0fcea650fab88463dd1dbd8c88a00022c81c68.jpg)

> **原书图 7.3**：所有 episode 从左上角开始，到蓝色 target 状态结束。奖励设置为 $r_{\mathrm{target}}=0$、$r_{\mathrm{forbidden}}=r_{\mathrm{boundary}}=-10$、$r_{\mathrm{other}}=-1$，学习率 $\alpha=0.1$，探索率 $\epsilon=0.1$。

这个例子演示的是 on-policy implementation of Q-learning：用 $\epsilon$-greedy 采样，同时用 Q-learning target 更新。

右侧曲线和 Sarsa 例子类似：

- total rewards 总奖励逐渐上升：说明踩 forbidden/boundary 和绕路变少。
- episode length 轨迹长度逐渐下降：说明从起点到目标的路径越来越短。
- 偶尔仍有波动：因为 $\epsilon$-greedy 还保留探索。

### 12. Off-policy 示例：为什么行为策略要足够探索

原书后面用 Figure 7.4 和 Figure 7.5 演示 off-policy Q-learning。当前 markdown 切片没有嵌入这些图，但文字给出了结论。

实验目标：学习所有状态的最优策略，而不只是固定起点到目标的路径。

奖励设置：

$$
r_{\mathrm{boundary}}
=
r_{\mathrm{forbidden}}
=
-1,
\qquad
r_{\mathrm{target}}=1,
\qquad
\gamma=0.9,
\qquad
\alpha=0.1.
$$

关键过程：

1. 用 model-based policy iteration 策略迭代算出 ground truth 真值，作为比较标准。
2. 用行为策略 $\pi_b$ 生成经验。一个例子中，$\pi_b$ 是 uniform policy 均匀策略，每个动作概率都是 $0.2$。
3. 因为均匀策略探索能力强，长 episode 会多次访问每个状态-动作对。
4. Q-learning 使用这些经验学习 greedy target policy，最终 RMSE 收敛到 0，说明估计状态值接近真值。

原书还比较了不同初始值和不同行为策略：

- 初始 $q_0$ 越接近真值，收敛越快。
- 行为策略探索越弱，学习越慢。
- 当行为策略从 uniform / $\epsilon=1$ 变成 $\epsilon=0.5$ 再到 $\epsilon=0.1$，探索能力下降，学习速度明显变慢。

⚠️ **off-policy 不等于“不需要探索”。** Q-learning 可以用任意行为策略生成数据，但如果行为策略没怎么访问某些状态-动作对，那这些地方就学不准。离策略的自由度不是魔法，样本覆盖仍然是硬条件。

### 13. 本节闭环

7.4 完成了本章从 evaluation 到 control 的关键跳跃：

| 小节 | 学什么 | target |
|---|---|---|
| 7.1 TD | $v_\pi$ | $r+\gamma v(s')$ |
| 7.2 Sarsa | $q_\pi$ | $r+\gamma q(s',a')$ |
| 7.3 $n$-step Sarsa | $q_\pi$ | 多步奖励 + $q(s_{t+n},a_{t+n})$ |
| 7.4 Q-learning | $q_*$ | $r+\gamma\max_{a'} q(s',a')$ |

最要紧的一句话：

$$
\text{Sarsa 用实际下一动作，Q-learning 用下个状态的最大动作价值。}
$$

这一个 $\max$，把 Sarsa 的 policy evaluation 方程换成了 Bellman optimality equation，也让 Q-learning 能用探索性行为策略的数据来学习贪心目标策略。

下一节 7.5 会给一个 unified viewpoint 统一视角，把本章这些 TD 方法放到同一个框架里看。

---

## 7.5 A unified viewpoint（统一视角）

**要解决的问题**：7.2 到 7.4 看起来有好几个算法：Sarsa、$n$-step Sarsa、Q-learning、MC。7.5 要说明：它们其实共享同一个更新骨架，只是 TD target 目标 $\bar q_t$ 不同。

统一更新式是：

$$
q _ {t + 1} \left(s _ {t}, a _ {t}\right)
=
q _ {t} \left(s _ {t}, a _ {t}\right)
- \alpha_ {t} \left(s _ {t}, a _ {t}\right)
\left[
q _ {t} \left(s _ {t}, a _ {t}\right)
- \bar {q} _ {t}
\right].
\tag{7.20}
$$

换成更熟悉的“朝 target 移动”的形式：

$$
q _ {t + 1} \left(s _ {t}, a _ {t}\right)
=
q _ {t} \left(s _ {t}, a _ {t}\right)
+
\alpha _t(s_t,a_t)
\left[
\bar q_t
-
q_t(s_t,a_t)
\right].
$$

读法：所有这些算法都在做同一件事：

$$
\text{新估计}
=
\text{旧估计}
+
\text{学习率}
\times
(\text{TD target}-\text{旧估计}).
$$

区别只在于：$\bar q_t$ 怎么定义。

### 1. 不同算法的 $\bar q_t$

| 算法 | $\bar q_t$ |
|---|---|
| Sarsa | $r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})$ |
| $n$-step Sarsa | $r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^n q_t(s_{t+n},a_{t+n})$ |
| Q-learning | $r_{t+1}+\gamma\max_{a'}q_t(s_{t+1},a')$ |
| Monte Carlo | $r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots$ |

这张表就是 Table 7.2 的第一层意思：同一个更新壳子，换不同 target，就得到不同算法。

注意 MC learning 也可以看成这个统一式的特例。若令：

$$
\alpha_t(s_t,a_t)=1,
$$

则式 (7.20) 变成：

$$
q_{t+1}(s_t,a_t)
=
\bar q_t.
$$

也就是直接把动作值估计替换成完整采样回报。

### 2. 这些 target 对应求解什么方程

统一地说，式 (7.20) 是一个 stochastic approximation 随机近似算法，用来求：

$$
q(s,a)
=
\mathbb E[\bar q_t\mid s,a].
$$

不同 $\bar q_t$ 会给出不同的方程。

| 算法 | 要解的方程 | 类型 |
|---|---|---|
| Sarsa | $q_\pi(s,a)=\mathbb E[R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})\mid S_t=s,A_t=a]$ | Bellman equation |
| $n$-step Sarsa | $q_\pi(s,a)=\mathbb E[R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^n q_\pi(S_{t+n},A_{t+n})\mid S_t=s,A_t=a]$ | Bellman equation 的多步展开 |
| Q-learning | $q(s,a)=\mathbb E[R_{t+1}+\gamma\max_{a'}q(S_{t+1},a')\mid S_t=s,A_t=a]$ | Bellman optimality equation |
| Monte Carlo | $q_\pi(s,a)=\mathbb E[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots\mid S_t=s,A_t=a]$ | Bellman equation / return 定义 |

原书表 7.2 的结论是：

> 除 Q-learning 外，这些算法都在求给定策略的 Bellman equation；Q-learning 求的是 Bellman optimality equation。

也就是：

$$
\text{Sarsa / n-step Sarsa / MC}
\quad\Rightarrow\quad
q_\pi.
$$

$$
\text{Q-learning}
\quad\Rightarrow\quad
q_*.
$$

### 3. 用一条轴看 TD 与 MC

从 target 的角度看，本章算法可以排成一条轴：

$$
\text{one-step TD}
\quad\longrightarrow\quad
n\text{-step TD}
\quad\longrightarrow\quad
\text{MC}.
$$

越靠左，越早 bootstrapping 自举：

$$
r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}).
$$

越靠右，越依赖真实采样回报：

$$
r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots.
$$

这也是 7.3 的 bias-variance tradeoff 偏差-方差权衡：

| 位置 | 特点 |
|---|---|
| 更像 TD | target 短，方差低，但依赖当前估计，可能有 bootstrapping bias |
| 更像 MC | target 长，更接近真实回报，但方差高，通常要等 episode 结束 |

### 4. 用另一条轴看 Sarsa 与 Q-learning

Sarsa 和 Q-learning 的差别不是“看几步”，而是 target 的最后一步怎么处理：

| 算法 | 下一状态的动作怎么处理 | 学到什么 |
|---|---|---|
| Sarsa | 用实际采样动作 $a_{t+1}$ | 当前策略价值 $q_\pi$ |
| Expected Sarsa | 对候选动作 $a'$ 按 $\pi(a'|s_{t+1})$ 求平均 | 当前策略价值 $q_\pi$ |
| Q-learning | 对候选动作 $a'$ 取最大值 | 最优动作值 $q_*$ |

所以：

$$
\text{Sarsa：跟着当前策略走。}
$$

$$
\text{Expected Sarsa：把当前策略的动作随机性平均掉。}
$$

$$
\text{Q-learning：假设下一步以后都选当前最优动作。}
$$

### 5. 一个统一小例子

设某一步经验为：

$$
r_{t+1}=1,\qquad \gamma=0.9,\qquad s_{t+1}=B.
$$

当前在 $B$ 的动作值为：

| 候选动作 $a'$ | $q_t(B,a')$ | 策略概率 $\pi(a'|B)$ |
|---|---:|---:|
| Left | 2 | 0.25 |
| Right | 8 | 0.75 |

如果实际采样到：

$$
a_{t+1}=\text{Left},
$$

那么不同 target 是：

| 算法 | target |
|---|---:|
| Sarsa | $1+0.9\times2=2.8$ |
| Expected Sarsa | $1+0.9(0.25\times2+0.75\times8)=6.85$ |
| Q-learning | $1+0.9\max\{2,8\}=8.2$ |

这三个数很能说明差别：

- Sarsa 尊重实际采样动作 Left，所以 target 低。
- Expected Sarsa 按策略平均 Left/Right，所以居中。
- Q-learning 直接拿最大动作 Right，所以最高。

### 6. 本节闭环

7.5 的统一视角是：

$$
\boxed{
q_{t+1}(s_t,a_t)
=
q_t(s_t,a_t)
+
\alpha_t(s_t,a_t)
[\bar q_t-q_t(s_t,a_t)]
}
$$

所有算法都在改同一个东西：刚访问的动作值 $q(s_t,a_t)$。不同算法只是在回答：

> 这次我要靠近哪个 target？

于是：

| 算法 | 选择的 target 思想 |
|---|---|
| Sarsa | 用实际下一动作的一步 TD target |
| $n$-step Sarsa | 用 $n$ 步真实奖励 + 自举 |
| Q-learning | 用下一状态最大动作值 |
| MC | 用完整采样回报 |

下一节 7.6 是本章 Summary，会把第 7 章的 TD learning 主线收束起来。

---

## 7.6 Summary（本章总结）

**要解决的问题**：这一节不是引入新算法，而是把第 7 章的 TD learning 时序差分学习主线收束成几个判断：哪些算法是在评估给定策略，哪些算法可以配合策略改进求最优，为什么 Q-learning 是特殊的 off-policy 方法。

### 1. 本章学了哪些算法

原书把本章的核心算法概括为：

| 算法 | 核心 target | 解决什么 |
|---|---|---|
| TD state-value learning | $r_{t+1}+\gamma v_t(s_{t+1})$ | 估计给定策略的 $v_\pi$ |
| Sarsa | $r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})$ | 估计当前策略的 $q_\pi$，并可配合策略改进 |
| $n$-step Sarsa | $r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^n q_t(s_{t+n},a_{t+n})$ | 在 Sarsa 和 MC 之间折中 |
| Q-learning | $r_{t+1}+\gamma\max_{a'}q_t(s_{t+1},a')$ | 直接估计最优动作值 $q_*$ |

它们共同的数学底色是 stochastic approximation 随机近似：拿一个带噪声的样本 target，反复把当前估计往正确方程的解推近。

统一写法是：

$$
q_{t+1}(s_t,a_t)
=
q_t(s_t,a_t)
+
\alpha_t(s_t,a_t)
\left[
\bar q_t-q_t(s_t,a_t)
\right].
$$

读法：每次只更新刚访问的 $(s_t,a_t)$，把它朝本次 target $\bar q_t$ 移动一小步。

### 2. 大部分 TD 方法是在评估给定策略

除了 Q-learning，本章这些 TD 方法本质上是在做 policy evaluation 策略评估：

$$
\text{给定策略 }\pi
\quad\Longrightarrow\quad
\text{从样本估计 }v_\pi\text{ 或 }q_\pi.
$$

例如 Sarsa 的目标方程是：

$$
q_\pi(s,a)
=
\mathbb E_\pi
\left[
R_{t+1}
+
\gamma q_\pi(S_{t+1},A_{t+1})
\mid
S_t=s,A_t=a
\right].
$$

这里的 $A_{t+1}$ 是按同一个策略 $\pi$ 采样出来的下一动作，所以 Sarsa 学的是“当前策略真实会走出来的动作值”。

如果把策略评估和 policy improvement 策略改进连起来，比如每次用 $\epsilon$-greedy 根据当前 $q_t$ 调整策略，就可以逐步学到更好的策略。

### 3. Q-learning 为什么特殊

Q-learning 特殊在于它不求某个给定策略的 $q_\pi$，而是求最优动作值：

$$
q_*(s,a)
=
\mathbb E
\left[
R_{t+1}
+
\gamma\max_{a'}q_*(S_{t+1},a')
\mid
S_t=s,A_t=a
\right].
$$

关键差别在最后一项：

$$
\max_{a'}q_*(S_{t+1},a').
$$

这不是问“行为策略下一步实际会选什么 $a_{t+1}$”，而是问“到了下一状态后，如果从候选动作 $a'$ 里选最好的，价值是多少”。

因此 Q-learning 的 target policy 目标策略天然是 greedy policy 贪心策略：

$$
\pi_T(s)
=
\arg\max_a q(s,a).
$$

但生成样本的 behavior policy 行为策略可以是别的，比如 $\epsilon$-greedy。只要行为策略有足够探索，Q-learning 就可以一边用探索策略收集数据，一边学习贪心目标策略。

### 4. on-policy 和 off-policy 的本章结论

本章最后把结论说得很清楚：

| 类型 | 代表算法 | 含义 |
|---|---|---|
| on-policy | Sarsa、$n$-step Sarsa、MC control 的常见形式 | 用同一个策略生成样本并学习它自己的价值 |
| off-policy | Q-learning | 行为策略负责探索，目标策略可以是另一个策略 |

但要注意：

$$
\text{off-policy}\neq\text{不需要探索}.
$$

Q-learning 虽然不使用实际采样的 $a_{t+1}$ 来构造 target，但它仍然需要行为策略覆盖足够多的状态-动作对。没有见过的数据，算法没有办法凭空学出来。

### 5. 和后续章节的连接

原书在 Summary 里还提前点了两个方向：

| 后续方向 | 作用 |
|---|---|
| importance sampling 重要性采样 | 第 10 章会讲，可把一些 on-policy 方法改造成 off-policy 方法 |
| $\mathrm{TD}(\lambda)$ | 更统一的 TD 框架，把多步 TD 和 MC 之间的折中做得更平滑 |

所以第 7 章完成的是“表格型 TD 学习”的主干：

$$
\text{one-step TD}
\quad\rightarrow\quad
\text{Sarsa}
\quad\rightarrow\quad
n\text{-step Sarsa}
\quad\rightarrow\quad
\text{Q-learning}
\quad\rightarrow\quad
\text{统一视角}.
$$

后面会继续处理两个大问题：

- 当状态空间太大，不能为每个状态-动作对存一张表时怎么办？
- 当采样策略和目标策略不同，如何更系统地修正样本分布差异？

### 6. 本章闭环

第 7 章可以压缩成一句话：

> TD learning 用“样本奖励 + 未来价值估计”构造 target，再用随机近似不断逼近 Bellman equation 或 Bellman optimality equation 的解。

其中：

- TD / Sarsa / $n$-step Sarsa / MC 主要围绕 Bellman equation 贝尔曼方程，也就是给定策略的价值。
- Q-learning 围绕 Bellman optimality equation 贝尔曼最优方程，也就是最优价值。
- Expected Sarsa、$n$-step Sarsa、Q-learning 的差异，本质上都是 target $\bar q_t$ 怎么定义。

本章最后一节 7.7 是本章 Q&A，会回答本章末尾的概念问题。

---

## 7.7 Q&A（问答）

**要解决的问题**：把这一章最容易混淆的几个词再钉牢一次，尤其是 TD、learning、on-policy / off-policy、epsilon-greedy 这些词到底在说什么。

### 1. TD 里的 "TD" 是什么意思

TD 是 **temporal-difference**，时间差分。

原因很直接：TD error TD 误差不是拿完整回报去比，而是拿“当前时刻的估计”和“下一时刻样本构造出来的 target”去比。也就是：

$$
\text{current estimate}
\quad \text{vs.}\quad
\text{sample from a later time}.
$$

所以它强调的是“跨时间步做差”。

### 2. TD 里的 "learning" 是什么意思

这里的 learning 学习，数学上就是 estimation 估计。

TD learning 的目标不是凭空“悟出”一个策略，而是先从样本里估计 state value 状态值或 action value 动作值，再根据估计值去改进策略。

### 3. Sarsa 怎么能学最优策略

Sarsa 本身先做的是 value estimation 价值估计，但它可以和 policy improvement 策略改进交替进行：

1. 用当前策略产生样本。
2. 用 Sarsa 更新 $q_t$。
3. 根据更新后的 $q_t$ 把策略改成更贪心的形式，比如 $\epsilon$-greedy。
4. 新策略再生成新样本。

这就是 generalized policy iteration 广义策略迭代。

所以 Sarsa 不是“天然直接输出最优策略”，而是“通过评估 + 改进”一步步逼近最优。

### 4. 为什么 Sarsa 的策略要设成 $\epsilon$-greedy

因为 Sarsa 用来更新价值的样本，就是这条策略自己生成的。

所以策略不能太死，得留一点探索性，不然很多状态-动作对根本采不到。$\epsilon$-greedy 正好兼顾两件事：

- 大部分时间选当前看起来最好的动作。
- 小部分时间随机探索。

### 5. 为什么实践中常用常数学习率

理论收敛条件通常喜欢递减步长，但实际里策略常常在变。

Sarsa 这类方法在做 policy evaluation 时，评估对象本身会随着策略改进而变化。若学习率衰减得太快，后面几乎学不动了，所以常用一个足够小的常数学习率。

代价是值函数会在真值附近抖一点，但通常可接受。

### 6. 要学所有状态还是一部分状态

看任务。

如果任务只关心从起点到目标的一条好路径，那不一定要把所有状态都学遍。这样更省数据，但也要承认：

$$
\text{没充分探索过的路径，不保证最优}.
$$

如果真要全局最优，还是得尽量覆盖足够多的状态-动作对。

### 7. 为什么 Q-learning 是 off-policy

因为它学的是 Bellman optimality equation 贝尔曼最优方程，不是某个固定策略的 Bellman equation。

它的 target 里用的是：

$$
\max_{a'}q(S_{t+1},a').
$$

这相当于目标策略是 greedy policy 贪心策略，而行为策略可以是别的，比如带探索的 $\epsilon$-greedy。

### 8. 为什么 off-policy 的 Q-learning 目标策略要 greedy，而不是 $\epsilon$-greedy

因为目标策略不用拿来采样。

既然它不负责生成经验，就不需要保留探索性。它只要在数学上表达“最优应该怎么选”，greedy 就够了。

### 9. 本章最后一句

TD learning 的核心不是某一个公式，而是这件事：

> 用样本构造 target，再用随机近似不断逼近某个 Bellman 方程的解。

这就是第 7 章整章的主线。

---

## 我的疑问与解答

**问：Box 7.2 的收敛证明到底在干嘛？**

答：它在把 TD 更新式翻译成第 6 章随机近似定理能识别的形状。原来我们看的是价值 $v_t(s)$，证明里改看误差 $\Delta_t(s)=v_t(s)-v_\pi(s)$。如果能证明误差 $\Delta_t(s)\to0$，就等于证明价值估计 $v_t(s)$ 收敛到真值 $v_\pi(s)$。接着把 TD 更新整理成

$$
\Delta_{t+1}(s)
=
(1-\alpha_t(s))\Delta_t(s)+\alpha_t(s)\eta_t(s).
$$

这表示：下一步误差由旧误差和样本扰动组成。然后证明这个扰动在期望意义上被 $\gamma<1$ 压缩，再加上学习率条件和方差有界，就能套用第 6 章定理。

**问：为什么实践中常用小的常数学习率 $\alpha$ 时，$\sum_t\alpha_t^2(s)<\infty$ 不成立？**

答：因为常数学习率不会随时间变小。比如 $\alpha_t(s)=0.1$，那么 $\alpha_t^2(s)=0.01$，平方和就是：

$$
0.01+0.01+0.01+\cdots=\infty.
$$

所以它不满足“平方和有限”的条件。递减步长如 $\alpha_t=1/t$ 不一样：它本身的和发散，保证算法还能继续学；但平方和 $\sum_t1/t^2$ 收敛，保证随机噪声不会无限累积。

直觉是：常数学习率会让估计值在真值附近持续小幅波动；递减学习率会让后期更新越来越小，更容易严格证明收敛到固定真值。

---

## 脉络总结 / 要点速记

7.1 一句话：TD learning 用一步样本构造 Bellman target，并用随机近似的方式在线估计给定策略的状态值。

7.2 一句话：Sarsa 把 TD 从 $v(s)$ 推广到 $q(s,a)$，用样本 $(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})$ 估计动作值，并通过 $\epsilon$-greedy 策略改进学习更好的策略。

7.3 一句话：$n$-step Sarsa 先累计 $n$ 步真实奖励，再用第 $n$ 步后的 $q$ 值自举，把 one-step Sarsa 和 MC learning 连成一条连续谱。

7.4 一句话：Q-learning 用 $r+\gamma\max_{a'} q(s',a')$ 作为 target，直接学习最优动作值 $q_*$，因此可以把生成数据的行为策略和要学习的贪心目标策略分开。

7.5 一句话：Sarsa、$n$-step Sarsa、Q-learning 和 MC 都可以写成 $q_{t+1}=q_t+\alpha(\bar q_t-q_t)$，区别只在 TD target $\bar q_t$ 的定义。

7.6 一句话：本章所有 TD 方法都可以看成用随机近似求 Bellman equation 或 Bellman optimality equation；除 Q-learning 外主要是 on-policy 策略评估，Q-learning 因为求最优方程所以天然是 off-policy。

7.7 一句话：TD 是跨时间步构造误差，learning 在数学上就是估计；Sarsa 靠评估与改进交替逼近最优，Q-learning 因求最优方程而可以 off-policy。

| 关键词 | 速记 |
|---|---|
| TD target | $\bar v_t=r_{t+1}+\gamma v_t(s_{t+1})$ |
| TD error | 本书写作 $\delta_t=v_t(s_t)-\bar v_t$ |
| TD update | $v_{t+1}(s_t)=v_t(s_t)-\alpha_t(s_t)\delta_t$ |
| bootstrapping | 用 $v_t(s_{t+1})$ 这个估计值来更新 $v_t(s_t)$ |
| online | 每来一步样本就能更新 |
| convergence | 在 Robbins-Monro 型步长条件和足够访问下，$v_t(s)\to v_\pi(s)$ |
| Sarsa target | $r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})$ |
| Sarsa update | $q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha[\text{target}-q_t(s_t,a_t)]$ |
| on-policy | 下一动作 $a_{t+1}$ 来自当前正在执行的策略 |
| Expected Sarsa | 用 $\sum_{a'}\pi(a'|s_{t+1})q_t(s_{t+1},a')$ 替代采样动作 $q_t(s_{t+1},a_{t+1})$ |
| $n$-step target | $r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n-1}r_{t+n}+\gamma^n q_t(s_{t+n},a_{t+n})$ |
| $n=1$ | 退化成普通 Sarsa |
| $n=\infty$ | 退化成 MC learning |
| Q-learning target | $r_{t+1}+\gamma\max_{a'} q_t(s_{t+1},a')$ |
| unified update | $q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha_t(s_t,a_t)[\bar q_t-q_t(s_t,a_t)]$ |
| Bellman equation | 给定策略 $\pi$ 的价值方程，目标是 $v_\pi$ 或 $q_\pi$ |
| Bellman optimality equation | 最优价值方程，目标是 $v_*$ 或 $q_*$ |
| behavior policy | $\pi_b$，负责生成经验样本 |
| target policy | $\pi_T$，算法真正想学到的策略 |
| off-policy | $\pi_b$ 和 $\pi_T$ 可以不同 |

易错点：

- TD 本节估计的是给定策略的 $v_\pi$，不是直接求最优策略。
- 未访问状态不更新，式 (7.2) 不能在理解上丢掉。
- 本书 TD error 的符号是“当前估计 - target”，和很多教材常见写法相反。
- TD target 里用了当前估计 $v_t(s_{t+1})$，所以它是 bootstrapping，不是完整回报。
- 收敛条件要求每个状态都被充分访问；探索不足时，再漂亮的更新式也学不到没见过的地方。
- Sarsa 的收敛条件要求每个状态-动作对都被充分访问，比只访问每个状态更强。
- Sarsa 的 target 用的是实际采样的 $a_{t+1}$，不是 $\max_{a'} q(s_{t+1},a')$；这一点和 Q-learning 形成鲜明对比。
- $n$-step Sarsa 要等到 $t+n$ 才能更新 $(s_t,a_t)$，因为 target 需要未来 $n$ 步的信息。
- $n$ 越小越像 TD：方差低但 bootstrapping bias 更明显；$n$ 越大越像 MC：偏差小但方差更高。
- Q-learning 的 target 不使用实际采样的 $a_{t+1}$，而是用 $\max_{a'} q(s_{t+1},a')$ 枚举下一状态的候选动作，所以它求的是最优动作值而不是某个给定策略的动作值。
- off-policy 不是不探索；恰恰相反，行为策略通常要足够探索，才能让每个状态-动作对有足够样本。
- online/offline 和 on-policy/off-policy 是两个不同维度：前者说什么时候更新，后者说采样策略和目标策略是不是同一个。
