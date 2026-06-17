# 补充：从“估计值减真值”到 Bellman target

> 对应第 8 章 value function approximation · 学习日期 2026-06-17

## 0. 你这句话抓住了主线

你说的“重点就是估计值减真值”很对。最理想的监督学习目标就是：

$$
\text{误差}
=
\text{估计值}
-
\text{真值}.
$$

放到 state value 状态值里：

$$
\hat v(s,w)-v_\pi(s).
$$

放到 action value 动作值里：

$$
\hat q(s,a,w)-q_\pi(s,a)
\quad\text{或}\quad
\hat q(s,a,w)-q^*(s,a).
$$

如果真值已知，事情会非常像普通监督学习：

$$
J_E(w)
=
\sum_s d_\pi(s)
\left(
\hat v(s,w)-v_\pi(s)
\right)^2.
$$

读法：让函数输出 $\hat v(s,w)$ 尽量贴近真实答案 $v_\pi(s)$。

问题是：在 reinforcement learning 强化学习里，$v_\pi(s)$、$q_\pi(s,a)$、$q^*(s,a)$ 通常都不知道。

所以核心转折是：

> 真值拿不到，就不用“真值标签”直接监督；改用 Bellman equation 贝尔曼方程提供的自洽关系来构造 target。

---

## 1. 真值虽然未知，但它满足 Bellman 方程

对给定策略 $\pi$，真实状态值满足 Bellman equation：

$$
v_\pi(s)
=
\mathbb E_\pi
\left[
R_{t+1}
+\gamma v_\pi(S_{t+1})
\mid S_t=s
\right].
$$

这句话不是直接告诉你 $v_\pi(s)$ 的数值，而是告诉你：

$$
\text{当前真值}
=
\text{即时奖励}
+\gamma\times\text{下一状态真值的期望}.
$$

如果写成算子形式：

$$
v_\pi=T_\pi v_\pi.
$$

也就是说，真实价值 $v_\pi$ 是 Bellman operator 贝尔曼算子 $T_\pi$ 的 fixed point 固定点。

这一步的思想很重要：

| 如果有真值 | 如果没有真值 |
|---|---|
| 直接让 $\hat v$ 接近 $v_\pi$ | 让 $\hat v$ 尽量满足 Bellman 自洽关系 |
| supervised error | Bellman error / TD error |

所以 Bellman 方程不是“直接给了真值标签”，而是给了一个判断价值函数是否合理的条件：

$$
\hat v
\approx
T_\pi\hat v.
$$

---

## 2. 从 true error 到 Bellman error

理想目标是：

$$
J_E(w)
=
\left\|
\hat v(w)-v_\pi
\right\|^2.
$$

但 $v_\pi$ 不知道，所以不能直接算。

Bellman 思路改成比较：

$$
\hat v(w)
\quad\text{和}\quad
T_\pi\hat v(w).
$$

于是得到 Bellman error 贝尔曼误差：

$$
J_{BE}(w)
=
\left\|
\hat v(w)
-T_\pi\hat v(w)
\right\|^2.
$$

展开读法：

$$
\underbrace{\hat v(s,w)}_{\text{当前估计}}
-
\underbrace{
\mathbb E_\pi
\left[
R_{t+1}
+\gamma\hat v(S_{t+1},w)
\mid S_t=s
\right]
}_{\text{用 Bellman backup 造出来的目标}}
$$

这里的 target 不是真实 $v_\pi(s)$，而是：

$$
T_\pi\hat v(s,w)
=
\mathbb E_\pi
\left[
R_{t+1}
+\gamma\hat v(S_{t+1},w)
\mid S_t=s
\right].
$$

所以可以这样理解：

> 既然我不知道真值，那我至少要求当前估计经过一次 Bellman backup 后不要和自己差太远。

---

## 3. 从 Bellman target 到 TD target

Bellman target 里有期望：

$$
\mathbb E_\pi
\left[
R_{t+1}
+\gamma\hat v(S_{t+1},w)
\mid S_t=s
\right].
$$

实际采样时，我们拿到的是一个样本：

$$
(s_t,r_{t+1},s_{t+1}).
$$

于是用单样本近似这个期望：

$$
R_{t+1}
+\gamma\hat v(S_{t+1},w)
\quad\leadsto\quad
r_{t+1}
+\gamma\hat v(s_{t+1},w).
$$

这就是 TD target：

$$
y_t
=
r_{t+1}
+\gamma\hat v(s_{t+1},w_t).
$$

于是 TD error 是：

$$
\delta_t
=
y_t-\hat v(s_t,w_t)
=
r_{t+1}
+\gamma\hat v(s_{t+1},w_t)
-\hat v(s_t,w_t).
$$

这就是从“估计值减真值”变成“估计值减 TD target”的关键链条：

$$
\hat v(s,w)-v_\pi(s)
\quad\text{不可算}
$$

$$
\Downarrow
$$

$$
\hat v(s,w)-T_\pi\hat v(s,w)
\quad\text{Bellman 自洽误差}
$$

$$
\Downarrow
$$

$$
\hat v(s_t,w_t)
-
\left[
r_{t+1}
+\gamma\hat v(s_{t+1},w_t)
\right]
\quad\text{样本 TD 误差}.
$$

符号方向有时写成 target 减估计：

$$
\delta_t
=
\left[
r_{t+1}
+\gamma\hat v(s_{t+1},w_t)
\right]
-\hat v(s_t,w_t).
$$

平方以后方向不重要；做参数更新时方向会影响加号/减号的写法。

---

## 4. 动作值里同样如此

对 action value 动作值，理想目标是：

$$
\hat q(s,a,w)-q_\pi(s,a)
$$

或最优控制时：

$$
\hat q(s,a,w)-q^*(s,a).
$$

但 $q_\pi$ 或 $q^*$ 通常不知道。

### Sarsa 的替代

Sarsa 跟随当前策略采样到的下一动作：

$$
y_t^{\text{Sarsa}}
=
r_{t+1}
+\gamma\hat q(s_{t+1},a_{t+1},w_t).
$$

所以：

$$
\delta_t^{\text{Sarsa}}
=
y_t^{\text{Sarsa}}
-\hat q(s_t,a_t,w_t).
$$

### Q-learning 的替代

Q-learning 用下一状态的最大动作值构造 target：

$$
y_t^{Q}
=
r_{t+1}
+\gamma\max_a\hat q(s_{t+1},a,w_t).
$$

所以：

$$
\delta_t^{Q}
=
y_t^{Q}
-\hat q(s_t,a_t,w_t).
$$

这里对应的是 Bellman optimality equation 贝尔曼最优方程：

$$
q^*(s,a)
=
\mathbb E
\left[
R_{t+1}
+\gamma\max_{a'}q^*(S_{t+1},a')
\mid S_t=s,A_t=a
\right].
$$

同样的思想：

$$
\hat q-q^*
\quad\text{不可算}
$$

$$
\Downarrow
$$

$$
\hat q-\text{Bellman optimality backup}
\quad\text{可用样本近似}.
$$

---

## 5. DQN 里的两个误差为什么不是一回事

DQN 训练时最小化的是：

$$
L_{\text{DQN}}(w)
=
\frac1B
\sum_{i=1}^B
\left(
y_T^{(i)}
-\hat q(s^{(i)},a^{(i)},w)
\right)^2,
$$

其中：

$$
y_T^{(i)}
=
r^{(i)}
+\gamma\max_a\hat q(s'^{(i)},a,w_T).
$$

这是 replay buffer 上的 empirical Bellman/TD loss 经验贝尔曼/TD 损失。

真实误差则是：

$$
E_{\text{value}}(w)
=
\sum_{s,a}
\left(
\hat q(s,a,w)-q^*(s,a)
\right)^2.
$$

两者关系是：

$$
L_{\text{DQN}}(w)
\neq
E_{\text{value}}(w).
$$

原因：

| 量 | 比较对象 | 能不能训练时直接算 |
|---|---|---|
| $L_{\text{DQN}}$ | $\hat q$ 和 Bellman target $y_T$ | 能 |
| $E_{\text{value}}$ | $\hat q$ 和真实 $q^*$ | 通常不能 |

所以：

$$
L_{\text{DQN}}(w)\approx 0
\not\Longrightarrow
E_{\text{value}}(w)\approx 0.
$$

如果 replay buffer 覆盖太少：

$$
\mathcal B\ \text{coverage small}
\quad\Longrightarrow\quad
L_{\text{DQN}}(w)\downarrow
\ \text{but}\ 
E_{\text{value}}(w)\nrightarrow 0.
$$

读法：网络可能只是把 replay buffer 里的样本拟合好了，但没见过或少见的状态-动作对仍然错得很远。

---

## 6. 为什么线性 TD 还要讲 projected Bellman error

前面说：

$$
\hat v\approx T_\pi\hat v.
$$

但在线性函数近似里：

$$
\hat v(w)=\Phi w.
$$

所有能表示的价值函数只在一个子空间里：

$$
\{\Phi w:w\in\mathbb R^m\}.
$$

问题是，Bellman backup 后：

$$
T_\pi(\Phi w)
$$

不一定还在这个子空间里。

所以要先投影回可表示空间：

$$
M T_\pi(\Phi w).
$$

于是 projected Bellman error 投影贝尔曼误差比较的是：

$$
\Phi w
\quad\text{和}\quad
M T_\pi(\Phi w).
$$

目标变成：

$$
J_{PBE}(w)
=
\left\|
\Phi w
-M T_\pi(\Phi w)
\right\|_D^2.
$$

这就是为什么本章说 TD-Linear 实际对应 projected Bellman error，而不是直接最小化：

$$
\left\|
\Phi w-v_\pi
\right\|_D^2.
$$

再压缩成一句：

> 真值误差是最想要的目标；Bellman 误差是用自洽性替代真值；projected Bellman 误差是函数空间表达能力有限时，再把 Bellman backup 投影回来。

---

## 7. 为什么动作值控制部分没展开 projected Bellman error

这个问题容易误会成：

> 是不是 action value 动作值不能讲 projected Bellman error？

不是。动作值也可以讲投影，只要问题是 fixed policy evaluation 固定策略评估。

如果给定策略 $\pi$，学习：

$$
q_\pi(s,a),
$$

并使用线性函数近似：

$$
\hat q(s,a,w)=\phi(s,a)^T w,
$$

把所有 state-action pair 状态-动作对的特征叠起来：

$$
\hat q(w)=\Phi_Qw.
$$

那么动作值 Bellman operator 可以写成：

$$
T_\pi^Q q
=
r+\gamma P_\pi^Q q,
$$

其中 $P_\pi^Q$ 是状态-动作对到下一状态-动作对的转移矩阵。

这时如果 Bellman backup 跑出了可表示空间：

$$
T_\pi^Q(\Phi_Qw)
\notin
\{\Phi_Qw:w\in\mathbb R^m\},
$$

就同样可以投影回来：

$$
\Phi_Q w
\approx
M_QT_\pi^Q(\Phi_Q w).
$$

对应的 projected Bellman error 可以写成：

$$
J_{PBE}^Q(w)
=
\left\|
\Phi_Q w
-M_QT_\pi^Q(\Phi_Q w)
\right\|_{D_Q}^2.
$$

所以，**不是动作值不能有 projected Bellman error**。

第 8.3 没展开它，主要是因为那一节开始讲 control 控制问题，而不是单纯的 policy evaluation 策略评估。

Sarsa 的 target 是：

$$
r+\gamma\hat q(s',a',w).
$$

Q-learning 的 target 是：

$$
r+\gamma\max_{a'}\hat q(s',a',w).
$$

尤其 Q-learning / DQN 对应 Bellman optimality operator：

$$
T_*q
=
r+\gamma\max_{a'}q(s',a').
$$

这个 $\max$ 让算子变成非线性的；再加上控制过程中策略也在变化，所以它不像第 8.2 的固定策略线性评估那样，能干净地推到：

$$
Aw=b
$$

以及：

$$
\Phi w=MT_\pi(\Phi w).
$$

一句话总结：

> projected Bellman error 不是“状态值专属”，而是“固定策略 + 线性函数近似 + 投影理论”最自然；8.3/8.4 进入动作值控制和 DQN 后，重点换成 Bellman optimality target、max、target network 和 replay buffer。

---

## 8. 一条总链路

整章这条思想链可以写成：

$$
\text{想做：}
\quad
\min_w
\left\|
\hat v(w)-v_\pi
\right\|^2
$$

$$
\text{但 }v_\pi\text{ 不知道}
$$

$$
\Downarrow
$$

$$
\text{利用 Bellman fixed point:}
\quad
v_\pi=T_\pi v_\pi
$$

$$
\Downarrow
$$

$$
\text{改成让估计值满足 Bellman 自洽：}
\quad
\hat v(w)\approx T_\pi\hat v(w)
$$

$$
\Downarrow
$$

$$
\text{用样本近似 Bellman target：}
\quad
y_t=r_{t+1}+\gamma\hat v(s_{t+1},w_t)
$$

$$
\Downarrow
$$

$$
\text{训练：}
\quad
\delta_t=y_t-\hat v(s_t,w_t)
$$

动作值和 DQN 只是把这条链里的 $v$ 换成 $q$，并把 target 换成 max 版本：

$$
y_T
=
r+\gamma\max_a\hat q(s',a,w_T).
$$

所以你可以把第 8 章最核心的思想记成：

> 原本想比较“估计值 vs 真值”；真值拿不到，就比较“估计值 vs Bellman backup 造出来的 target”；再用样本把这个 target 落地成 TD/DQN loss。
