# 补充：随机策略梯度与确定性策略梯度公式来源

> 对应第 10 章 10.4。本文只解释两件事：
>
> 1. 随机策略梯度里的期望公式从哪里来。
> 2. 确定性策略梯度为什么是“动作梯度乘 critic 对动作的梯度”。

## 0. 统一记号

为了避免价值函数记号来回切换，这里统一用下标 $\theta$ 表示“当前参数诱导出的策略”。

| 场景 | 策略 | 价值函数记号 |
|---|---|---|
| 随机策略 | $\pi_\theta(a\mid s)$ | $q_\theta(s,a)=q_{\pi_\theta}(s,a)$ |
| 确定性策略 | $a=\mu_\theta(s)$ | $q_\theta(s,a)=q_{\mu_\theta}(s,a)$ |

也就是说，$q_\theta(s,a)$ 不是一个固定函数。它表示“从 $(s,a)$ 出发，之后继续执行当前策略参数 $\theta$ 时的长期价值”。因此 $q_\theta$ 本身依赖 $\theta$。

---

## 1. 随机策略梯度公式从哪里来？

第 9 章 policy gradient theorem 的动作层求和形式是：

$$
\nabla_\theta J(\theta)
=
\sum_s\eta(s)\sum_a
\nabla_\theta\pi_\theta(a\mid s)q_\theta(s,a).
\tag{1}
$$

这里 $\eta(s)$ 是状态分布。它具体是 $\rho_\theta$ 还是 $d_\theta$，不影响下面的动作层转换。

关键一步是 log-trick：

$$
\nabla_\theta\pi_\theta(a\mid s)
=
\pi_\theta(a\mid s)\nabla_\theta\ln\pi_\theta(a\mid s).
\tag{2}
$$

把 (2) 代入 (1)：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=
\sum_s\eta(s)\sum_a
\pi_\theta(a\mid s)
\nabla_\theta\ln\pi_\theta(a\mid s)q_\theta(s,a).
\end{aligned}
$$

这正是期望形式。因为如果

$$
s\sim\eta,\qquad a\sim\pi_\theta(\cdot\mid s),
$$

则：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{s\sim\eta,\ a\sim\pi_\theta}
\left[
\nabla_\theta\ln\pi_\theta(a\mid s)q_\theta(s,a)
\right].
}
\tag{3}
$$

所以随机策略公式的来源可以压缩成：

$$
\boxed{
\nabla_\theta\pi_\theta
\xrightarrow{\text{log-trick}}
\pi_\theta\nabla_\theta\ln\pi_\theta
\xrightarrow{\text{写成采样期望}}
\mathbb E_{s,a}[\nabla_\theta\ln\pi_\theta(a\mid s)q_\theta(s,a)]
}
$$

---

## 2. 为什么确定性策略不能直接代入 $a=\mu_\theta(s)$？

随机策略里，参数 $\theta$ 改的是动作概率：

$$
\pi_\theta(a\mid s).
$$

所以梯度工具是：

$$
\nabla_\theta\ln\pi_\theta(a\mid s).
$$

确定性策略里，参数 $\theta$ 改的是动作本身：

$$
a=\mu_\theta(s).
$$

它没有 $\ln\pi_\theta(a\mid s)$ 这个概率项。因此 deterministic policy gradient 不是把 (3) 里的 $a$ 直接替换成 $\mu_\theta(s)$，而是从确定性策略自己的价值关系开始。

---

## 3. 确定性策略的求导路径

确定性策略下：

$$
v_\theta(s)=q_\theta(s,\mu_\theta(s)).
\tag{4}
$$

为看清楚求导路径，把它想成：

$$
v_\theta(s)=q_\theta(s,a)\big|_{a=\mu_\theta(s)}.
$$

依赖树是：

```text
theta
├─ changes q_theta(s, a) itself
│  └─ gives (∇_theta q_theta(s,a)) |_{a=mu_theta(s)}
└─ changes current action a = mu_theta(s)
   └─ then changes q_theta(s,a)
      └─ gives ∇_theta mu_theta(s) · (∇_a q_theta(s,a)) |_{a=mu_theta(s)}
```

所以链式法则给出：

$$
\nabla_\theta v_\theta(s)
=
\left(\nabla_\theta q_\theta(s,a)\right)\big|_{a=\mu_\theta(s)}
+
\nabla_\theta\mu_\theta(s)\cdot
\left(\nabla_a q_\theta(s,a)\right)\big|_{a=\mu_\theta(s)}.
\tag{5}
$$

第二项是相乘。若 $a\in\mathbb R^d,\theta\in\mathbb R^m$，可以按列向量约定理解为：

$$
\nabla_\theta\mu_\theta(s)\in\mathbb R^{m\times d},
\qquad
\nabla_a q_\theta(s,a)\in\mathbb R^d,
$$

所以：

$$
\nabla_\theta\mu_\theta(s)\cdot\nabla_a q_\theta(s,a)\in\mathbb R^m.
$$

原书把第二项记成：

$$
\boxed{
u_\theta(s)
\doteq
\nabla_\theta\mu_\theta(s)\cdot
\left(\nabla_a q_\theta(s,a)\right)\big|_{a=\mu_\theta(s)}.
}
\tag{6}
$$

区分定理推导和算法更新：

```text
Theorem proof:
theta -> q_theta(s,a) itself      gives first term
theta -> mu_theta(s) -> q_theta   gives second term u_theta(s)

Algorithm actor update:
fix critic q(s,a,w_t)
theta -> mu_theta(s) -> q(s,a,w_t)
only uses the second term
```

---

## 4. 第一项怎么处理：Bellman 展开

现在处理式 (5) 的第一项：

$$
\left(\nabla_\theta q_\theta(s,a)\right)\big|_{a=\mu_\theta(s)}.
$$

动作价值满足 Bellman 关系：

$$
q_\theta(s,a)
=
r(s,a)
+
\gamma\sum_{s'}p(s'\mid s,a)v_\theta(s').
\tag{7}
$$

环境的 $r(s,a)$ 和 $p(s'\mid s,a)$ 不直接依赖 $\theta$，所以：

$$
\nabla_\theta q_\theta(s,a)
=
\gamma\sum_{s'}p(s'\mid s,a)\nabla_\theta v_\theta(s').
\tag{8}
$$

代入 $a=\mu_\theta(s)$：

$$
\left(\nabla_\theta q_\theta(s,a)\right)\big|_{a=\mu_\theta(s)}
=
\gamma\sum_{s'}p(s'\mid s,\mu_\theta(s))\nabla_\theta v_\theta(s').
\tag{9}
$$

把 (9) 放回 (5)：

$$
\nabla_\theta v_\theta(s)
=
u_\theta(s)
+
\gamma\sum_{s'}p(s'\mid s,\mu_\theta(s))\nabla_\theta v_\theta(s').
\tag{10}
$$

读法：

$$
\boxed{
\text{当前价值梯度}
=
\text{当前动作路径 }u_\theta(s)
+
\text{未来状态价值梯度的传播}
}
$$

---

## 5. 展开递归，得到 Lemma 10.1

令 $P_\theta$ 表示确定性策略 $\mu_\theta$ 下的状态转移矩阵：

$$
[P_\theta]_{ss'}=p(s'\mid s,\mu_\theta(s)).
$$

把所有状态的式 (10) 合起来：

$$
\nabla_\theta v_\theta
=
u_\theta+\gamma P_\theta\nabla_\theta v_\theta.
$$

解这个线性递归：

$$
\begin{aligned}
\nabla_\theta v_\theta
&=
(I-\gamma P_\theta)^{-1}u_\theta\\
&=
\left(I+\gamma P_\theta+\gamma^2P_\theta^2+\cdots\right)u_\theta.
\end{aligned}
$$

第 $s$ 个分量为：

$$
\nabla_\theta v_\theta(s)
=
\sum_{s'}
\left[(I-\gamma P_\theta)^{-1}\right]_{ss'}u_\theta(s').
$$

定义折扣总转移权重：

$$
\Pr_\theta(s'\mid s)
\doteq
\left[(I-\gamma P_\theta)^{-1}\right]_{ss'}
=
\sum_{k=0}^{\infty}\gamma^k[P_\theta^k]_{ss'}.
$$

于是：

$$
\boxed{
\nabla_\theta v_\theta(s)
=
\sum_{s'}\Pr_\theta(s'\mid s)
\nabla_\theta\mu_\theta(s')\cdot
\left(\nabla_a q_\theta(s',a)\right)\big|_{a=\mu_\theta(s')}.
}
\tag{11}
$$

这就是 Lemma 10.1 的内容。

---

## 6. 从 Lemma 到 deterministic policy gradient theorem

若目标是折扣平均价值：

$$
J(\theta)=\sum_s d_0(s)v_\theta(s),
$$

则：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=
\sum_s d_0(s)\nabla_\theta v_\theta(s)\\
&=
\sum_s d_0(s)
\sum_{s'}\Pr_\theta(s'\mid s)u_\theta(s')\\
&=
\sum_{s'}
\underbrace{
\left[
\sum_s d_0(s)\Pr_\theta(s'\mid s)
\right]
}_{\rho_\theta(s')}
u_\theta(s').
\end{aligned}
$$

因此：

$$
\boxed{
\nabla_\theta J(\theta)
=
\sum_s\rho_\theta(s)
\nabla_\theta\mu_\theta(s)\cdot
\left(\nabla_a q_\theta(s,a)\right)\big|_{a=\mu_\theta(s)}
}
$$

也就是：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{s\sim\rho_\theta}
\left[
\nabla_\theta\mu_\theta(s)\cdot
\left(\nabla_a q_\theta(s,a)\right)\big|_{a=\mu_\theta(s)}
\right].
}
\tag{12}
$$

平均奖励目标时，把状态分布 $\rho_\theta$ 换成平稳分布 $d_\theta$，局部梯度形状不变。

---

## 7. 最后对比

| 策略 | 参数改变什么 | 梯度形状 |
|---|---|---|
| 随机策略 $\pi_\theta(a\mid s)$ | 改动作概率 | $\mathbb E_{s,a}[\nabla_\theta\ln\pi_\theta(a\mid s)q_\theta(s,a)]$ |
| 确定性策略 $\mu_\theta(s)$ | 改动作本身 | $\mathbb E_s[\nabla_\theta\mu_\theta(s)\cdot(\nabla_a q_\theta(s,a))\vert_{a=\mu_\theta(s)}]$ |

一句话：

$$
\boxed{
\text{随机策略用 log-trick 调概率；确定性策略用链式法则调动作。}
}
$$
