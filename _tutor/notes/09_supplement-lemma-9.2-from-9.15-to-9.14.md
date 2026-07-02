# 补充：Lemma 9.2 从 (9.15) 到 (9.14) 怎么算

> 对应原书 9.3 / Lemma 9.2。主题：把

$$
 \nabla_\theta v_\pi(s)
=
 \sum_a
 \left[
 \nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a)
 +
 \pi(a\mid s,\theta)\nabla_\theta q_\pi(s,a)
 \right]
 $$
逐步化成

 $$
 \nabla_\theta v_\pi(s)
 =
 \sum_{s'\in\mathcal S}
 \Pr_\pi(s'\mid s)
 \sum_{a\in\mathcal A}
 \nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
 $$

---

## 1. 先把 $\nabla_\theta q_\pi(s,a)$ 展开

折扣情形下，动作价值满足 Bellman 展开：

$$
q_\pi(s,a)
=
r(s,a)
+
\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

环境奖励 $r(s,a)$ 不依赖 $\theta$，所以：

$$
\nabla_\theta q_\pi(s,a)
=
\gamma\sum_{s'}p(s'\mid s,a)\nabla_\theta v_\pi(s').
$$

把它代回 (9.15)：

$$
\begin{array}{l}
\nabla_\theta v_\pi(s)
=
\sum_a
\left[
\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a)
+
\pi(a\mid s,\theta)
\gamma\sum_{s'}p(s'\mid s,a)\nabla_\theta v_\pi(s')
\right] \\[4pt]
=
\sum_a
\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a)
+
\gamma
\sum_a
\pi(a\mid s,\theta)
\sum_{s'}p(s'\mid s,a)\nabla_\theta v_\pi(s').
\end{array}
$$

## 2. 把动作求和合成 $p_\pi(s'\mid s)$

第二项里有两个求和：先对动作 $a$ 求和，再对下一个状态 $s'$ 求和。有限和可以交换顺序：

$$
\gamma
\sum_a
\pi(a\mid s,\theta)
\sum_{s'}p(s'\mid s,a)\nabla_\theta v_\pi(s')
=
\gamma
\sum_{s'}
\left[
\sum_a
\pi(a\mid s,\theta)p(s'\mid s,a)
\right]
\nabla_\theta v_\pi(s').
$$

中括号里的东西是 policy-induced transition probability 策略诱导的状态转移概率：

$$
p_\pi(s'\mid s)
\doteq
\sum_a
\pi(a\mid s,\theta)p(s'\mid s,a).
$$

意思是：在状态 $s$，先按策略 $\pi$ 随机选动作 $a$，再按环境概率 $p(s'\mid s,a)$ 转移到 $s'$；对所有动作路径求和，就得到从 $s$ 到 $s'$ 的总转移概率。

因此第二项变成：

$$
\gamma
\sum_{s'}
p_\pi(s'\mid s)
\nabla_\theta v_\pi(s').
$$

于是：

$$
\nabla_\theta v_\pi(s)
=
\underbrace{
\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a)
}_{u(s):\ \text{当前状态的局部策略梯度项}}
+
\gamma\sum_{s'}p_\pi(s'\mid s)\nabla_\theta v_\pi(s'). \tag{9.16}
$$

严格对照原书，(9.16) 先写成：

$$
\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a)
+
\gamma
\sum_a\pi(a\mid s,\theta)\sum_{s'}p(s'\mid s,a)\nabla_\theta v_\pi(s')
$$

然后再把第二项简写成：

$$
\gamma\sum_{s'}[P_\pi]_{ss'}\nabla_\theta v_\pi(s').
$$

这里：

$$
[P_\pi]_{ss'}=p_\pi(s'\mid s).
$$

## 3. 把递推式展开成折扣访问权重

令：

$$
g(s)\doteq\nabla_\theta v_\pi(s),
\qquad
u(s)\doteq\sum_a\nabla_\theta\pi(a\mid s,\theta)q_\pi(s,a).
$$

那么 (9.16) 就是：

$$
g(s)
=
u(s)
+
\gamma\sum_{s_1}p_\pi(s_1\mid s)g(s_1).
$$

右边还有 $g(s_1)$，继续用同一个公式展开：

$$
g(s_1)
=
u(s_1)
+
\gamma\sum_{s_2}p_\pi(s_2\mid s_1)g(s_2).
$$

代回去：

$$
\begin{array}{l}
g(s)
=
u(s)
+
\gamma\sum_{s_1}p_\pi(s_1\mid s)u(s_1) \\
\qquad
+
\gamma^2
\sum_{s_1}\sum_{s_2}
p_\pi(s_1\mid s)p_\pi(s_2\mid s_1)g(s_2).
\end{array}
$$

注意上式最后一项里还是 $g(s_2)$，所以还没有变成只含 $u$ 的级数。再展开一次：

$$
g(s_2)
=
u(s_2)
+
\gamma\sum_{s_3}p_\pi(s_3\mid s_2)g(s_3).
$$

代入最后一项：

$$
\begin{array}{l}
g(s)
=
u(s)
+
\gamma\sum_{s_1}p_\pi(s_1\mid s)u(s_1) \\
\qquad
+
\gamma^2
\sum_{s_1}\sum_{s_2}
p_\pi(s_1\mid s)p_\pi(s_2\mid s_1)u(s_2) \\
\qquad
+
\gamma^3
\sum_{s_1}\sum_{s_2}\sum_{s_3}
p_\pi(s_1\mid s)p_\pi(s_2\mid s_1)p_\pi(s_3\mid s_2)g(s_3).
\end{array}
$$

现在看二步项：

$$
\gamma^2
\sum_{s_1}\sum_{s_2}
p_\pi(s_1\mid s)p_\pi(s_2\mid s_1)u(s_2).
$$

这里 $s_2$ 是“两步后到达的状态”。为了和后面统一，把它重命名为 $s'$：

$$
\gamma^2
\sum_{s'}\left[
\sum_{s_1}
p_\pi(s_1\mid s)p_\pi(s'\mid s_1)
\right]u(s').
$$

中括号里的量就是从 $s$ 出发两步到 $s'$ 的概率。矩阵乘法正是这样定义的：

$$
[P_\pi^2]_{ss'}
=
\sum_{s_1}[P_\pi]_{ss_1}[P_\pi]_{s_1s'}
=
\sum_{s_1}p_\pi(s_1\mid s)p_\pi(s'\mid s_1).
$$

所以二步项可以写成：

$$
\gamma^2\sum_{s'}[P_\pi^2]_{ss'}u(s').
$$

同理，一步项也可以统一写成：

$$
\gamma\sum_{s_1}p_\pi(s_1\mid s)u(s_1)
=
\gamma\sum_{s'}[P_\pi]_{ss'}u(s').
$$

三步项如果再展开一次，会出现：

$$
\gamma^3
\sum_{s'}[P_\pi^3]_{ss'}u(s'),
$$

因为

$$
[P_\pi^3]_{ss'}
=
\sum_{s_1}\sum_{s_2}
p_\pi(s_1\mid s)p_\pi(s_2\mid s_1)p_\pi(s'\mid s_2).
$$

所以展开一次、两次、无限次，得到：

$$
g(s)
=
u(s)
+
\gamma\sum_{s'}[P_\pi]_{ss'}u(s')
+
\gamma^2\sum_{s'}[P_\pi^2]_{ss'}u(s')
+
\cdots.
$$

把同一个 $u(s')$ 的系数收集起来：

$$
g(s)
=
\sum_{s'}
\left(
[I]_{ss'}
+
\gamma[P_\pi]_{ss'}
+
\gamma^2[P_\pi^2]_{ss'}
+
\cdots
\right)
u(s').
$$

括号里的系数就是 discounted total probability 折扣总访问权重：

$$
\Pr_\pi(s'\mid s)
\doteq
\sum_{k=0}^{\infty}\gamma^k[P_\pi^k]_{ss'}
=
\left[(I-\gamma P_\pi)^{-1}\right]_{ss'}.
$$

所以：

$$
g(s)
=
\sum_{s'}\Pr_\pi(s'\mid s)u(s').
$$

最后把 $g(s)$ 和 $u(s')$ 换回原来的符号：

$$
\nabla_\theta v_\pi(s)
=
\sum_{s'\in\mathcal S}
\Pr_\pi(s'\mid s)
\sum_{a\in\mathcal A}
\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a). \tag{9.14}
$$

一句话：$\nabla_\theta q_\pi$ 不是消失了，而是通过 Bellman 递推变成未来状态的 $\nabla_\theta v_\pi(s')$；递推展开后，未来每个状态的局部项 $u(s')$ 被折扣访问权重 $\Pr_\pi(s'\mid s)$ 加权汇总。
