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

### 一个数值例子：为什么中括号是“两步到达概率”

假设只有三个状态：

$$
\mathcal S=\{A,B,C\},
$$

从当前状态 $s=A$ 出发，一步转移矩阵为：

$$
P_\pi=
\begin{bmatrix}
0.2 & 0.5 & 0.3\\
0.1 & 0.6 & 0.3\\
0.4 & 0.2 & 0.4
\end{bmatrix}.
$$

第 $A$ 行的意思是：

$$
p_\pi(A\mid A)=0.2,\quad
p_\pi(B\mid A)=0.5,\quad
p_\pi(C\mid A)=0.3.
$$

现在问：从 $A$ 出发，两步后到达 $C$ 的概率是多少？

两步到 $C$ 可以经过三个中间状态：

| 路径 | 概率 |
|---|---:|
| $A\to A\to C$ | $p_\pi(A\mid A)p_\pi(C\mid A)=0.2\times0.3=0.06$ |
| $A\to B\to C$ | $p_\pi(B\mid A)p_\pi(C\mid B)=0.5\times0.3=0.15$ |
| $A\to C\to C$ | $p_\pi(C\mid A)p_\pi(C\mid C)=0.3\times0.4=0.12$ |

把所有可能的中间状态加起来：

$$
\begin{aligned}
[P_\pi^2]_{AC}
&=
\sum_{s_1}p_\pi(s_1\mid A)p_\pi(C\mid s_1)\\
&=
0.2\times0.3+0.5\times0.3+0.3\times0.4\\
&=
0.33.
\end{aligned}
$$

所以中括号

$$
\sum_{s_1}p_\pi(s_1\mid s)p_\pi(s'\mid s_1)
$$

不是一个神秘的新东西，它就是“从 $s$ 出发，走两步，最后落在 $s'$”的总概率。这里的 $s_1$ 只是枚举第一步可能踩到的中间状态。

如果局部项取

$$
u(A)=10,\quad u(B)=20,\quad u(C)=30,
$$

且 $\gamma=0.9$，那么二步项里 $u(C)$ 对 $g(A)$ 的贡献就是：

$$
\gamma^2[P_\pi^2]_{AC}u(C)
=
0.9^2\times0.33\times30
=
8.019.
$$

读法：两步后有 $0.33$ 的概率到达 $C$，到达后拿到局部梯度项 $u(C)=30$，因为这是两步之后的影响，所以还要乘 $\gamma^2$。

再完整看一次矩阵平方。矩阵平方就是：

$$
P_\pi^2=P_\pi P_\pi.
$$

也就是说，$P_\pi^2$ 的每个元素都用“当前行乘目标列”算出来。还是用上面的矩阵：

$$
P_\pi=
\begin{bmatrix}
0.2 & 0.5 & 0.3\\
0.1 & 0.6 & 0.3\\
0.4 & 0.2 & 0.4
\end{bmatrix}.
$$

例如第一行第三列，也就是 $[P_\pi^2]_{AC}$：

$$
\begin{aligned}
[P_\pi^2]_{AC}
&=
\begin{bmatrix}0.2&0.5&0.3\end{bmatrix}
\begin{bmatrix}0.3\\0.3\\0.4\end{bmatrix}\\
&=
0.2\times0.3+0.5\times0.3+0.3\times0.4\\
&=
0.33.
\end{aligned}
$$

如果把所有位置都这样算一遍：

$$
\begin{aligned}
P_\pi^2
&=
\begin{bmatrix}
0.2 & 0.5 & 0.3\\
0.1 & 0.6 & 0.3\\
0.4 & 0.2 & 0.4
\end{bmatrix}
\begin{bmatrix}
0.2 & 0.5 & 0.3\\
0.1 & 0.6 & 0.3\\
0.4 & 0.2 & 0.4
\end{bmatrix}\\
&=
\begin{bmatrix}
0.21 & 0.46 & 0.33\\
0.20 & 0.47 & 0.33\\
0.26 & 0.40 & 0.34
\end{bmatrix}.
\end{aligned}
$$

这一整个新矩阵就是“两步转移矩阵”。比如：

| 元素 | 含义 |
|---|---|
| $[P_\pi^2]_{AA}=0.21$ | 从 $A$ 出发，两步后到 $A$ 的概率 |
| $[P_\pi^2]_{AB}=0.46$ | 从 $A$ 出发，两步后到 $B$ 的概率 |
| $[P_\pi^2]_{AC}=0.33$ | 从 $A$ 出发，两步后到 $C$ 的概率 |

注意第一行加起来仍然是：

$$
0.21+0.46+0.33=1.
$$

这很合理：从 $A$ 出发，两步后一定会落在 $A,B,C$ 里的某一个状态。所以 $P_\pi^2$ 依然是一个转移概率矩阵，只不过它描述的不是“一步后”，而是“两步后”。

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
