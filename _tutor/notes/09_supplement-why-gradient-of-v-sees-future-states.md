# 补充：为什么 $\nabla_\theta v_\pi(s)$ 不是只看当前状态 $s$

>对应第 9 章 Lemma 9.2。主题：图 1 的


$$
v_\pi(s)=\sum_a\pi(a\mid s,\theta)q_\pi(s,a)
 $$

> 图 2 的

$$
\nabla_\theta v_\pi(s)
=
\sum_{s'}\Pr_\pi(s'\mid s)
\sum_a\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a)
$$

到底有什么区别。

---

## 1. 先说最容易误解的点

看到普通值函数公式：

$$
v_\pi(s)
=
\sum_a\pi(a\mid s,\theta)q_\pi(s,a),
$$

很容易产生一个直觉：

> 既然右边只写了当前状态 $s$，那对 $\theta$ 求导时，是不是也只需要看当前状态 $s$ 的策略 $\pi(a\mid s,\theta)$？

这个直觉的问题在于：它忽略了 $q_\pi(s,a)$ 里面也依赖策略参数 $\theta$。

更完整地说：

$$
v_\pi(s)
=
\sum_a
\pi(a\mid s,\theta)
q_\pi(s,a;\theta).
$$

这里我临时把 $q_\pi$ 写成 $q_\pi(s,a;\theta)$，是为了提醒你：

> 动作价值 $q_\pi(s,a)$ 是“先在 $s$ 做动作 $a$，之后继续按策略 $\pi_\theta$ 行动”的长期回报，所以它也会随着 $\theta$ 变化。

原书通常不把这个 $\theta$ 显式写出来，是为了记号简洁；但理解求导时，必须记住它藏在里面。

---

## 2. 图 1：只是在当前状态内做动作平均

图 1 是：

$$
v_\pi(s)
=
\sum_a\pi(a\mid s,\theta)q_\pi(s,a).
$$

它回答的问题是：

> 在当前策略下，状态 $s$ 的价值是多少？

这里的求和只发生在当前状态 $s$ 的动作集合上：

| 部分 | 含义 |
|---|---|
| $s$ | 当前状态，固定住 |
| $a$ | 当前状态下可能选的动作 |
| $\pi(a\mid s,\theta)$ | 当前状态选动作 $a$ 的概率 |
| $q_\pi(s,a)$ | 当前先做 $a$，后面继续按策略走的长期价值 |

所以图 1 的形式确实只显式写了当前状态 $s$。

但是请注意：$q_\pi(s,a)$ 不是“一步奖励”，而是长期价值。它把未来状态已经藏进去了。

---

## 3. 图 2：求导时，未来状态必须显式冒出来

图 2 是：

$$
\nabla_\theta v_\pi(s)
=
\sum_{s'}
\Pr_\pi(s'\mid s)
\sum_a
\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
$$

它回答的问题变了：

> 参数 $\theta$ 改一点点时，从状态 $s$ 出发的长期价值会怎么变？

因为 $v_\pi(s)$ 是长期价值，所以从 $s$ 出发之后，未来会经过很多状态。参数 $\theta$ 改变后，不仅当前状态 $s$ 的动作概率可能变，未来状态 $s'$ 的动作概率也可能变。

所以图 2 需要两层求和：

| 层次 | 求和 | 含义 |
|---|---|---|
| 未来状态层 | $\sum_{s'}$ | 从起点 $s$ 出发，未来可能经过哪些状态 |
| 动作层 | $\sum_a$ | 在每个未来状态 $s'$，动作概率对 $\theta$ 有多敏感 |

这里也要小心：如果完整地回到“无导数的价值计算”，结果应该还是同一个 $v_\pi(s)$。图 1 把未来藏进 $q_\pi(s,a)$，而未来展开会把未来状态 $s'$ 显式写出来。

所以我们现在不是把 $\nabla_\theta\pi$ 简单删掉，而是先抽象出共同的外层骨架：

$$
H(s)
=
\sum_{s'}
\Pr_\pi(s'\mid s)g(s').
$$

这里的 $g(s')$ 可以代表不同的局部贡献：

| 要算什么 | $g(s')$ 代表什么 |
|---|---|
| 普通价值 $v_\pi(s)$ | 未来状态 $s'$ 的奖励/价值贡献 |
| 梯度 $\nabla_\theta v_\pi(s)$ | 未来状态 $s'$ 的局部策略梯度贡献 |

在 Lemma 9.2 里：

$$
g(s')
=
\sum_a\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
$$

读法：

> 从起点 $s$ 出发，未来会以多大折扣权重经过每个 $s'$；每经过一个 $s'$，就把那个状态的局部策略梯度贡献 $g(s')$ 算进来。

---

## 4. 一个真正用到 $\Pr_\pi(s'\mid s)$ 的例子

上一个“一步到终点”的例子太短，$\Pr_\pi(s'\mid s)$ 只会变成 1，看不出它作为访问权重的作用。现在换成一个会循环的两状态例子。

状态有两个：

$$
\mathcal S=\{s_1,s_2\}.
$$

转移和奖励如下：

| 当前状态 | 动作 | 奖励 | 下一状态 |
|---|---|---:|---|
| $s_1$ | only | 0 | $s_2$ |
| $s_2$ | good | 10 | $s_1$ |
| $s_2$ | bad | 0 | $s_1$ |

折扣因子取：

$$
\gamma=0.5.
$$

策略参数只控制 $s_2$ 里选 good 的概率：

$$
\pi(good\mid s_2,\theta)=\theta,
\qquad
\pi(bad\mid s_2,\theta)=1-\theta.
$$

在 $s_1$ 没有选择，只有一个动作 only：

$$
\pi(only\mid s_1,\theta)=1.
$$

注意：无论在 $s_2$ 选 good 还是 bad，下一状态都回到 $s_1$。所以状态转移矩阵不依赖 $\theta$：

$$
P_\pi
=
\left[
\begin{array}{cc}
0 & 1\\
1 & 0
\end{array}
\right].
$$

也就是说，状态会这样来回走：

$$
s_1\to s_2\to s_1\to s_2\to\cdots
$$

---

## 5. 先用图 1 算普通价值 $v_\pi(s_1)$

图 1 说：

$$
v_\pi(s_1)
=
\sum_a\pi(a\mid s_1,\theta)q_\pi(s_1,a).
$$

因为 $s_1$ 只有一个动作 only：

$$
v_\pi(s_1)
=
q_\pi(s_1,only).
$$

写 Bellman 方程更直接。设：

$$
v_1\doteq v_\pi(s_1),
\qquad
v_2\doteq v_\pi(s_2).
$$

从 $s_1$ 出发，奖励是 0，然后去 $s_2$：

$$
v_1=0+\gamma v_2=0.5v_2.
$$

从 $s_2$ 出发，期望即时奖励是：

$$
r_\pi(s_2)
=
10\theta+0(1-\theta)
=
10\theta.
$$

然后回到 $s_1$，所以：

$$
v_2=10\theta+0.5v_1.
$$

代入第一式：

$$
v_1
=
0.5(10\theta+0.5v_1)
=
5\theta+0.25v_1.
$$

移项：

$$
0.75v_1=5\theta,
\qquad
v_1=\frac{20}{3}\theta.
$$

取具体数值：

$$
\theta=0.3.
$$

那么：

$$
v_\pi(s_1)=v_1=\frac{20}{3}\times0.3=2.
$$

这是图 1 / Bellman 方程算出来的普通价值。

---

## 6. 再用 $\Pr_\pi(s'\mid s_1)$ 算同一个 $v_\pi(s_1)$

现在不用图 1 的压缩写法，而是显式展开未来访问状态。

先算折扣总访问权重：

$$
\Pr_\pi(\cdot\mid s_1)
=
\left[(I-\gamma P_\pi)^{-1}\right]_{s_1,\cdot}.
$$

因为：

$$
I-\gamma P_\pi
=
\left[
\begin{array}{cc}
1 & 0\\
0 & 1
\end{array}
\right]
-
0.5
\left[
\begin{array}{cc}
0 & 1\\
1 & 0
\end{array}
\right]
=
\left[
\begin{array}{cc}
1 & -0.5\\
-0.5 & 1
\end{array}
\right].
$$

它的逆是：

$$
(I-\gamma P_\pi)^{-1}
=
\frac{1}{1-0.25}
\left[
\begin{array}{cc}
1 & 0.5\\
0.5 & 1
\end{array}
\right]
=
\left[
\begin{array}{cc}
\frac{4}{3} & \frac{2}{3}\\
\frac{2}{3} & \frac{4}{3}
\end{array}
\right].
$$

所以从 $s_1$ 出发：

$$
\Pr_\pi(s_1\mid s_1)=\frac{4}{3},
\qquad
\Pr_\pi(s_2\mid s_1)=\frac{2}{3}.
$$

这两个数不是概率分布，它们加起来是：

$$
\frac{4}{3}+\frac{2}{3}=2=\frac{1}{1-\gamma}.
$$

现在用它们算普通价值。每个状态的期望即时奖励是：

$$
r_\pi(s_1)=0,
\qquad
r_\pi(s_2)=10\theta.
$$

当 $\theta=0.3$ 时：

$$
r_\pi(s_2)=3.
$$

于是：

$$
\begin{array}{l}
v_\pi(s_1)
=
\Pr_\pi(s_1\mid s_1)r_\pi(s_1)
+
\Pr_\pi(s_2\mid s_1)r_\pi(s_2)\\
=
\frac{4}{3}\times0+\frac{2}{3}\times3\\
=
2.
\end{array}
$$

这和图 1 算出来的 $v_\pi(s_1)=2$ 完全一样。

所以你说得对：

> 如果去掉求导，完整地算普通价值，那么图 1 和未来访问展开应该得到同一个 $v_\pi(s)$。

---

## 7. 最后用 $\Pr_\pi(s'\mid s_1)$ 算梯度

现在才回到 Lemma 9.2：

$$
\nabla_\theta v_\pi(s_1)
=
\sum_{s'}
\Pr_\pi(s'\mid s_1)
\sum_a
\nabla_\theta\pi(a\mid s',\theta)q_\pi(s',a).
$$

外层访问权重已经算出来了：

| 未来状态 $s'$ | $\Pr_\pi(s'\mid s_1)$ |
|---|---:|
| $s_1$ | $\frac{4}{3}$ |
| $s_2$ | $\frac{2}{3}$ |

现在算每个未来状态的局部梯度贡献。

### $s'=s_1$

$s_1$ 只有动作 only，且概率恒为 1：

$$
\nabla_\theta\pi(only\mid s_1,\theta)=0.
$$

所以：

$$
g(s_1)=0.
$$

### $s'=s_2$

在 $s_2$：

$$
\nabla_\theta\pi(good\mid s_2,\theta)=1,
\qquad
\nabla_\theta\pi(bad\mid s_2,\theta)=-1.
$$

两个动作的 $q$ 值是：

$$
q_\pi(s_2,good)=10+\gamma v_\pi(s_1),
$$

$$
q_\pi(s_2,bad)=0+\gamma v_\pi(s_1).
$$

所以 $s_2$ 的局部梯度贡献是：

$$
\begin{array}{l}
g(s_2)
=
1\cdot q_\pi(s_2,good)
+
(-1)\cdot q_\pi(s_2,bad)\\
=
\left(10+\gamma v_\pi(s_1)\right)
-
\left(\gamma v_\pi(s_1)\right)\\
=
10.
\end{array}
$$

最后用 $\Pr_\pi$ 加权：

$$
\begin{array}{l}
\nabla_\theta v_\pi(s_1)
=
\Pr_\pi(s_1\mid s_1)g(s_1)
+
\Pr_\pi(s_2\mid s_1)g(s_2)\\
=
\frac{4}{3}\times0+\frac{2}{3}\times10\\
=
\frac{20}{3}.
\end{array}
$$

这和直接对图 1 算出来的结果一致，因为：

$$
v_\pi(s_1)=\frac{20}{3}\theta
\quad\Rightarrow\quad
\nabla_\theta v_\pi(s_1)=\frac{20}{3}.
$$

---

## 8. 这个例子说明了什么

这个例子把三件事分开了：

| 算什么 | 算法 | 结果 |
|---|---|---:|
| 普通价值 | 图 1 / Bellman 方程 | $v_\pi(s_1)=2$ |
| 普通价值 | 用 $\Pr_\pi(s'\mid s_1)$ 展开未来奖励 | $v_\pi(s_1)=2$ |
| 价值梯度 | 用 Lemma 9.2 的 $\Pr_\pi(s'\mid s_1)$ 加权未来局部梯度 | $\nabla_\theta v_\pi(s_1)=20/3$ |

所以正确理解是：

- 去掉求导，完整计算普通价值，结果应该还是同一个 $v_\pi(s)$。
- $\Pr_\pi(s'\mid s)$ 的作用是告诉我们：从 $s$ 出发，未来各状态 $s'$ 被访问的折扣总权重是多少。
- 求梯度时，不是只看当前状态 $s$，而是把未来状态 $s'$ 的局部策略梯度贡献也按 $\Pr_\pi(s'\mid s)$ 加进来。

---

## 9. 图 1 和图 2 的最终对照

| 对比 | 图 1：普通值函数 | 图 2：值函数梯度 |
|---|---|---|
| 问题 | 状态 $s$ 的价值是多少 | 状态 $s$ 的价值对参数有多敏感 |
| 显式状态 | 当前状态 $s$ | 起点 $s$ 和未来状态 $s'$ |
| 动作求和在哪里 | 当前状态 $s$ 内部 | 每个未来状态 $s'$ 内部 |
| 容易漏掉的东西 | $q_\pi(s,a)$ 里面藏着未来 | 未来状态的策略也依赖 $\theta$ |

一句话总结：

> 图 1 是“当前状态里的动作平均”；图 2 是“从当前状态出发，未来所有状态的策略变化对长期价值的影响总和”。
