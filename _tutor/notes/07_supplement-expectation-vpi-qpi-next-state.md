# 第 7 章补充：为什么 $\mathbb E[q_\pi(S_{t+1},A_{t+1})|s,a]=\mathbb E[v_\pi(S_{t+1})|s,a]$

> 学习日期 2026-06-13 · 对应 7.2-7.4 · 重点：嵌套期望、$v_\pi$ 与 $q_\pi$ 的转换

## 要解决的问题

我们在 7.2 里会反复看到两种写法：

$$
q_\pi(s,a)
=
\mathbb E_\pi[
R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]
$$

和

$$
q_\pi(s,a)
=
\mathbb E_\pi[
R_{t+1}+\gamma v_\pi(S_{t+1})
\mid S_t=s,A_t=a
].
$$

它们为什么等价？关键就是：

$$
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]
=
\mathbb E_\pi[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
].
$$

这篇补充把这件事拆开讲清楚。

---

## 1. 先看随机变量是谁

条件里已经固定了：

$$
S_t=s,\qquad A_t=a.
$$

所以当前状态和当前动作都不是随机的了。

但下一步仍然有两层随机性。

第一层，环境决定下一个状态：

$$
S_{t+1}=s'
\quad
\text{with probability}
\quad
p(s'|s,a).
$$

第二层，到达 $s'$ 后，策略 $\pi$ 决定下一个动作：

$$
A_{t+1}=a'
\quad
\text{with probability}
\quad
\pi(a'|s').
$$

所以：

$$
q_\pi(S_{t+1},A_{t+1})
$$

是一个随机变量，因为里面的 $S_{t+1}$ 和 $A_{t+1}$ 都还没固定。

而：

$$
v_\pi(S_{t+1})
$$

也是一个随机变量，因为 $S_{t+1}$ 还没固定。但它已经把“到达 $S_{t+1}$ 后按 $\pi$ 选动作”的那层平均算进去了。

---

## 2. 一个核心定义：$v_\pi$ 是 $q_\pi$ 的策略加权平均

对任意状态 $s'$，有：

$$
v_\pi(s')
=
\sum_{a'}\pi(a'|s')q_\pi(s',a').
$$

读法：在状态 $s'$，策略 $\pi$ 会以概率 $\pi(a'|s')$ 选择动作 $a'$；每个动作有动作值 $q_\pi(s',a')$；所以状态值就是这些动作值的加权平均。

也可以写成条件期望形式：

$$
v_\pi(s')
=
\mathbb E_\pi[
q_\pi(s',A)
\mid S=s'
].
$$

把 $s'$ 换成随机状态 $S_{t+1}$：

$$
v_\pi(S_{t+1})
=
\sum_{a'}\pi(a'|S_{t+1})q_\pi(S_{t+1},a').
$$

也就是：

$$
v_\pi(S_{t+1})
=
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_{t+1}
].
$$

这句非常重要：

> 给定下一个状态 $S_{t+1}$ 后，对下一动作 $A_{t+1}$ 按策略 $\pi$ 求平均，就得到 $v_\pi(S_{t+1})$。

---

## 3. 先补充：联合概率 $p(s',a'|s,a)$ 是什么

联合概率 joint probability 指的是“两件事同时发生”的概率。

这里的：

$$
p(s',a'|s,a)
$$

意思是：

> 已知当前 $S_t=s,A_t=a$，下一步同时发生 $S_{t+1}=s'$ 且 $A_{t+1}=a'$ 的概率。

也就是：

$$
p(s',a'|s,a)
=
P(S_{t+1}=s',A_{t+1}=a'\mid S_t=s,A_t=a).
$$

为什么它可以拆成：

$$
p(s',a'|s,a)
=
p(s'|s,a)\pi(a'|s')?
$$

用概率乘法公式先写：

$$
P(X,Y|Z)
=
P(X|Z)P(Y|X,Z).
$$

在这里令：

$$
X=(S_{t+1}=s'),\qquad
Y=(A_{t+1}=a'),\qquad
Z=(S_t=s,A_t=a).
$$

于是：

$$
\begin{aligned}
&P(S_{t+1}=s',A_{t+1}=a'\mid S_t=s,A_t=a)\\
&=
P(S_{t+1}=s'\mid S_t=s,A_t=a)\\
&\quad\times
P(A_{t+1}=a'\mid S_{t+1}=s',S_t=s,A_t=a).
\end{aligned}
$$

第一项就是环境转移概率：

$$
P(S_{t+1}=s'\mid S_t=s,A_t=a)
=
p(s'|s,a).
$$

第二项为什么等于策略概率？

因为策略 $\pi$ 在时刻 $t+1$ 只看当前状态 $s'$ 来选动作：

$$
P(A_{t+1}=a'\mid S_{t+1}=s',S_t=s,A_t=a)
=
\pi(a'|s').
$$

这用到了 Markov decision process 马尔可夫决策过程的记号约定：到达 $s'$ 后，动作选择由当前策略在 $s'$ 的分布决定，不再额外依赖上一时刻的 $s,a$。

所以：

$$
\boxed{
p(s',a'|s,a)
=
p(s'|s,a)\pi(a'|s')
}
$$

### 一个小数字例子

假设已知当前：

$$
S_t=s,\qquad A_t=a.
$$

从当前 $(s,a)$ 出发，只走一步，下一状态 $S_{t+1}$ 可能取值为 $x$ 或 $y$：

$$
S_{t+1}\in\{x,y\}.
$$

环境转移概率为：

| 下个状态 $s'$ | $p(s'|s,a)$ |
|---|---:|
| $x$ | 0.6 |
| $y$ | 0.4 |

到达不同的下一状态后，策略 $\pi$ 选择下一动作 $A_{t+1}$ 的概率为：

| 状态 | 动作 $a'$ | $\pi(a'|s')$ |
|---|---|---:|
| $x$ | $L$ | 0.25 |
| $x$ | $R$ | 0.75 |
| $y$ | $L$ | 0.5 |
| $y$ | $R$ | 0.5 |

那么联合概率就是逐项相乘：

| $(s',a')$ | $p(s'|s,a)$ | $\pi(a'|s')$ | $p(s',a'|s,a)$ |
|---|---:|---:|---:|
| $(x,L)$ | 0.6 | 0.25 | 0.15 |
| $(x,R)$ | 0.6 | 0.75 | 0.45 |
| $(y,L)$ | 0.4 | 0.5 | 0.20 |
| $(y,R)$ | 0.4 | 0.5 | 0.20 |

检查一下总和：

$$
0.15+0.45+0.20+0.20=1.
$$

这说明联合分布覆盖了所有可能的“下个状态 + 下个动作”组合。

⚠️ **不要把 $p(s',a'|s,a)$ 理解成环境直接生成动作 $a'$。** 环境只负责从 $(s,a)$ 生成 $s'$ 和奖励；$a'$ 是到达 $s'$ 后由策略 $\pi$ 生成的。联合概率把这两步合在一起描述。

---

## 4. 把右边的期望一步步展开

我们要看：

$$
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
].
$$

因为 $S_{t+1}$ 和 $A_{t+1}$ 都随机，所以先写成二重求和：

$$
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]
=
\sum_{s'}\sum_{a'}
p(s',a'|s,a)
q_\pi(s',a').
$$

联合概率可以拆成：

$$
p(s',a'|s,a)
=
p(s'|s,a)\pi(a'|s').
$$

为什么能这么拆？

因为过程是：

$$
(s,a)
\xrightarrow{\text{environment}}
s'
\xrightarrow{\pi}
a'.
$$

环境先根据 $(s,a)$ 产生 $s'$，策略再根据 $s'$ 产生 $a'$。

代入：

$$
\begin{aligned}
&\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]\\
&=
\sum_{s'}\sum_{a'}
p(s'|s,a)\pi(a'|s')
q_\pi(s',a').
\end{aligned}
$$

把只和 $s'$ 有关的 $p(s'|s,a)$ 提到内层求和外面：

$$
=
\sum_{s'}
p(s'|s,a)
\sum_{a'}
\pi(a'|s')q_\pi(s',a').
$$

而内层：

$$
\sum_{a'}\pi(a'|s')q_\pi(s',a')
$$

正是：

$$
v_\pi(s').
$$

所以：

$$
=
\sum_{s'}p(s'|s,a)v_\pi(s').
$$

这最后一行就是：

$$
\mathbb E_\pi[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
].
$$

因此：

$$
\boxed{
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]
=
\mathbb E_\pi[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
]
}
$$

---

## 5. 用“先内层平均，再外层平均”理解

上面的推导可以用一句话记住：

> 先固定下个状态 $s'$，对下个动作 $a'$ 求平均，$q_\pi(s',a')$ 就变成 $v_\pi(s')$；再对可能的下个状态 $s'$ 求平均。

对应公式：

$$
\underbrace{
\sum_{s'}p(s'|s,a)
}_{\text{外层：平均下个状态}}
\left[
\underbrace{
\sum_{a'}\pi(a'|s')q_\pi(s',a')
}_{\text{内层：平均下个动作}=v_\pi(s')}
\right].
$$

所以：

$$
\sum_{s'}p(s'|s,a)
\sum_{a'}\pi(a'|s')q_\pi(s',a')
=
\sum_{s'}p(s'|s,a)v_\pi(s').
$$

---

## 6. 一个最小数值例子

假设当前固定：

$$
S_t=s,\qquad A_t=a.
$$

从当前 $(s,a)$ 出发，只走一步，下一状态 $S_{t+1}$ 可能取值为 $x$ 或 $y$：

| 下个状态 $s'$ | $p(s'|s,a)$ |
|---|---:|
| $x$ | 0.6 |
| $y$ | 0.4 |

在每个下个状态，策略 $\pi$ 有两个动作 $L,R$。

状态 $x$：

| 动作 | $\pi(a'|x)$ | $q_\pi(x,a')$ |
|---|---:|---:|
| $L$ | 0.25 | 4 |
| $R$ | 0.75 | 8 |

所以：

$$
v_\pi(x)
=
0.25\times4+0.75\times8
=
1+6
=
7.
$$

状态 $y$：

| 动作 | $\pi(a'|y)$ | $q_\pi(y,a')$ |
|---|---:|---:|
| $L$ | 0.5 | 10 |
| $R$ | 0.5 | 2 |

所以：

$$
v_\pi(y)
=
0.5\times10+0.5\times2
=
6.
$$

现在先用 $q_\pi$ 的二重期望算：

$$
\begin{aligned}
&\mathbb E[
q_\pi(S_{t+1},A_{t+1})
\mid s,a
]\\
&=
0.6[
0.25\times4+0.75\times8
]
+
0.4[
0.5\times10+0.5\times2
]\\
&=
0.6\times7+0.4\times6\\
&=
4.2+2.4\\
&=
6.6.
\end{aligned}
$$

再用 $v_\pi$ 算：

$$
\begin{aligned}
&\mathbb E[
v_\pi(S_{t+1})
\mid s,a
]\\
&=
0.6v_\pi(x)+0.4v_\pi(y)\\
&=
0.6\times7+0.4\times6\\
&=
6.6.
\end{aligned}
$$

两个结果一样。

为什么一样？因为 $v_\pi(x)$ 和 $v_\pi(y)$ 本来就是把各自状态下的动作 $q_\pi$ 按策略平均后的结果。

---

## 7. 用全期望公式看

这件事也可以看成 law of total expectation 全期望公式。

全期望公式说：

$$
\mathbb E[X|Y]
=
\mathbb E[
\mathbb E[X|Z,Y]
\mid Y
].
$$

在这里令：

$$
X=q_\pi(S_{t+1},A_{t+1}),
$$

$$
Y=(S_t=s,A_t=a),
$$

$$
Z=S_{t+1}.
$$

那么：

$$
\mathbb E[
q_\pi(S_{t+1},A_{t+1})
\mid S_{t+1}
]
=
v_\pi(S_{t+1}).
$$

所以：

$$
\begin{aligned}
&\mathbb E[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]\\
&=
\mathbb E[
\mathbb E[
q_\pi(S_{t+1},A_{t+1})
\mid S_{t+1}
]
\mid S_t=s,A_t=a
]\\
&=
\mathbb E[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
].
\end{aligned}
$$

这就是同一个证明的更抽象版本。

---

## 8. 另一个思路：把“期望的期望”看成先结算动作随机性

这条思路不替代前面的联合概率展开，只是换一个读法。

先写 $v_\pi$ 的定义：

$$
v_\pi(S_{t+1})
=
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_{t+1}
].
$$

把它放进外层期望：

$$
\mathbb E_\pi[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
]
$$

可以读成：

> 到了某个下个状态 $S_{t+1}$ 以后，先把下一动作 $A_{t+1}$ 的随机性按策略 $\pi$ 平均掉。

所以：

$$
\mathbb E_\pi[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
]
$$

可以读成：

> 先对动作求平均得到 $v_\pi(S_{t+1})$，再对下个状态 $S_{t+1}$ 求平均。

而：

$$
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]
$$

可以读成：

> 不提前结算，直接对“下个状态 + 下个动作”一起求平均。

根据全期望公式，这两种平均顺序等价：

$$
\boxed{
\mathbb E_\pi[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
]
=
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]
}
$$

直觉版：

$$
\text{先平均动作，再平均状态}
=
\text{直接平均状态和动作}.
$$

⚠️ 这不是把一个期望随便删掉；是内层期望已经完整结算了动作随机性，所以外层只剩下状态随机性。

---

## 9. 它在 Bellman 方程里怎么用

动作值定义可以写成：

$$
q_\pi(s,a)
=
\mathbb E_\pi[
R_{t+1}
+\gamma q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
].
$$

因为刚才证明了：

$$
\mathbb E_\pi[
q_\pi(S_{t+1},A_{t+1})
\mid S_t=s,A_t=a
]
=
\mathbb E_\pi[
v_\pi(S_{t+1})
\mid S_t=s,A_t=a
].
$$

所以也可以写成：

$$
q_\pi(s,a)
=
\mathbb E_\pi[
R_{t+1}
+\gamma v_\pi(S_{t+1})
\mid S_t=s,A_t=a
].
$$

两种写法都对：

| 写法 | 强调什么 |
|---|---|
| $R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})$ | 把下一动作 $A_{t+1}$ 显式写出来，适合理解 Sarsa |
| $R_{t+1}+\gamma v_\pi(S_{t+1})$ | 把下一动作按策略平均掉，适合理解 Expected Sarsa 和 Bellman 展开 |

---

## 10. 易错点

- 不要把 $v_\pi(S_{t+1})$ 理解成某个固定数字。它仍然是随机变量，因为 $S_{t+1}$ 随机。
- $v_\pi(s')=\sum_{a'}\pi(a'|s')q_\pi(s',a')$ 是在固定状态 $s'$ 后，对动作求平均。
- $\mathbb E[v_\pi(S_{t+1})|s,a]$ 是对下个状态 $S_{t+1}$ 求平均。
- $\mathbb E[q_\pi(S_{t+1},A_{t+1})|s,a]$ 是先对下个动作求平均，再对下个状态求平均。
- 两边相等，不是因为 $q_\pi(S_{t+1},A_{t+1})$ 和 $v_\pi(S_{t+1})$ 每次采样都相等；它们通常不相等。相等的是条件期望。
