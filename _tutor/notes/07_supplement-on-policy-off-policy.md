# 第 7 章补充：on-policy 与 off-policy 怎么理解

> 学习日期 2026-06-14 · 对应 7.2-7.7 · 重点：behavior policy、target policy、Sarsa、Q-learning

## 要解决的问题

第 7 章里最容易混的是：

> Sarsa 明明也在更新策略，为什么叫 on-policy？Q-learning 明明也可以用 $\epsilon$-greedy 采样，为什么原书又说它有 on-policy version 和 off-policy version？

关键不是“有没有探索”，也不是“是不是在线更新”，而是看两种策略是不是同一个：

- behavior policy 行为策略：负责生成经验样本的策略。
- target policy 目标策略：算法真正想评估或学到的策略。

一句话判断：

$$
\boxed{
\text{behavior policy}=\text{target policy}
\quad\Longrightarrow\quad
\text{on-policy}
}
$$

$$
\boxed{
\text{behavior policy}\neq\text{target policy}\text{ 也可以学习}
\quad\Longrightarrow\quad
\text{off-policy}
}
$$

---

## 1. 先分清两个“策略”

### behavior policy 行为策略

behavior policy 是真正用来和环境互动、生成样本的策略。

它决定：

$$
A_t\sim \pi_b(\cdot|S_t).
$$

所以经验序列：

$$
S_t,A_t,R_{t+1},S_{t+1},A_{t+1},\ldots
$$

是由 $\pi_b$ 采样出来的。

### target policy 目标策略

target policy 是你真正想评估或学到的策略。

如果你想估计 $q_\pi$，那么 target policy 就是 $\pi$。

如果你想学最优策略，那么 target policy 通常是 greedy policy：

$$
\pi_T(s)
=
\arg\max_a q(s,a).
$$

注意：目标策略不一定负责采样。它可以只是算法心里真正想学的那个策略。

---

## 2. on-policy：我用谁采样，就学谁

on-policy 同策略的意思是：

$$
\pi_b=\pi_T.
$$

也就是：

> 我当前用什么策略行动，就用样本去学习这个策略自己的价值。

Sarsa 就是典型 on-policy。

Sarsa target 是：

$$
r_{t+1}
+
\gamma q_t(s_{t+1},a_{t+1}).
$$

这里的 $a_{t+1}$ 是当前行为策略实际采样出来的动作：

$$
a_{t+1}\sim \pi_t(\cdot|s_{t+1}).
$$

所以 Sarsa 在说：

> 既然我当前策略下一步真的可能选 $a_{t+1}$，那我就把这个动作的后果算进价值里。

这就是 on-policy 的味道：学习结果反映当前策略真实会经历的路径，包括探索动作带来的好坏。

---

## 3. Sarsa 为什么是 on-policy

Sarsa 的一次更新流程可以写成：

$$
s_t
\xrightarrow[\pi_t]{\text{采样}}
a_t
\to
r_{t+1},s_{t+1}
\xrightarrow[\pi_t\text{ 或更新后的 }\pi_{t+1}]{\text{采样}}
a_{t+1}.
$$

然后用：

$$
q_{t+1}(s_t,a_t)
=
q_t(s_t,a_t)
+
\alpha
\left[
r_{t+1}
+
\gamma q_t(s_{t+1},a_{t+1})
-
q_t(s_t,a_t)
\right].
$$

关键在 $a_{t+1}$：

$$
a_{t+1}
\text{ 是当前策略实际选出来的动作。}
$$

所以 Sarsa 学的是当前策略自己的动作值：

$$
q_{\pi_t}.
$$

如果当前策略是 $\epsilon$-greedy，那么 Sarsa 学到的价值就会包含 $\epsilon$-greedy 的探索风险。

---

## 4. 一个小例子：Sarsa 会把探索动作算进去

假设到了下一个状态 $B$，当前估计为：

| 动作 | $q_t(B,a)$ |
|---|---:|
| Left | 2 |
| Right | 8 |

如果当前 $\epsilon$-greedy 策略这次探索到了 Left，那么：

$$
a_{t+1}=\text{Left}.
$$

Sarsa target 是：

$$
r_{t+1}
+
\gamma q_t(B,\text{Left})
=
r_{t+1}
+
2\gamma.
$$

它不会说“Right 更大，所以我改用 Right”。因为 Sarsa 的 target 尊重实际采样动作。

这说明 Sarsa 学的是：

> 当前这条带探索的策略实际会得到什么价值。

所以它是 on-policy。

---

## 5. off-policy：用一种策略采样，学另一种策略

off-policy 异策略的意思是：

$$
\pi_b\neq\pi_T
$$

也可以学习。

最典型的是 Q-learning。原书 7.4.2 先给总判断：Q-learning 是 off-policy；7.4.3 再说它可以用 on-policy 或 off-policy 两种方式实现。

Q-learning 的 target 是：

$$
r_{t+1}
+
\gamma
\max_{a'}q_t(s_{t+1},a').
$$

注意它不用实际采样出来的 $a_{t+1}$。

它只问：

$$
\text{到了 }s_{t+1}\text{ 后，哪个候选动作 }a'\text{ 的当前估计最大？}
$$

在这个 off-policy 视角里，Q-learning 的 target policy 是 greedy policy：

$$
\pi_T(s)
=
\arg\max_a q_t(s,a).
$$

但 behavior policy 可以是另一个足够探索的策略，比如 $\epsilon$-greedy，或者原书 Figure 7.4 里的 uniform policy 均匀策略：

$$
\pi_b=\epsilon\text{-greedy}
\quad\text{or}\quad
\pi_b=\text{uniform}.
$$

于是：

$$
\pi_b\neq\pi_T.
$$

这就是原书说 Q-learning 是 off-policy 的根本原因：它解的是 Bellman optimality equation，target 不依赖实际采样出来的 $a_{t+1}$。

但这里要加一个原书里的精细区分：

> Q-learning 的更新目标本身指向最优动作值；具体实现时，原书又给了 on-policy version 和 off-policy version 两种。

也就是说，不要把这两句话混成一句：

| 说法 | 含义 |
|---|---|
| Q-learning 是 off-policy | 这是 7.4.2 的总判断：它求 Bellman optimality equation，不求某个给定策略的 Bellman equation |
| Q-learning 的 on-policy version | Algorithm 7.2：行为策略和当前更新的策略都用 $\epsilon$-greedy，但 TD target 仍然用 $\max_{a'}q(s',a')$ |
| Q-learning 的 off-policy version | Algorithm 7.3：行为策略可以是 uniform 等强探索策略，目标策略是 greedy |

所以你说的“用均匀采样才是 off-policy”很接近原书的例子：原书 Figure 7.4 的确用 uniform behavior policy 展示 off-policy Q-learning。

---

## 6. 同一个例子看 Q-learning

还是状态 $B$：

| 动作 | $q_t(B,a)$ |
|---|---:|
| Left | 2 |
| Right | 8 |

即使行为策略这次实际采样到了：

$$
a_{t+1}=\text{Left},
$$

Q-learning target 仍然是：

$$
r_{t+1}
+
\gamma\max\{2,8\}
=
r_{t+1}
+
8\gamma.
$$

它直接使用 Right 的价值，因为 Right 是当前估计下的最大动作。

这说明 Q-learning 学的不是“当前探索策略实际会怎样”，而是：

> 如果以后都按当前估计的最优动作走，价值会怎样。

如果这个探索策略和目标策略分开，比如用 uniform policy 采样、用 greedy policy 作为目标策略，那么这就是 off-policy Q-learning。

如果行为策略也就是当前的 $\epsilon$-greedy 策略，那么它就是原书 Algorithm 7.2 的 on-policy version。注意：即使是这个版本，Q-learning 的 TD target 仍然不是 Sarsa target，它仍然用 $\max_{a'}q(s',a')$。所以这里的 “on-policy/off-policy version” 主要说的是实现时行为策略和目标策略是否一致，不是在改 Q-learning 的核心 target。

---

## 7. 各算法放在一张表里

| 算法 | behavior policy 行为策略 | target policy 目标策略 | on/off-policy | 原因 |
|---|---|---|---|---|
| TD state-value prediction | 给定 $\pi$ | 同一个 $\pi$ | on-policy | 用 $\pi$ 的样本估计 $v_\pi$ |
| Sarsa | 当前 $\epsilon$-greedy 策略 | 同一个当前策略 | on-policy | target 用实际采样的 $a_{t+1}$ |
| Expected Sarsa | 通常是当前策略 | 通常是同一个当前策略 | 通常 on-policy | target 用 $\sum_{a'}\pi(a'|s')q(s',a')$，若这里的 $\pi$ 就是行为策略，则 on-policy |
| $n$-step Sarsa | 当前策略 | 同一个当前策略 | on-policy | 多步轨迹中的动作来自当前策略 |
| Monte Carlo prediction | 给定 $\pi$ | 同一个 $\pi$ | on-policy | 用 $\pi$ 生成的完整回报估计 $v_\pi$ 或 $q_\pi$ |
| Q-learning on-policy version | 当前 $\epsilon$-greedy 策略 | 同一个 $\epsilon$-greedy 策略 | on-policy implementation | 原书 Algorithm 7.2 |
| Q-learning off-policy version | 另一个探索策略，如 uniform policy | greedy policy | off-policy implementation | 原书 Algorithm 7.3 |

⚠️ Expected Sarsa 有一个细节：它本身既可以写成 on-policy，也可以写成 off-policy。关键看求和里的那个策略是谁：

$$
\sum_{a'}\pi(a'|s')q(s',a').
$$

如果这个 $\pi$ 就是生成样本的行为策略，那是 on-policy。

如果样本来自 $\pi_b$，但求和用的是另一个目标策略 $\pi_T$，那就是 off-policy Expected Sarsa。

第 7 章这里主要把它作为 Sarsa 的低方差变体来理解，所以先按 on-policy 理解最自然。

---

## 8. Q-learning 的 on-policy version 与 off-policy version

原书 7.4.3 说得很明确：

> Since Q-learning is off-policy, it can be implemented in either an on-policy or off-policy fashion.

这句话的意思是：Q-learning 的更新式不依赖实际下一动作 $a_{t+1}$，因此它有能力使用别的行为策略产生的数据；但你具体跑算法时，也可以让行为策略和当前策略保持一致。

### Algorithm 7.2：on-policy version

这一版中：

$$
\pi_b=\pi_T=\epsilon\text{-greedy}.
$$

智能体用当前 $\epsilon$-greedy 策略采样：

$$
a_t\sim \pi_t(\cdot|s_t).
$$

更新完 $q$ 后，又把当前状态的策略更新成新的 $\epsilon$-greedy：

$$
\pi_{t+1}=\epsilon\text{-greedy}(q_{t+1}).
$$

所以在“谁生成样本、谁被更新”这个意义上，它是 on-policy。

但是它的 Q-learning target 仍然是：

$$
r_{t+1}
+
\gamma\max_{a'}q_t(s_{t+1},a').
$$

这就是它和 Sarsa 的区别。

### Algorithm 7.3：off-policy version

这一版中：

$$
\pi_b\neq\pi_T.
$$

行为策略 $\pi_b$ 可以是 uniform policy：

$$
\pi_b(a|s)=\frac{1}{|\mathcal A(s)|}.
$$

它负责强探索，生成大量经验。

目标策略 $\pi_T$ 是 greedy：

$$
\pi_T(s)=\arg\max_a q(s,a).
$$

它不负责采样，只负责表示“当前学到的最优策略”。

所以这个版本才是最典型的 off-policy Q-learning。

---

## 9. on-policy 不是“不更新策略”

一个常见误解是：

> Sarsa 一边更新 $q$，一边把策略改成 $\epsilon$-greedy，那策略一直在变，怎么还是 on-policy？

答案是：on-policy 不要求策略永远固定。它只要求：

$$
\text{采样用的策略}
\quad
\text{和}
\quad
\text{正在学习的策略}
\quad
\text{是同一个。}
$$

Sarsa 中策略可以随 $q_t$ 变化：

$$
q_t
\to
\pi_t=\epsilon\text{-greedy}(q_t)
\to
\text{采样}
\to
q_{t+1}
\to
\pi_{t+1}=\epsilon\text{-greedy}(q_{t+1}).
$$

但每个时刻它都在用当前策略的数据，学习当前策略对应的价值。

所以它仍然是 on-policy。

---

## 10. off-policy 也不是“不探索”

另一个常见误解是：

> Q-learning 是 off-policy，所以是不是不需要探索？

不是。

Q-learning 的目标策略可以是 greedy，不负责探索；但行为策略必须负责探索。

典型搭配是：

$$
\pi_b=\text{uniform 或 }\epsilon\text{-greedy},
\qquad
\pi_T=\text{greedy}.
$$

如果行为策略不探索，很多状态-动作对永远没有样本，Q-learning 也学不到。

所以：

$$
\text{off-policy}
\neq
\text{不需要探索}.
$$

更准确地说：

> off-policy 是把“探索采样”和“学习目标”分开。

---

## 11. online/offline 和 on/off-policy 不是一回事

online/offline 说的是什么时候更新：

| 概念 | 问的问题 |
|---|---|
| online | 是不是每来一步样本就能更新？ |
| offline | 是不是先收集一批数据，之后再统一训练？ |
| on-policy | 采样策略和目标策略是不是同一个？ |
| off-policy | 能不能用一个策略的数据去学另一个策略？ |

所以 Sarsa 可以是 online on-policy。

Q-learning 可以是 online on-policy，也可以是 online/offline off-policy。

这两个维度不要混在一起。

---

## 12. 最后记忆法

看一个 TD control 算法是不是 on-policy，最简单就看 target 的最后一项。

Sarsa：

$$
r+\gamma q(s',a_{t+1})
$$

用了实际采样动作 $a_{t+1}$，所以是 on-policy。

Q-learning：

$$
r+\gamma\max_{a'}q(s',a')
$$

不用实际采样动作，而是用贪心动作；这就是原书把 Q-learning 归为 off-policy 的主要原因。更细一点说，Q-learning 的 target 本身始终偏向 greedy，但算法外层可以按原书分成 on-policy version 和 off-policy version 两种跑法。

Expected Sarsa：

$$
r+\gamma\sum_{a'}\pi(a'|s')q(s',a')
$$

要看这里的 $\pi$ 是行为策略还是另一个目标策略。

最短版：

$$
\boxed{
\text{用实际走出来的策略学它自己：on-policy}
}
$$

$$
\boxed{
\text{用一套策略采样，学另一套策略：off-policy}
}
$$
