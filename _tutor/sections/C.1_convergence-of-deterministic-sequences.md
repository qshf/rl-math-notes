# C.1 Convergence of deterministic sequences

# Convergence of monotonic sequences

Consider a sequence $\{x_{k}\} \doteq \{x_{1}, x_{2}, \ldots, x_{k}, \ldots\}$ where $x_{k} \in \mathbb{R}$ . Suppose that this sequence is deterministic in the sense that $x_{k}$ is not a random variable.

One of the most well-known convergence results is that a nonincreasing sequence with a lower bound converges. The following is a formal statement of this result.

Theorem C.1 (Convergence of monotonic sequences). If the sequence $\{x_{k}\}$ is nonincreasing and bounded from below:

Nonincreasing: $x_{k + 1} \leq x_k$ for all $k$ ;   
Lower bound: $x_{k}\geq \alpha$ for all $k$

then $x_{k}$ converges to a limit, which is the infimum of $\{x_{k}\}$ , as $k \to \infty$ .

Similarly, if $\{x_{k}\}$ is nondecreasing and bounded from above, then the sequence is convergent.

# Convergence of nonmonotonic sequences

We next analyze the convergence of nonmonotonic sequences.

Consider a nonnegative sequence $\{x_{k}\geq 0\}$ satisfying

$$
x _ {k + 1} \leq x _ {k} + \eta_ {k}.
$$

In the simple case of $\eta_{k} = 0$ , we have $x_{k + 1}\leq x_{k}$ , and the sequence is monotonic. We now focus on a more general case where $\eta_k\geq 0$ . In this case, the sequence is not monotonic because $x_{k + 1}$ may be greater than $x_{k}$ . Nevertheless, we can still ensure the convergence of the sequence under some mild conditions.

To analyze the convergence of nonmonotonic sequences, we introduce the following useful operator [103]. For any $z \in \mathbb{R}$ , define

$$
z ^ {+} \doteq \left\{ \begin{array}{l l} z, & \text {i f} z \geq 0, \\ 0, & \text {i f} z <   0, \end{array} \right.
$$

$$
z ^ {-} \doteq \left\{ \begin{array}{l l} z, & \text {i f} z \leq 0, \\ 0, & \text {i f} z > 0. \end{array} \right.
$$

It is obvious that $z^+ \geq 0$ and $z^- \leq 0$ for any $z$ . Moreover, it holds that

$$
z = z ^ {+} + z ^ {-}
$$

for all $z\in \mathbb{R}$

To analyze the convergence of $\{x_{k}\}$ , we rewrite $x_{k}$ as

$$
\begin{array}{l} x _ {k} = x _ {k} - x _ {k - 1} + x _ {k - 1} - x _ {k - 2} + \dots - x _ {2} + x _ {2} - x _ {1} + x _ {1} \\ = \sum_ {i = 1} ^ {k - 1} (x _ {i + 1} - x _ {i}) + x _ {1} \\ \stackrel {\cdot} {=} S _ {k} + x _ {1}, \tag {C.1} \\ \end{array}
$$

where $S_{k} \doteq \sum_{i=1}^{k-1}(x_{i+1} - x_{i})$ . Note that $S_{k}$ can be decomposed as

$$
S _ {k} = \sum_ {i = 1} ^ {k - 1} (x _ {i + 1} - x _ {i}) = S _ {k} ^ {+} + S _ {k} ^ {-},
$$

where

$$
S _ {k} ^ {+} = \sum_ {i = 1} ^ {k - 1} (x _ {i + 1} - x _ {i}) ^ {+} \geq 0, \qquad S _ {k} ^ {-} = \sum_ {i = 1} ^ {k - 1} (x _ {i + 1} - x _ {i}) ^ {-} \leq 0.
$$

Some useful properties of $S_{k}^{+}$ and $S_{k}^{-}$ are given below.

$\diamond$ $\{S_k^+ \geq 0\}$ is a nondecreasing sequence since $S_{k + 1}^{+} \geq S_{k}^{+}$ for all $k$ .   
$\diamond$ $\{S_k^- \leq 0\}$ is a nonincreasing sequence since $S_{k+1}^- \leq S_k^-$ for all $k$ .   
$\diamond$ If $S_{k}^{+}$ is bounded from above, then $S_{k}^{-}$ is bounded from below. This is because $S_{k}^{-} \geq -S_{k}^{+} - x_{1}$ due to the fact that $S_{k}^{-} + S_{k}^{+} + x_{1} = x_{k} \geq 0$ .

With the above preparation, we can show the following result.

Theorem C.2 (Convergence of nonmonotonic sequences). For any nonnegative sequence $\{x_{k}\geq 0\}$ , if

$$
\sum_ {k = 1} ^ {\infty} \left(x _ {k + 1} - x _ {k}\right) ^ {+} <   \infty , \tag {C.2}
$$

then $\{x_{k}\}$ converges as $k\to \infty$

Proof. First, the condition $\sum_{k=1}^{\infty}(x_{k+1} - x_k)^+ < \infty$ indicates that $S_k^+ = \sum_{i=1}^{k-1}(x_{i+1} - x_i)^+$ is bounded from above for all $k$ . Since $\{S_k^+\}$ is nondecreasing, the convergence of $\{S_k^+\}$ immediately follows from Theorem C.1. Suppose that $S_k^+$ converges to $S_*^+$ .

Second, the boundedness of $S_{k}^{+}$ implies that $S_{k}^{-}$ is bounded from below since $S_{k}^{-} \geq -S_{k}^{+} - x_{1}$ . Since $\{S_{k}^{-}\}$ is nonincreasing, the convergence of $\{S_{k}^{-}\}$ immediately follows from Theorem C.1. Suppose that $S_{k}^{-}$ converges to $S_{*}^{-}$ .

Finally, since $x_{k} = S_{k}^{+} + S_{k}^{-} + x_{1}$ , as shown in (C.1), the convergence of $S_{k}^{+}$ and $S_{k}^{-}$ implies that $\{x_{k}\}$ converges to $S_{*}^{+} + S_{*}^{-} + x_{1}$ .

Theorem C.2 is more general than Theorem C.1 because it allows $x_{k}$ to increase as long as the increase is damped as in (C.2). In the monotonic case, Theorem C.2 still applies. In particular, if $x_{k + 1} \leq x_{k}$ , then $\sum_{k = 1}^{\infty}(x_{k + 1} - x_k)^+ = 0$ . In this case, (C.2) is still satisfied and the convergence follows.

If $x_{k + 1} \leq x_k + \eta_k$ , the next result provides a condition for $\eta_k$ to ensure the convergence of $\{x_k\}$ . This result is an immediate corollary of Theorem C.2.

Corollary C.1. For any nonnegative sequence $\{x_{k}\geq 0\}$ , if

$$
x _ {k + 1} \leq x _ {k} + \eta_ {k}
$$

and $\{\eta_k\geq 0\}$ satisfies

$$
\sum_ {k = 1} ^ {\infty} \eta_ {k} <   \infty ,
$$

then $\{x_{k}\geq 0\}$ converges.

Proof. Since $x_{k+1} \leq x_k + \eta_k$ , we have $(x_{k+1} - x_k)^+ \leq \eta_k$ for all $k$ . Then, we have

$$
\sum_ {k = 1} ^ {\infty} (x _ {k + 1} - x _ {k}) ^ {+} \leq \sum_ {k = 1} ^ {\infty} \eta_ {k} <   \infty .
$$

As a result, (C.2) is satisfied and the convergence follows from Theorem C.2.
