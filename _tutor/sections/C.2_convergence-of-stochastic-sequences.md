# C.2 Convergence of stochastic sequences

We now consider stochastic sequences. While various definitions of stochastic sequences have been given in Appendix B, how to determine the convergence of a given stochastic sequence has not yet been discussed. We next present an important class of stochastic sequences called martingales. If a sequence can be classified as a martingale (or one of its variants), then the convergence of the sequence immediately follows.

# Convergence of martingale sequences

Definition: A stochastic sequence $\{X_k\}_{k=1}^{\infty}$ is called a martingale if $\mathbb{E}[|X_k|] < \infty$ and

$$
\mathbb {E} \left[ X _ {k + 1} \mid X _ {1}, \dots , X _ {k} \right] = X _ {k} \tag {C.3}
$$

almost surely for all $k$ .

Here, $\mathbb{E}[X_{k + 1}|X_1,\ldots ,X_k]$ is a random variable rather than a deterministic value. The term "almost surely" in the second condition is due to the definition of such expectations. In addition, $\mathbb{E}[X_{k + 1}|X_1,\dots,X_k]$ is often written as $\mathbb{E}[X_{k + 1}|\mathcal{H}_k]$ for short where $\mathcal{H}_k = \{X_1,\ldots ,X_k\}$ represents the "history" of the sequence. $\mathcal{H}_k$ has a specific name called a filtration. More information can be found in [96, Chapter 14] and [104].

Example: An example that can demonstrate martingales is random walk, which is a stochastic process describing the position of a point that moves randomly. Specifically, let $X_{k}$ denote the position of the point at time step $k$ . Starting from $X_{k}$ , the expectation of the next position $X_{k + 1}$ equals $X_{k}$ if the mean of the one-step displacement is zero. In this case, we have $\mathbb{E}[X_{k + 1}|X_1,\ldots ,X_k] = X_k$ and hence $\{X_k\}$ is a martingale.

A basic property of martingales is that

$$
\mathbb {E} [ X _ {k + 1} ] = \mathbb {E} [ X _ {k} ]
$$

for all $k$ and hence

$$
\mathbb {E} [ X _ {k} ] = \mathbb {E} [ X _ {k - 1} ] = \dots = \mathbb {E} [ X _ {2} ] = \mathbb {E} [ X _ {1} ]
$$

This result can be obtained by calculating the expectation on both sides of (C.3) based on property (b) in Lemma B.2.

While the expectation of a martingale is constant, we next extend martingales to submartingales and supermartingales, whose expectations vary monotonically.

Definition: A stochastic sequence $\{X_k\}$ is called a submartingale if it satisfies $\mathbb{E}[|X_k|] < \infty$ and

$$
\mathbb {E} \left[ X _ {k + 1} \mid X _ {1}, \dots , X _ {k} \right] \geq X _ {k} \tag {C.4}
$$

for all $k$

Taking the expectation on both sides of (C.4) yields $\mathbb{E}[X_{k + 1}] \geq \mathbb{E}[X_k]$ . In particular, the left-hand side leads to $\mathbb{E}[\mathbb{E}[X_{k + 1}|X_1,\ldots ,X_k]] = \mathbb{E}[X_{k + 1}]$ due to property (b) in Lemma B.2. By induction, we have

$$
\mathbb {E} [ X _ {k} ] \geq \mathbb {E} [ X _ {k - 1} ] \geq \dots \geq \mathbb {E} [ X _ {2} ] \geq \mathbb {E} [ X _ {1} ].
$$

Therefore, the expectation of a submartingale is nondecreasing.

It may be worth mentioning that, for two random variables $X$ and $Y$ , $X \leq Y$ means $X(\omega) \leq Y(\omega)$ for all $\omega \in \Omega$ . It does not mean the maximum of $X$ is less than the minimum of $Y$ .

$\diamond$ Definition: A stochastic sequence $\{X_k\}$ is called a supermartingale if it satisfies $\mathbb{E}[|X_k|] < \infty$ and

$$
\mathbb {E} \left[ X _ {k + 1} \mid X _ {1}, \dots , X _ {k} \right] \leq X _ {k} \tag {C.5}
$$

for all $k$

Taking expectation on both sides of (C.5) gives $\mathbb{E}[X_{k + 1}] \leq \mathbb{E}[X_k]$ . By induction, we have

$$
\mathbb {E} \left[ X _ {k} \right] \leq \mathbb {E} \left[ X _ {k - 1} \right] \leq \dots \leq \mathbb {E} \left[ X _ {2} \right] \leq \mathbb {E} \left[ X _ {1} \right].
$$

Therefore, the expectation of a supmartingale is nonincreasing.

The names "submartingale" and "supmartingale" are standard, but it may not be easy for beginners to distinguish them. Some tricks can be employed to do so. For example, since "supermartingale" has a letter "p" that points down, its expectation decreases; since submartingale has a letter "b" that points up, its expectation increases [104].

A supermartingale or submartingale is comparable to a deterministic monotonic sequence. While the convergence result for monotonic sequences has been given in Theorem C.1, we provide a similar convergence result for martingales as follows.

Theorem C.3 (Martingale convergence theorem). If $\{X_k\}$ is a submartingale (or supermartingale), then there is a finite random variable $X$ such that $X_k \to X$ almost surely.

The proof is omitted. A comprehensive introduction to martingales can be found in [96, Chapter 14] and [104].

# Convergence of quasimartingale sequences

We next introduce quasimartingales, which can be viewed as a generalization of martingales since their expectations are not monotonic. They are comparable to nonmonotonic deterministic sequences. The rigorous definition and convergence results of quasimartingales are nontrivial. We merely list some useful results.

The event $A_{k}$ is defined as $A_{k} \doteq \{\omega \in \Omega : \mathbb{E}[X_{k+1} - X_{k}|\mathcal{H}_{k}] \geq 0\}$ , where $\mathcal{H}_{k} = \{X_{1}, \ldots, X_{k}\}$ . Intuitively, $A_{k}$ indicates that $X_{k+1}$ is greater than $X_{k}$ in expectation. Let $\mathbb{1}_{A_{k}}$ be an indicator function:

$$
\mathbb {1} _ {A _ {k}} = \left\{ \begin{array}{l l} 1, & \mathbb {E} [ X _ {k + 1} - X _ {k} | \mathcal {H} _ {k} ] \geq 0, \\ 0, & \mathbb {E} [ X _ {k + 1} - X _ {k} | \mathcal {H} _ {k} ] <   0. \end{array} \right.
$$

The indicator function has a property that

$$
1 = \mathbb {1} _ {A} + \mathbb {1} _ {A ^ {c}}
$$

for any event $A$ where $A^c$ denotes the complementary event of $A$ . As a result, it holds for any random variable that

$$
X = \mathbb {1} _ {A} X + \mathbb {1} _ {A ^ {c}} X.
$$

Although quasimartingales do not have monotonic expectations, their convergence is still ensured under some mild conditions as shown below.

Theorem C.4 (Quasimartingale convergence theorem). For a nonnegative stochastic sequence $\{X_{k}\geq 0\}$ , if

$$
\sum_ {k = 1} ^ {\infty} \mathbb {E} [ (X _ {k + 1} - X _ {k}) \mathbb {1} _ {A _ {k}} ] <   \infty ,
$$

then $\sum_{k=1}^{\infty} \mathbb{E}[(X_{k+1} - X_k) \mathbb{1}_{A_k^c}] > -\infty$ and there is a finite random variable such that $X_k \to X$ almost surely as $k \to \infty$ .

Theorem C.4 can be viewed as an analogy of Theorem C.2, which is for nonmonotonic deterministic sequences. The proof of this theorem can be found in [105, Proposition 9.5]. Note that $X_{k}$ here is required to be nonnegative. As a result, the boundedness of $\sum_{k = 1}^{\infty}\mathbb{E}[(X_{k + 1} - X_k)\mathbb{1}_{A_k}]$ implies the boundedness of $\sum_{k = 1}^{\infty}\mathbb{E}[(X_{k + 1} - X_k)\mathbb{1}_{A_k^c}]$ .

# Summary and comparison

We finally summarize and compare the results for deterministic and stochastic sequences.

Deterministic sequences:

- Monotonic sequences: As shown in Theorem C.1, if a sequence is monotonic and bounded, then it converges.   
- Nonmonotonic sequences: As shown in Theorem C.2, given a nonnegative sequence, even if it is nonmonotonic, it can still converge as long as its variation is damped in the sense that $\sum_{k=1}^{\infty}(x_{k+1} - x_k)^+$ < $\infty$ .

Stochastic sequences:

- Supermartingale/submartingale sequences: As shown in Theorem C.3, the expectation of a supermartingale or submartingale is monotonic. If a sequence is a supermartingale or submartingale, then the sequence converges almost surely.   
- Quasimartingale sequences: As shown in Theorem C.4, even if a sequence's expectation is nonmonotonic, it can still converge as long as its variation is damped in the sense that $\sum_{k=1}^{\infty} \mathbb{E}\left[(X_{k+1} - X_k)\mathbf{1}_{\mathbb{E}[X_{k+1} - X_k|\mathcal{H}_k] > 0}\right] < \infty$ .

The above properties are summarized in Table C.1.

Table C.1: Summary of the monotonicity of different variants of martingales.   

<table><tr><td>Variants of martingales</td><td>Monotonicity of E[Xk]</td></tr><tr><td>Martingale</td><td>Constant: E[Xk+1] = E[Xk]</td></tr><tr><td>Submartingale</td><td>Increasing: E[Xk+1] ≥ E[Xk]</td></tr><tr><td>Supermartingale</td><td>Decreasing: E[Xk+1] ≤ E[Xk]</td></tr><tr><td>Quasimartingale</td><td>Non-monotonic</td></tr></table>
