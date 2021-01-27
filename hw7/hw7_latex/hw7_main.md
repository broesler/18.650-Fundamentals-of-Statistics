# QQ-Plots

Consider the QQ-plots of five i.i.d. random variables with the following
distributions:

1.  Standard Normal, $\mathcal{N}\left( 0, 1 \right)$,

2.  Uniform distribution,
    $\mathcal{U}\left(\left[ -\sqrt{3}, \sqrt{3} \right]\right)$,

3.  Cauchy distribution $\sim g(x) = \frac{1}{\pi}\frac{2}{1+x^2}$,

4.  Exponential distribution $\sim \operatorname{Exp}(\lambda) = \lambda
        e^{-\lambda x}, \lambda = 1$,

5.  Laplace distribution $\sim \operatorname{Laplace}(\lambda)
        = \frac{\lambda}{2} e^{-\lambda x}, \lambda = \sqrt{2}$.

Figure
<a href="#fig:qqplots" data-reference-type="ref" data-reference="fig:qqplots">1</a>
shows the samples labeled with the appropriate distribution.

<figure>
<embed src="qqplots.pdf" id="fig:qqplots" /><figcaption aria-hidden="true">QQ-plots of five i.i.d. random variables from different distributions.</figcaption>
</figure>

# Kolmogorov-Smirnov Test for Two Samples

Consider two independent samples $X_1, \dots, X_n$, and
$Y_1, \dots, Y_m$ of independent, real-valued, continuous random
variables, and assume that the $X_i$’s are i.i.d. with some cdf $F$ and
that the $Y_i$’s are i.i.d. with some cdf $G$. [^1] We want to test
whether $F = G$. Consider the following hypotheses:

$$
\begin{aligned}
  H_0 \colon ``F = G" \\
  H_1 \colon ``F \ne G"\end{aligned}
$$

For simplicity, we will assume
that $F$ and $G$ are continuous and increasing.

## Example Experiment

An example experiment in which testing if two samples are from the same
distribution is of interest may be encountered in a lab setting where we
have two devices for measurement, and wish to determine if the errors
have the same distribution for our analysis.

## CDF Distributions

Let

$$
\begin{aligned}
  U_i &= F(X_i), \quad \forall i = 1, \dots, n, \\
  V_j &= G(Y_j), \quad \forall j = 1, \dots, n.\end{aligned}
$$

<div class="prop">

**Proposition 1**. *The distribution of the cdf of a continuous random
variable is uniform on $[0,
  1]$.*

</div>

<div class="proof">

*Proof.* The distributions of $U_i$ and $V_j$ can be determined by
finding their cdfs. The cdf of $U_i$ is defined by
$F_U(t) \coloneqq \mathbb{P}\left[U_i \le t\right]$. Assuming that
$F(X)$ and $G(Y)$ are invertible, it follows that

$$
\begin{aligned}
{3}
  \mathbb{P}\left[U_i \le t\right] &= \mathbb{P}\left[F(X_i) \le t\right] &\quad&\text{(definition of $U_i$)} \\
                   &= \mathbb{P}\left[X_i \le F^{-1}(t)\right] \\
                   &= F(F^{-1}(t)) &\quad&\text{(definition of cdf)} \\
                   &= t \\
  \therefore F_U(t) &= t \\
  \implies f_U(t) &= \mathcal{U}\left(\left[ 0, 1 \right]\right) \tag*{\qedhere}\end{aligned}$$ ◻

</div>

Likewise, $f_V(t) = \mathcal{U}\left(\left[ 0, 1 \right]\right)$.

## Empirical CDFs

Let $F_n$ be the empirical cdf of $\{X_1, \dots, X_n\}$ and $G_m$ be the
empirical cdf of $\{Y_1, \dots, Y_m\}$.

### The Test Statistic

Let

$$
T_{n,m} = \sup_{t \in \mathbb{R}} \left| F_n(t) - G_m(t) \right|
$$

<div class="prop">

**Proposition 2**. *The test statistic $T_{n,m}$ can be written as the
maximum value of a finite set of numbers.*

</div>

<div class="proof">

*Proof.* By definition, the cdf

$$
\begin{aligned}
{3}
    F(t) &= \mathbb{P}\left[X \le t\right] \quad \forall t \in \mathbb{R}\\
         &= \mathbb{E}\left[\mathbbm{1}\!\left\{X \le t\right\}\right]. \\
    \intertext{By the Law of Large Numbers, the expectation can be approximated
        by the sample average, so we can define the \emph{empirical cdf} as}
    F_n(t) &= \frac{1}{n}\sum_{i=1}^{n} \mathbbm{1}\!\left\{X_i \le t\right\} \addtocounter{equation}{1}\tag{\theequation}\label{eq:F_n}
    \intertext{Likewise,}
    G_m(t) &= \frac{1}{m}\sum_{j=1}^{m} \mathbbm{1}\!\left\{Y_j \le t\right\}. \addtocounter{equation}{1}\tag{\theequation}\label{eq:G_m}
  \end{aligned}
$$

$$
\therefore T_{n,m} = \sup_{t \in \mathbb{R}} \left| \frac{1}{n}\sum_{i=1}^{n} \mathbbm{1}\!\left\{X_i \le t\right\} - \frac{1}{m}\sum_{j=1}^{m} \mathbbm{1}\!\left\{Y_j \le t\right\} \right|.
$$

The empirical
cdfs <a href="#eq:F_n" data-reference-type="eqref" data-reference="eq:F_n">[eq:F_n]</a> and <a href="#eq:G_m" data-reference-type="eqref" data-reference="eq:G_m">[eq:G_m]</a>
can also be written

$$
\begin{aligned}
{3}
    F_n(t) &= \#\{i=1, \dots, n \colon X_i \le t\} \cdot \frac{1}{n} \\
    G_m(t) &= \#\{i=1, \dots, m \colon Y_j \le t\} \cdot \frac{1}{m},
  \end{aligned}
$$

so the only values that the empirical cdfs can take
are the discrete sets

$$
\begin{aligned}
    F_n(i) &= \frac{i}{n} \quad \forall i = 1, \dots, n \\
    G_m(j) &= \frac{j}{m} \quad \forall j = 1, \dots, m.
  \end{aligned}
$$

Therefore, the test statistic can be rewritten as the
maximum value of a finite set of numbers:

$$
\begin{split}
      T_{n,m} = \max_{i=0,\dots,n} \Bigg[
      &\max_{j=0,\dots,m} \left| \frac{i}{n} - \frac{j}{m} \right| 
        \mathbbm{1}\!\left\{Y^{(j)} \le X^{(i)} < Y^{(j+1)}\right\}, \\ 
      &\max_{k=j+1, \dots, m} \left| \frac{i}{n} - \frac{k}{m} \right| 
        \mathbbm{1}\!\left\{Y^{(k)} \le X^{(i+1)}\right\} \Bigg]
    \end{split}
$$

where $X^{(i)}$ is the $i^\text{th}$ value in the
ordered set of data $X^{(1)} \le \cdots \le X^{(n)}$. The values
$X^{(0)}, Y^{(0)} \coloneqq -\infty$ are prepended to the otherwise
finite realizations to simplify the computation. ◻

</div>

The following algorithm calculates the KS test statistic for two given
samples.

$X, Y$ are vectors of real numbers. $0 \le T_{n,m} \le 1$.
$X_s \gets \{-\infty,$ $\}$ $Y_s \gets$ $n \gets \dim X_s$
$m \gets \dim Y_s$ $T_v \gets$ empty array of size $n$ $j \gets j$ +
$k \gets j$ + $\displaystyle{T_v^{(i)} \gets 
        \max\left(\left|\frac{i}{n} - \frac{j}{m}\right|,
              \left|\frac{i}{n} - \frac{k}{m}\right|\right)}$
$\max_i T_v$ $A$ is sorted in ascending order.
$\#\{i=1,\dots,\dim A \colon k < A_i\}$

The following subroutine is an implementation of
Algorithm <a href="#alg:ks_stat" data-reference-type="ref" data-reference="alg:ks_stat">[alg:ks_stat]</a>.
It computes an array of values $T_v(i)$ for each value of $X_i$. The
test statistic $T_{n,m}$ is the maximum of these values.

``` python
def _ks_2samp(X, Y):
    """Compute the Kolmogorov-Smirnov statistic on 2 samples.

    Parameters
    ----------
    X, Y : (N,), (M,) array_like
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can differ.

    Returns
    -------
    Tv : (N+1,) ndarray
        Maximum difference in CDFs for each value of X.

    .. note:: The KS statistic itself is the maximum of these `Tv` values, but
        use this helper function for debugging.
    """
    n = len(X)
    m = len(Y)
    # Sort copies of the data
    Xs = np.hstack([-np.inf, np.sort(X)])  # pad extra point
    Ys = np.sort(Y)
    # Calculate the maximum difference in the empirical CDFs
    Tv = np.zeros(n+1)  # extra value at Fn = 0 (Xs -> -infty)
    js = np.zeros(n+1, dtype=int)
    j = 0
    for i in range(n+1):
        # Find greatest Ys point j s.t. Ys[j] <= Xs[i] and Xs[i] < Ys[j+1]
        j += _rank(Ys[j:], Xs[i])  # only search remaining values

        test_lo = np.abs(i/n - j/m)
        j_lo = j

        # Find next greatest Ys point k s.t. Ys[k] < X[i+1]
        k = _rank(Ys[j:], Xs[min(i+1, n)]) + j
        test_hi = np.abs(i/n - k/m)
        j_hi = k

        # Take the maximum distance, and corresponding index
        Tv[i] = np.max((test_lo, test_hi))
        js[i] = j_lo if np.argmax((test_lo, test_hi)) == 0 else j_hi

    return Tv, js

def _rank(A, k):
    """Return the number of keys in `A` strictly less than `k`."""
    assert all(A == sorted(A))
    lo = 0
    hi = len(A) - 1
    while lo <= hi:
        mid = (hi + lo) // 2
        if k < A[mid]:
            hi = mid - 1
        elif k > A[mid]:
            lo = mid + 1
        else:  # k == A[mid]
            return mid
    return lo
```

An example two-sample KS-test is shown in
Figure <a href="#fig:ks_test" data-reference-type="ref" data-reference="fig:ks_test">2</a>.

<figure>
<embed src="ks_test.pdf" id="fig:ks_test" style="width:90.0%" /><figcaption aria-hidden="true">The empirical cdfs of two independent random samples from <span class="math inline">\(\mathcal{N}\left( 0, 1 \right)\)</span> and <span class="math inline">\(\mathcal{N}\left( 0, 2 \right)\)</span>. The test statistic <span class="math inline">\(T_{n,m}\)</span> is shown by the double arrow.</figcaption>
</figure>

### The Null Hypothesis

<div class="prop">

**Proposition 3**. *If $H_0$ is true, then

$$
T_{n,m} = \sup_{0 \le x \le 1} \left| \frac{1}{n}\sum_{i=1}^{n} \mathbbm{1}\!\left\{U_i \le x\right\}
- \frac{1}{m}\sum_{j=1}^{m} \mathbbm{1}\!\left\{V_j \le x\right\} \right|.$$*

</div>

<div class="proof">

*Proof.*
By <a href="#eq:F_n" data-reference-type="eqref" data-reference="eq:F_n">[eq:F_n]</a> and <a href="#eq:G_m" data-reference-type="eqref" data-reference="eq:G_m">[eq:G_m]</a>,

$$
\label{eq:Tnm_supt}
    T_{n,m} = \sup_{t \in \mathbb{R}} \left| \frac{1}{n}\sum_{i=1}^{n} \mathbbm{1}\!\left\{X_i \le t\right\} - \frac{1}{m}\sum_{j=1}^{m} \mathbbm{1}\!\left\{Y_j \le t\right\} \right|.
$$

To show the proposition is true, we make a change of variable. Let

$$
x = F(t).
$$

Then,

$$
t \in \mathbb{R}\implies x \in [0, 1].
$$

Since $F$
and $G$ are continuous and monotonically increasing,

$$
\begin{aligned}
{3}
    X_i \le t &\iff F(X_i) \le F(t) \\
              &\iff U_i \le x &\quad&\text{(definition)}.
  \end{aligned}
$$

Similarly,

$$
\begin{aligned}
{3}
    Y_i \le t &\iff G(Y_i) \le G(t) \\
              &\iff G(Y_i) \le F(t) &\quad&\text{(under $H_0$)} \\
              &\iff V_i \le x &\quad&\text{(definition)}.
  \end{aligned}
$$

Substitution of these expressions
into <a href="#eq:Tnm_supt" data-reference-type="eqref" data-reference="eq:Tnm_supt">[eq:Tnm_supt]</a>
completes the proof. ◻

</div>

### The Joint Distribution of the Samples

<div id="prop:Tnm" class="prop">

**Proposition 4**. *If $H_0$ is true, the joint distribution of
$U_1, \dots, U_n, V_1, \dots, V_m$ $(n+m)$ random variables is uniform
on $[0, 1]$.*

</div>

<div class="proof">

*Proof.*

$$
\begin{aligned}
{3}
    \mathbb{P}\left[U_i \le t\right] &= \mathbb{P}\left[F(X_i) \le t\right] \\
                     &= \mathbb{P}\left[F(X_1) \le t\right] &\quad&\text{(i.i.d.)} \\
                     &= \mathbb{P}\left[G(X_1) \le t\right] &\quad&\text{(under $H_0$)} \\
                     &= \mathbb{P}\left[G(Y_1) \le t\right] &\quad&\text{(i.i.d.)} \\
                     &= \mathbb{P}\left[V_1 \le t\right] &\quad&\text{(definition)} \\
    \intertext{These probabilities can be rearranged to find the cdfs of $U$ and $V$}
                     &= \mathbb{P}\left[X_1 \le F^{-1}(t)\right] \\
                     &= F(F^{-1}(t)) &\quad&\text{(definition of cdf)} \\
                     &= t \\
    \therefore F_U(t) &= G_V(t) = t \\
    \implies f_{U,V}(t) &= \mathcal{U}\left(\left[ 0, 1 \right]\right) \tag*{\qedhere}
  \end{aligned}$$ ◻

</div>

### The Test Statistic is Pivotal

Since
Proposition <a href="#prop:Tnm" data-reference-type="ref" data-reference="prop:Tnm">Proposition 4</a>
has been shown to be true under the null hypothesis $H_0$, and the
distributions of $U_i$ and $V_j$ have been shown to be
$\mathcal{U}\left(\left[ 0, 1 \right]\right)$ independent of the
distributions of the underlying samples $X_i$, $Y_j$, we conclude that
$T_{n,m}$ is *pivotal*, *i.e. *it does not itself depend on the unknown
distributions of the samples.

### Quantiles of the Test Statistic

Let $\alpha \in (0, 1)$ and $q_\alpha$ be the $(1 - \alpha)$-quantile of
the distribution of $T_{n,m}$ under $H_0$. The quantile $q_\alpha$ is
given by

$$
\begin{aligned}
  q_\alpha &= F^{-1}(1-\alpha) \\
           &= \inf\{x \colon F(x) \ge 1 - \alpha\} \\
           &\approx \min\{x \colon F_n(x) \ge 1 - \alpha\}, \quad n < \infty \\
           \implies q_\alpha \approx \hat{q}_\alpha &= \min_i \left\{
               T_{n,m}^{(i)} \colon \tfrac{i}{M} \ge 1 - \alpha \right\}\end{aligned}
$$

where $M \in \mathbb{N}$ is large, and $T_{n,m}^{(i)}$ is the
$i^\text{th}$ value in a sorted sample of $M$ test statistics. Thus,
$q_\alpha$ can be approximated by choosing $i = \ceil{M(1 - \alpha)}$.
An algorithm to approximate $q_\alpha$ given $\alpha$ is as follows.

$n = \dim X$. $m = \dim Y$. $M \in \mathbb{N}$. $\alpha
    \in (0, 1)$. $q_\alpha \in [0, 1]$. $T_v \gets$ empty array of size
$n$ $X_s \gets$ sample of size $n$ from
$\mathcal{N}\left( 0, 1 \right)$. $Y_s \gets$ sample of size $m$ from
$\mathcal{N}\left( 0, 1 \right)$. $T_v^{(i)} \gets$ $T_{vs} \gets$
$j \gets \ceil*{M(1 - \alpha)}$ $T_{vs}^{(j)}$

A plot of the distribution of

$$
\frac{T_{n,m}^M - \overline{T}_{n,m}^M}{\sqrt{\operatorname{Var}\left(T_{n,m}^M\right)}}
$$

is shown in
Figure <a href="#fig:Tnm" data-reference-type="ref" data-reference="fig:Tnm">3</a>
in comparison to a standard normal. The test statistic distribution is
skewed to the left, and has a longer right tail than the standard
normal. Since the asymptotic distribution of the test statistic is not
readily found in theory, we rely on simulation via
Algorithm <a href="#alg:ks_q" data-reference-type="ref" data-reference="alg:ks_q">[alg:ks_q]</a>
to estimate the quantiles.

<figure>
<embed src="ks_dist.pdf" id="fig:Tnm" style="width:95.0%" /><figcaption aria-hidden="true">Empirical distribution of samples of the test statistic <span class="math inline">\(T_{n,m}\)</span>.</figcaption>
</figure>

### The Hypothesis Test

Given the aproximation for $\hat{q}_\alpha$ for $q_\alpha$ from
Algorithm <a href="#alg:ks_q" data-reference-type="ref" data-reference="alg:ks_q">[alg:ks_q]</a>,
we define a test with non-asymptotic level $\alpha$ for $H_0$ vs. $H_1$:

$$
\delta_\alpha = \mathbbm{1}\!\left\{T_{n,m} > \hat{q}_\alpha^{(n, M)}\right\}
$$

where $T_{n,m}$ is found by
Algorithm <a href="#alg:ks_stat" data-reference-type="ref" data-reference="alg:ks_stat">[alg:ks_stat]</a>.
The p-value for this test is

$$
\begin{aligned}
  \text{p-value} &\coloneqq \mathbb{P}\left[Z \ge T_{n,m}\right] \\
  &\approx \frac{\#\{j = 1, \dots, M \colon T_{n,m}^{(j)} \ge T_{n,m}\}}{M}\end{aligned}
$$

where $Z$ is a random variable distributed as $T_{n,m}$.

# Aside: Homework 6, Problem 3 – Test of Independence for Bernoulli Random Variables

Let $X, Y$ be two Bernoulli random variables, not necessarily
independent, and let $p = \mathbb{P}\left[X=1\right]$,
$q = \mathbb{P}\left[Y=1\right]$, and
$r = \mathbb{P}\left[X=1, Y=1\right]$.

## Condition for Independence

<div class="prop">

**Proposition 5**. *$X \perp \!\!\! \perp Y \iff r = pq$.*

</div>

<div class="proof">

*Proof.* Two random variables are independent iff

$$
\mathbb{P}\left[X \cap Y\right] = \mathbb{P}\left[X\right]\mathbb{P}\left[Y\right] \label{eq:indep}
    % \iff \Prob{X} = \frac{\Prob{X,Y}}{\Prob{Y}} = \Prob{X | Y}
$$

By
the given definition of the Bernoulli random variables,

$$
\begin{aligned}
    \mathbb{P}\left[X \cap Y\right] &= \mathbb{P}\left[X, Y\right] = \mathbb{P}\left[X=1, Y=1\right] = r \\
    \mathbb{P}\left[X\right] &= \mathbb{P}\left[X=1\right] = p \\
    \mathbb{P}\left[Y\right] &= \mathbb{P}\left[Y=1\right] = q \\
    \therefore r = pq &\iff \mathbb{P}\left[X,Y\right] = \mathbb{P}\left[X\right]\mathbb{P}\left[Y\right] \\
    &\iff X \perp \!\!\! \perp Y  \qedhere
  \end{aligned}$$ ◻

</div>

## Test for Independence

Let $(X_1, Y_1), \dots, (X_n, Y_n)$ be a sample of $n$ i.i.d. copies of
$(X, Y)$ (*i.e. *$X_i \perp \!\!\! \perp X_j$ for $i \ne j$, but $X_i$
may not be independent of $Y_i$). Based on this sample, we want to test
whether $X \perp \!\!\! \perp Y$, *i.e. *whether $r = pq$.

### Estimators of $p, q, r$

Define the estimators:

$$
\begin{aligned}
  \hat{p}&= \frac{1}{n}\sum_{i=1}^{n} X_i, \\
  \hat{q}&= \frac{1}{n}\sum_{i=1}^{n} Y_i, \\
  \hat{r}&= \frac{1}{n}\sum_{i=1}^{n} X_i Y_i.\end{aligned}
$$

<div class="prop">

**Proposition 6**. *These estimators $\hat{p}$, $\hat{q}$, and $\hat{r}$
are consistent estimators of the true values $p$, $q$, and $r$.*

</div>

<div class="proof">

*Proof.* To show that an estimator is *consistent*, we must prove that
it converges to the true value of the parameter in the limit as
$n \to \infty$. Since the sequence of $X_i$’s are i.i.d., we can use the
Weak Law of Large Numbers (LLN) to prove that $\hat{p}$ converges to
$p$.

<div id="eq:LLN" class="theorem">

**Theorem 1** (Weak Law of Large Numbers). *If the sequence of random
variables $X_1, \dots, X_n$ are i.i.d., then

$$
\frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow[n \to \infty]{\mathbb{P}}\mathbb{E}\left[X\right].$$*

</div>

The expectation of $X$ is given by

$$
\begin{aligned}
{3}
    \mathbb{E}\left[X\right] &= \mathbb{E}\left[\mathop{\mathrm{Ber}}(p)\right] &\quad&\text{(given)} \\
          &= p &\quad&\text{(definition of Bernoulli r.v.)} \\
    \therefore \frac{1}{n}\sum_{i=1}^{n} X_i &\xrightarrow[n \to \infty]{\mathbb{P}}p &\quad&\text{(LLN)} \\
    \implies \hat{p}&\xrightarrow[n \to \infty]{\mathbb{P}}p.
  \end{aligned}
$$

Likewise
$\hat{q}\xrightarrow[n \to \infty]{\mathbb{P}}q$.

To show that $\hat{r}$ converges to $r$, let $R \coloneqq X Y$ be a
Bernoulli random variable with parameter
$r = \mathbb{P}\left[X=1, Y=1\right]$, so that the estimator

$$
\label{eq:rhat}
    \hat{r}= \frac{1}{n}\sum_{i=1}^{n} X_i Y_i = \frac{1}{n}\sum_{i=1}^{n} R_i.
$$

Note that the values of $R_i$ *are* i.i.d. since each pair $(X_i, Y_i)$
are i.i.d., even though $X_i$ and $Y_j$ may not be independent for
$i \ne j$. As before, we apply the Law of Large Numbers to the average
of $R_i$’s. The expectation of $R$ is

$$
\begin{aligned}
{3}
    \mathbb{E}\left[R\right] &= \mathbb{E}\left[\mathop{\mathrm{Ber}}(r)\right] &\quad&\text{(definition)} \\
          &= r &\quad&\text{(definition of Bernoulli r.v.)} \\
    \therefore \frac{1}{n}\sum_{i=1}^{n} R_i &\xrightarrow[n \to \infty]{\mathbb{P}}r &\quad&\text{(LLN)} \\
    \implies \hat{r}&\xrightarrow[n \to \infty]{\mathbb{P}}r.
  \end{aligned}
$$

Thus, each estimator $(\hat{p}, \hat{q}, \hat{r})$
converges to its respective parameter $(p, q, r)$. ◻

</div>

### Asymptotic Normality of the Estimators

<div id="prop:normality_pqr" class="prop">

**Proposition 7**. *The vector of estimators
$(\hat{p}, \hat{q}, \hat{r})$ is asymptotically normal, *i.e. *

$$
\sqrt{n} ((\hat{p}, \hat{q}, \hat{r}) - (p, q, r)) 
        \xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, \operatorname{Cov}\left((\hat{p}, \hat{q}, \hat{r})\right) \right).$$*

</div>

*Proof.* To prove that the vector of estimators is asymptotically
normal, we employ the Central Limit Theorem (CLT).

<div class="theorem">

**Theorem 2** (Central Limit Theorem). *Let $X_1, \dots, X_n$ be a
sequence of i.i.d. random vectors $X_i \in \mathbb{R}^k$, and
$\overline{X}_n= \frac{1}{n}\sum_{i=1}^{n} X_i$. Then,

$$
\sqrt{n}(\overline{X}_n- \mathbb{E}\left[X\right]) \xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, \Sigma \right).
$$

where $\Sigma$ is the $k$-by-$k$ matrix
$\operatorname{Cov}\left(X\right)$.*

</div>

By the CLT,

$$
\label{eq:CLT}
  \sqrt{n} ((\hat{p}, \hat{q}, \hat{r}) - \mathbb{E}\left[(\hat{p}, \hat{q}, \hat{r})\right]) \xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, \Sigma \right)
$$

where $\Sigma$ is the 3-by-3 symmetric covariance matrix, defined as

$$
\Sigma \coloneqq 
  \begin{bmatrix}
    \operatorname{Var}\left(\hat{p}\right) & \operatorname{Cov}\left(\hat{p}, \hat{q}\right) & \operatorname{Cov}\left(\hat{p}, \hat{r}\right) \\
    \cdot & \operatorname{Var}\left(\hat{q}\right) & \operatorname{Cov}\left(\hat{q}, \hat{r}\right) \\
    \cdot & \cdot & \operatorname{Var}\left(\hat{r}\right)
  \end{bmatrix}.
$$

We first need to determine the expectations of the estimators.

<div class="prop">

**Proposition 8**. *The expectation of the estimator $\hat{p}$ is
$\mathbb{E}\left[\hat{p}\right] = p$.*

</div>

<div class="proof">

*Proof.*

$$
\begin{aligned}
{3}
    \mathbb{E}\left[\hat{p}\right] &= \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^{n} X_i\right] &\quad&\text{(definition)} \\
              &= \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}\left[X_i\right] &\quad&\text{(linearity of expectation)} \\
              &= \frac{1}{n} n \mathbb{E}\left[X\right] &\quad&\text{(i.i.d.)} \\
              &= \mathbb{E}\left[\mathop{\mathrm{Ber}}(p)\right] &\quad&\text{(definition)} \\
    \implies \mathbb{E}\left[\hat{p}\right] &= p &\quad&\text{(definition)} \tag*{\qedhere}
  \end{aligned}$$ ◻

</div>

Similarly, $\mathbb{E}\left[\hat{q}\right] = q$ and
$\mathbb{E}\left[\hat{r}\right] = r$. This proposition also shows that
the estimators are *unbiased*, since
$\mathbb{E}\left[\hat{p}- p\right] = 0$, *etc*.

We now determine the entries in the covariance matrix to complete the
proof of asymptotic normality.

<div class="prop">

**Proposition 9**. *The variance of $\hat{p}$ is given by
$\operatorname{Var}\left(\hat{p}\right) = \frac{1}{n} p(1 - p)$.*

</div>

<div class="proof">

*Proof.* Using the definition of $\hat{p}$,

$$
\begin{aligned}
{3}
    \operatorname{Var}\left(\hat{p}\right) &= \operatorname{Var}\left(\frac{1}{n}\sum_{i=1}^{n} X_i\right) &\quad&\text{(definition)} \\
                &= \frac{1}{n^2}\operatorname{Var}\left(\sum_{i=1}^{n} X_i\right) &\quad&\text{(variance rule)} \\
                &= \frac{1}{n^2}\sum_{i=1}^{n}\operatorname{Var}\left(X_i\right) &\quad&\text{(i.i.d.)} \\
                &= \frac{1}{n^2}n\operatorname{Var}\left(X\right) &\quad&\text{(i.i.d.)} \\
   \therefore \operatorname{Var}\left(\hat{p}\right) &= \frac{1}{n} p(1-p) &\quad&\text{(variance of $\mathop{\mathrm{Ber}}(p)$)}
  \end{aligned}
$$

Likewise,
$\operatorname{Var}\left(\hat{q}\right) = \frac{1}{n} q(1 - q)$, and
$\operatorname{Var}\left(\hat{r}\right) = \frac{1}{n} r(1 - r)$. ◻

</div>

<div class="prop">

**Proposition 10**. *The covariance of $\hat{p}$ and $\hat{q}$ is given
by $\operatorname{Cov}\left(\hat{p}, \hat{q}\right) = r - pq$.*

</div>

<div class="proof">

*Proof.*

$$
\begin{aligned}
{3}
    \operatorname{Cov}\left(\hat{p}, \hat{q}\right) &= \operatorname{Cov}\left(\frac{1}{n}\sum_{i=1}^{n} X_i, \frac{1}{n}\sum_{i=1}^{n} Y_i\right) \\
    &= \frac{1}{n^2} \operatorname{Cov}\left(\sum_{i=1}^{n} X_i, \sum_{i=1}^{n} Y_i\right) &\quad&\text{(covariance property)} \\
    &= \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} \operatorname{Cov}\left(X_i, Y_j\right) &\quad&\text{(bilinearity of covariance)} \\
    &= \frac{1}{n^2} n^2 \operatorname{Cov}\left(X, Y\right) &\quad&\text{(identically distributed)} \\
    &= \operatorname{Cov}\left(X, Y\right)  \\
    &= \mathbb{E}\left[XY\right] - \mathbb{E}\left[X\right]\mathbb{E}\left[Y\right] &\quad&\text{(definition of covariance)} \\
    &= \mathbb{E}\left[R\right] - \mathbb{E}\left[X\right]\mathbb{E}\left[Y\right] &\quad&\text{(definition of $R$)} \\
    \therefore \operatorname{Cov}\left(\hat{p}, \hat{q}\right) &= r - pq \tag*{\qedhere}
  \end{aligned}$$ ◻

</div>

<div class="prop">

**Proposition 11**. *The covariance of $\hat{p}$ and $\hat{r}$ is given
by $\operatorname{Cov}\left(\hat{p}, \hat{r}\right) = r(1 - p)$.*

</div>

<div class="proof">

*Proof.*

$$
\begin{aligned}
{3}
    \operatorname{Cov}\left(\hat{p}, \hat{r}\right) &= \operatorname{Cov}\left(\frac{1}{n}\sum_{i=1}^{n} X_i, \frac{1}{n}\sum_{i=1}^{n} R_i\right) \\
    &= \frac{1}{n^2} \operatorname{Cov}\left(\sum_{i=1}^{n} X_i, \sum_{i=1}^{n} R_i\right) &\quad&\text{(covariance property)} \\
    &= \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} \operatorname{Cov}\left(X_i, R_j\right) &\quad&\text{(bilinearity of covariance)} \\
    &= \frac{1}{n^2} n^2 \operatorname{Cov}\left(X, R\right) &\quad&\text{(identically distributed)} \\
    &= \operatorname{Cov}\left(X, R\right)  \\
    &= \mathbb{E}\left[X R\right] - \mathbb{E}\left[X\right]\mathbb{E}\left[R\right] &\quad&\text{(definition of covariance)} \\
    &= \mathbb{E}\left[X R\right] - pr &\quad&\text{(given)} \\
    &= \mathbb{E}\left[X (X Y)\right] - pr &\quad&\text{(definition of $R$)} \\
    \intertext{Since $X \sim \mathop{\mathrm{Ber}}(p) \in \{0, 1\}$, $X^2 = X$, so we have}
    &= \mathbb{E}\left[X Y\right] - pr \\
    &= r - pr \\
    \therefore \operatorname{Cov}\left(\hat{p}, \hat{r}\right) &= r(1 - p) \tag*{\qedhere}
  \end{aligned}$$ ◻

</div>

Similarly, $\operatorname{Cov}\left(\hat{q}, \hat{r}\right) = r(1 - q)$.

The entire asymptotic covariance matrix is then

$$
\label{eq:sigma}
  \Sigma =
  \begin{bmatrix}
    p(1-p) & r - pq & r(1-p) \\
    \cdot & q(1-q) & r(1-q) \\
    \cdot & \cdot & r(1-r)
  \end{bmatrix}.
$$

Since we have determined the expectation
$\mathbb{E}\left[(\hat{p}, \hat{q}, \hat{r})\right] = (p, q,
r)$, and the covariance matrix $\Sigma$ in terms of $p$, $q$, and $r$,
we conclude that
Proposition <a href="#prop:normality_pqr" data-reference-type="ref" data-reference="prop:normality_pqr">Proposition 7</a>
is true, and the vector of estimators $(\hat{p}, \hat{q}, \hat{r})$ is
asymptotically normal. 0◻

### The Delta Method

<div id="prop:delta" class="prop">

**Proposition 12**.
*$$\sqrt{n}\left((\hat{r}- \hat{p}\hat{q}) - (r - pq)\right) \xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, V \right)
$$

where $V$ depends only on $p$, $q$, and $r$.*

</div>

<div class="proof">

*Proof.* Let $\hat{\theta}$ and $\theta$ be vectors in $\mathbb{R}^3$

$$
\hat{\theta} = \begin{bmatrix} \hat{p}\\ \hat{q}\\ \hat{r}\end{bmatrix} \text{, and } 
    \theta = \begin{bmatrix} p \\ q \\ r \end{bmatrix}.
$$

From our proof
of
Proposition <a href="#prop:normality_pqr" data-reference-type="ref" data-reference="prop:normality_pqr">Proposition 7</a>,
we have

$$
\begin{aligned}
{3}
    \sqrt{n}(\hat{\theta} - \theta) &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, \Sigma \right) &\quad&\text{(CLT)} \\
    \implies \sqrt{n}(g(\hat{\theta}) - g(\theta)) &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, \nabla g(\theta)^\top\Sigma
      \nabla g(\theta) \right) &\quad&\text{(Delta method)}
  \end{aligned}
$$

for any differentiable function
$g \colon \mathbb{R}^k \to \mathbb{R}$, and $\Sigma$ given by
Equation <a href="#eq:sigma" data-reference-type="eqref" data-reference="eq:sigma">[eq:sigma]</a>.
Define the function

$$
\label{eq:g}
    g(u, v, w) = w - uv
$$

such that

$$
\begin{aligned}
    g(\hat{\theta}) &= \hat{r}- \hat{p}\hat{q}, \\
    g(\theta) &= r - pq.
  \end{aligned}
$$

The gradient of $g(\theta)$ is then

$$
\nabla g(u,v,w) = \begin{bmatrix} -v \\ -u \\ 1 \end{bmatrix} 
    \implies \nabla g(\theta) = \begin{bmatrix} -q \\ -p \\ 1 \end{bmatrix}
$$

The asymptotic variance
$V = \nabla g(\theta)^\top\Sigma \nabla g(\theta)$, which we will now
show is a function only of the parameters $(p, q, r)$.

$$
\begin{aligned}
{3}
    V &= \begin{bmatrix} -q & -p & 1 \end{bmatrix} 
    \begin{bmatrix}
      p(1-p) & r - pq & r(1-p) \\
      \cdot & q(1-q) & r(1-q) \\
      \cdot & \cdot & r(1-r)
    \end{bmatrix}
    \begin{bmatrix} -q \\ -p \\ 1 \end{bmatrix} \addtocounter{equation}{1}\tag{\theequation}\label{eq:V_mat} \\
    &= \begin{bmatrix} -q & -p & 1 \end{bmatrix} 
    \begin{bmatrix} 
      -qp(1-p) - p(r - pq) + r(1-p) \\
      -q(r - pq) - pq(1-q) + r(1-q) \\
      -qr(1-p) - pr(1-q) + r(1-r)
    \end{bmatrix} \\
    &= \begin{bmatrix} -q & -p & 1 \end{bmatrix} 
    \begin{bmatrix} 
      (r - pq)(1 - 2p) \\
      (r - pq)(1 - 2q) \\
      r((1-p)(1-q) - (r-pq))
    \end{bmatrix} \\
    \begin{split}
      &= -q(r - pq)(1 - 2p) - p(r - pq)(1 - 2q)) \\
      &\,\quad + r((1-p)(1-q) - (r-pq))
    \end{split} \\
    \therefore V &= (r - pq)[-q(1 - 2p) - p(1 - 2q) - r] + r(1-p)(1-q) 
    \addtocounter{equation}{1}\tag{\theequation}\label{eq:V}
  \end{aligned}
$$

which is a function only of $(p, q, r)$. ◻

</div>

### The Null Hypothesis

Consider the hypotheses

$$
\begin{aligned}
  H_0 \colon X \perp \!\!\! \perp Y \\
  H_1 \colon X \centernot\perp \!\!\! \perp Y\end{aligned}
$$

<div id="prop:V_H0" class="prop">

**Proposition 13**. *If $H_0$ is true, then $V = pq(1-p)(1-q)$.*

</div>

<div class="proof">

*Proof.* Under $H_0$, $r = pq$. Using the previous expression for $V$,
Equation <a href="#eq:V" data-reference-type="eqref" data-reference="eq:V">[eq:V]</a>,
replace $r$ by $pq$ to find

$$
V = (pq - pq)[-q(1 - 2p) - p(1 - 2q) - pq] + pq(1-p)(1-q).
$$

The first
term is identically 0, so

$$
V = pq(1-p)(1-q). \qedhere$$ ◻

</div>

<div id="prop:Vhat" class="prop">

**Proposition 14**. *Given

$$
V = pq(1-p)(1-q),
$$

a consistent estimator
is given by

$$
\hat{V} = \hat{p}\hat{q}(1 - \hat{p})(1 - \hat{q}).$$*

</div>

<div class="proof">

*Proof.* To prove that $\hat{V}$ converges to $V$, we employ the
Continuous Mapping Theorem.

<div class="theorem">

**Theorem 3** (Continuous Mapping Theorem). *Let $X \in \mathbb{R}^n$ be
a vector of random variables, and $g \colon \mathbb{R}^n \to \mathbb{R}$
be a continuous function. Let $X_n = X_1, X_2, \dots$ be a sequence of
random vectors. If $X_n \xrightarrow[n \to \infty]{\mathbb{P}}X$, then
$g(X_n) \xrightarrow[n \to \infty]{\mathbb{P}}g(X)$.*

</div>

Since $\hat{p}\xrightarrow[n \to \infty]{\mathbb{P}}p$ and
$\hat{q}\xrightarrow[n \to \infty]{\mathbb{P}}q$,
$\hat{V}(\hat{p}, \hat{q}) \xrightarrow[n \to \infty]{\mathbb{P}}V(p, q)$. ◻

</div>

### A Hypothesis Test

<div class="prop">

**Proposition 15**. *Given $\alpha \in (0, 1)$, we propose the test
statistic

$$
T_n \coloneqq \frac{\sqrt{n}(\hat{r}- \hat{p}\hat{q})}{\sqrt{\hat{V}}} \xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, 1 \right)
$$

where $\hat{V}$ is given by
Proposition <a href="#prop:Vhat" data-reference-type="ref" data-reference="prop:Vhat">Proposition 14</a>,
and $t_{n-1}$ is Student’s $t$-distribution with $n-1$ degrees of
freedom.*

</div>

<div class="proof">

*Proof.*
Proposition <a href="#prop:delta" data-reference-type="ref" data-reference="prop:delta">Proposition 12</a>
gives the distribution of $g(\theta)$ (given by
Equation <a href="#eq:g" data-reference-type="eqref" data-reference="eq:g">[eq:g]</a>)
under $H_0$. Assume that $p, q \in (0, 1)$ s.t. $V > 0$.

$$
\begin{aligned}
{3}
  \sqrt{n}\left((\hat{r}- \hat{p}\hat{q}) - (r - pq)\right) &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, V \right)
  &\quad&\text{(Proposition~\ref{prop:delta})} \\
  \sqrt{n}(\hat{r}- \hat{p}\hat{q}) &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, V \right) &\quad&\text{($r = pq$ under $H_0$)} \\
  \sqrt{n}\frac{(\hat{r}- \hat{p}\hat{q})}{\sqrt{V}} &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, 1 \right) \addtocounter{equation}{1}\tag{\theequation}
  \label{eq:Tn_norm}\end{aligned}
$$

The asymptotic variance $V$,
however, is unknown, so we divide the estimator by
$\sqrt{\frac{\hat{V}}{V}}$ to get an expression that will evaluate to
our desired test statistic

$$
\label{eq:t1}
  T_n = \frac{\displaystyle \sqrt{n}\frac{(\hat{r}- \hat{p}\hat{q})}{\sqrt{V}}}{\displaystyle \sqrt{\frac{\hat{V}}{V}}}
$$

Given this expression, we can determine the distribution of $T_n$.
Equation <a href="#eq:Tn_norm" data-reference-type="eqref" data-reference="eq:Tn_norm">[eq:Tn_norm]</a>
shows that the numerator is a standard normal random variable.
[<u>Cochran’s
theorem</u>](https://en.wikipedia.org/wiki/Cochran's_theorem) gives the
distribution of the denominator.

<div class="lemma">

**Lemma 4** (Result of Cochran’s Theorem). *If $X_1, \dots, X_n$ are
i.i.d. random variables drawn from the distribution
$\mathcal{N}\left( \mu, \sigma^2 \right)$, and
$S_n^2 \coloneqq \sum_{i=1}^{n} (X_i - \overline{X}_n)^2$, then

$$
\overline{X}_n\perp \!\!\! \perp S_n,
$$

and

$$
\frac{n S_n^2}{\sigma^2} \sim \chi^2_{n-1}.$$*

</div>

Since $\hat{V}$ and $V$ describe the sample variance and variance of a
(asymptotically) normal distribution, $T_n$ is asymptotically
characterized by

$$
T_n \xrightarrow[n \to \infty]{(d)}\frac{\displaystyle \mathcal{N}\left( 0, 1 \right)}{\displaystyle \sqrt{\frac{\chi^2_{n-1}}{n}}}
$$

which is the definition of a random variable drawn from *Student’s
T-distribution* with $n-1$ degrees of freedom. In this case, however,
the normality of the underlying random variables is asymptotic, so the
$t_{n-1}$ distribution approaches a standard normal distribution

$$
\begin{aligned}
  T_n &\xrightarrow[n \to \infty]{(d)}t_{n-1} \\
  t_{n-1} &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, 1 \right) \addtocounter{equation}{1}\tag{\theequation}\label{eq:t_to_N} \\
  \implies T_n &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, 1 \right) \qedhere\end{aligned}
$$

A proof of
Equation <a href="#eq:t_to_N" data-reference-type="eqref" data-reference="eq:t_to_N">[eq:t_to_N]</a>
is given in . ◻

</div>

Given the test statistic $T_n$, define the rejection region

$$
R_\psi = \left\{ \hat{\theta} \colon |T_n| > q_{\alpha/2} \right\}
$$

where

$$
q_{\alpha/2} = \Phi^{-1}\left(1 - \frac{\alpha}{2}\right)
$$

is
the $\left(1-\frac{\alpha}{2}\right)$-quantile of the standard normal
$\mathcal{N}\left( 0, 1 \right)$ distribution.

We would like to know whether the facts of being happy and being in a
relationship are independent of each other. In a given population, 1000
people (aged at least 21 years old) are sampled and asked two questions:
“Do you consider yourself as happy?” and “Are you involved in a
relationship?”. The answers are summarized in
Table <a href="#tab:tab1" data-reference-type="ref" data-reference="tab:tab1">1</a>.

<div id="tab:tab1">

|                           | **Happy** | **Not Happy** | **Total** |
|--------------------------:|:---------:|:-------------:|:---------:|
|     **In a Relationship** |    205    |      301      |    506    |
| **Not in a Relationship** |    179    |      315      |    494    |
|                 **Total** |    384    |      616      |   1000    |
|                           |           |               |           |

</div>

The values of our estimators are as follows:

$$
\begin{aligned}
  \hat{p}&= \frac{\text{\# Happy}}{N} = \frac{384}{1000} = 0.384 \\
  \hat{q}&= \frac{\text{\# In a Relationship}}{N} = \frac{506}{1000} = 0.506 \\
  \hat{r}&= \frac{\text{\# Happy} \cap \text{\# In a Relationship}}{N} = \frac{205}{1000} = 0.205.\end{aligned}
$$

The estimate of the asymptotic variance of the test statistic is

$$
\hat{V} = \hat{p}\hat{q}(1-\hat{p})(1-\hat{q}) = (0.384)(0.506)(1 - 0.384)(1 - 0.506)
  = 0.05913,
$$

giving the test statistic

$$
T_n = \frac{\sqrt{n}(\hat{r}- \hat{p}\hat{q})}{\sqrt{\hat{V}}}
  = \frac{\sqrt{1000}(0.205 - 0.384\cdot0.506)}{\sqrt{0.05913}} = 1.391.
$$

The standard normal quantile at $\alpha = 0.05$ is $q_{\alpha/2}
= \Phi^{-1}\left(1
- \frac{\alpha}{2}\right) = 1.96$, so the test result is

$$
|T_n| = 1.391 < q_{\alpha/2} = 1.96
$$

so we *fail to reject $H_0$ at
the 5% level*. The $p$-value of the test is

$$
\begin{aligned}
  \text{$p$-value} &\coloneqq \mathbb{P}\left[Z > |T_n|\right] &\quad&\text{($Z \sim \mathcal{N}\left( 0, 1 \right)$)}  \\
                   &= \mathbb{P}\left[Z \le |T_n|\right] &\quad&\text{(symmetry)} \\
                   &= \mathbb{P}\left[Z \le -T_n\right] + \mathbb{P}\left[Z > T_n\right] \\
                   &= 2\Phi(-|T_n|) \\
  \implies \text{$p$-value} &= 0.1642.\end{aligned}
$$

In other words,
the lowest level at which we could reject the null hypothesis is at
$\alpha = \text{$p$-value} = 0.1642 = 16.42\%$.

## Appendix A: Additional Proofs

<div class="prop">

**Proposition 16**. *A $t$-distribution with $n$ degrees of freedrom
approaches a standard normal distribution as $n$ approaches infinity:

$$
t_n \xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, 1 \right).$$*

</div>

*Proof.* Student’s $t$-distribution with $\nu$ degrees of freedom is
defined as the distribution of the random variable $T$ such that

$$
t_{\nu} \sim T = \frac{\displaystyle Z}{\displaystyle \sqrt{\frac{V}{\nu}}}
$$

where $Z \sim \mathcal{N}\left( 0, 1 \right)$, $V \sim \chi^2_{\nu}$,
and $Z \perp \!\!\! \perp V$.

Let $X_1, \dots, X_n \sim \mathcal{N}\left( \mu, \sigma^2 \right)$ be a
sequence of i.i.d. random variables. Define the sample mean and sample
variance

$$
\begin{aligned}
  \overline{X}_n&\coloneqq \frac{1}{n}\sum_{i=1}^{n} X_i \\
  S_n^2 &\coloneqq \frac{1}{n}\sum_{i=1}^{n} (X_i - \overline{X}_n)^2.\end{aligned}
$$

Let the random variables

$$
\begin{aligned}
  Z &= \frac{\sqrt{n}(\overline{X}_n- \mu)}{\sigma} \\
  V &= \frac{n S_n^2}{\sigma^2}.\end{aligned}
$$

such that
$Z \sim \mathcal{N}\left( 0, 1 \right)$ by the Central Limit Theorem,
and $V \sim
\chi^2_{n-1}$ by Cochran’s Theorem (which also shows that
$Z \perp \!\!\! \perp V$). Then, the $t$-distribution is *pivotal*

$$
t_{n-1} = \frac{\displaystyle Z}{\displaystyle \sqrt{\frac{V}{n-1}}}.
$$

<div class="lemma">

**Lemma 5**. *The sample variance converges in probability to the
variance,

$$
S_n^2 \xrightarrow[n \to \infty]{\mathbb{P}}\sigma^2.$$*

</div>

<div class="proof">

*Proof.*

$$
\begin{aligned}
{3}
    S_n^2 &\coloneqq \frac{1}{n}\sum_{i=1}^{n}(X_i - \overline{X}_n)^2 \\
          &= \frac{1}{n}\sum_{i=1}^{n}(X_i^2 - 2 \overline{X}_nX_i + \overline{X}_n^2) \\
          &= \frac{1}{n}\sum_{i=1}^{n} X_i^2 - \frac{1}{n}\sum_{i=1}^{n} 2 \overline{X}_nX_i + \frac{1}{n}\sum_{i=1}^{n} \overline{X}_n^2  \\
          &= \frac{1}{n}\sum_{i=1}^{n} X_i^2 - 2 \overline{X}_n\frac{1}{n}\sum_{i=1}^{n} X_i + \overline{X}_n^2  \\
          &= \frac{1}{n}\sum_{i=1}^{n} X_i^2 - 2 \overline{X}_n^2  + \overline{X}_n^2  \\
          &= \frac{1}{n}\sum_{i=1}^{n} X_i^2 - \overline{X}_n^2.
  \end{aligned}
$$

The second term in the expression for $S_n^2$ is
determined by

$$
\begin{aligned}
{3}
    \overline{X}_n&\xrightarrow[n \to \infty]{\mathbb{P}}\mathbb{E}\left[X\right] &\quad&\text{(LLN)} \\
    \mathbb{E}\left[X\right] &= \mu &\quad&\text{(given)}. \\
    g(\overline{X}_n) &\xrightarrow[n \to \infty]{\mathbb{P}}g(\mu) &\quad&\text{(CMT)} \\
    \implies \overline{X}_n^2 &\xrightarrow[n \to \infty]{\mathbb{P}}\mu^2.
  \end{aligned}
$$

The first term in the expression for $S_n^2$ is then
determined by

$$
\begin{aligned}
{3}
    \frac{1}{n}\sum_{i=1}^{n} X_i^2 &\xrightarrow[n \to \infty]{\mathbb{P}}\mathbb{E}\left[X^2\right] &\quad&\text{(LLN)} \\
    \operatorname{Var}\left(X\right) &= \mathbb{E}\left[X^2\right] - \mathbb{E}\left[X\right]^2 &\quad&\text{(definition)} \\
    \implies \mathbb{E}\left[X^2\right] &= \operatorname{Var}\left(X\right) + \mathbb{E}\left[X\right]^2 \\
                     &= \sigma^2 + \mu^2. &\quad&\text{(given)} \\
    \therefore S_n^2 &\xrightarrow[n \to \infty]{\mathbb{P}}\sigma^2 + \mu^2 - \mu^2 \\
    \implies S_n^2 &\xrightarrow[n \to \infty]{\mathbb{P}}\sigma^2 \tag*{\qedhere}
  \end{aligned}$$ ◻

</div>

Thus,
$V \xrightarrow[n \to \infty]{\mathbb{P}}\frac{n \sigma^2}{\sigma^2} = n$,
a constant.

<div class="theorem">

**Theorem 6** (Slutsky’s Theorem). *If the sequences of random variables
$X_n~\xrightarrow[n \to \infty]{(d)}~X$, and
$Y_n~\xrightarrow[n \to \infty]{(d)}~c$, a constant, then

$$
\begin{aligned}
{3}
    X_n + Y_n &\xrightarrow[n \to \infty]{(d)}X + c \text{, and} \\
    X_n Y_n &\xrightarrow[n \to \infty]{(d)}cX.
  \end{aligned}$$*

</div>

Since convergence in probability implies convergence in distribution,
and $Z
\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, 1 \right)$,
Slutsky’s theorem implies that

$$
\begin{aligned}
  t_{n-1} = \frac{\displaystyle Z}{\displaystyle \sqrt{\frac{V}{n-1}}} 
        &\xrightarrow[n \to \infty]{(d)}\frac{\displaystyle \mathcal{N}\left( 0, 1 \right)}{\displaystyle \sqrt{\frac{n}{n-1}}} \\
        \implies t_{n-1} &\xrightarrow[n \to \infty]{(d)}\mathcal{N}\left( 0, 1 \right).  \qed\end{aligned}
$$

# Test of Independence for Samples with Continuous CDF

Consider the i.i.d. pairs of random variables
$(X_1, Y_1), \dots, (X_n, Y_n)$ with some continuous distribution. While
each pair is independent, $X_i \perp \!\!\! \perp
X_j$ for $i \ne j$, we would like to test if
$X_i \perp \!\!\! \perp Y_i$ for all $i$.

Define the hypotheses

$$
\begin{aligned}
  H_0 &\colon X_1 \perp \!\!\! \perp Y_1 \\
  H_1 &\colon X_1 \centernot\perp \!\!\! \perp Y_1.\end{aligned}
$$

For
$i = 1, \dots, n$, let $R_i$ be the *rank* of $X_i$ in the sample $X_1,
\dots, X_n$. The rank function is defined as

$$
R_i = \operatorname{rank}(X_i) \coloneqq \#\{j \colon X_j \le X_i\}
$$

*i.e. *if $X_i = \min_j X_j$, then $R_i = 1$, and if $X_i = \max_j
X_j$, then $R_j = n$. Similarly, let $Q_i$ be the rank of $Y_i$ in
$Y_1, \dots, Y_n$.

## Example Experiment

An example experiment in which testing for independence of two
continuous random variables is important is in a scientific experiment
in which two devices are measured using sensors powered by the same
circuitry. We would like to ensure that the measurements are not
correlated.

## Dependence of the Ranks

The ranks $R_1, \dots, R_n$ are *not* independent because the rank of
$X_i$ in the sample is unique. Therefore, if you know that $R_1 = 1$,
*e.g.*, then $R_2,\dots,R_n \ne 1$.

## Proof of Distribution of Ranks

<div class="prop">

**Proposition 17**. *The distribution of the vector $(R_1, \dots, R_n)$
does *not* depend on the distribution of $X_i$’s (and, similarly, the
distribution of $(Q_1,
  \dots, Q_n)$ does not depend on the distribution of $Y_i$’s).*

</div>

<div class="proof">

*Proof.* Given the definition of $R_i$

$$
R_i = \operatorname{rank}(X_i) \coloneqq \#\{j \colon X_j \le X_i
$$

we
can determine its distribution. Let

$$
F_n(t) = \frac{1}{n}\sum_{i=1}^{n} \mathbbm{1}\!\left\{X_i \le t\right\}
$$

for some $t \in \mathbb{R}$ be the empirical cdf of the sample $X_i$’s.
We can then rewrite the definition of the rank as

$$
R_i = \sum_{j=1}^{n} \mathbbm{1}\!\left\{X_j \le X_i\right\} = n F_n(X_i).
$$

The cdf of $R_i$, $F_R(t)$ is then

$$
\begin{aligned}
    \mathbb{P}\left[R_i \le t\right] &= \mathbb{P}\left[n F_n(X_i) \le t\right] \\
                     &= \mathbb{P}\left[X_i \le F_n^{-1}\left(\tfrac{t}{n}\right)\right] \\
                     &= F_n\left(F_n^{-1}\left(\tfrac{t}{n}\right)\right)
  \end{aligned}
$$

Since $F_n$ is a piecewise-constant function,
$F_n^{-1}$ does not exist. To avoid ambiguity, define

$$
F_n^{-1}(p) = \inf\{x \colon F_n(x) \ge p\}
$$

where $p \in (0, 1)$.
Given the definition of the empirical cdf, its inverse can only take the
discrete values $X_1, \dots, X_n$, so

$$
F_n\left(F_n^{-1}\left(\tfrac{t}{n}\right)\right) 
      = \frac{\floor{t}}{n} \quad t \in [0, n]
$$

Therefore,

$$
\begin{aligned}
    \mathbb{P}\left[R_i \le t\right] &= \frac{\floor{t}}{n} \quad t \in [0, n] \\
                    &= \left\{0, \tfrac{1}{n}, \tfrac{2}{n}, \dots, 1\right\} \\
    \implies F_R(t) &= \left\{0, \tfrac{1}{n}, \tfrac{2}{n}, \dots, 1\right\} \\
    \implies f_R(t) &= \mathcal{U}(\{0,\dots,n\}) \\
    \implies F_R(t) &\perp \!\!\! \perp F_X(t) \qedhere
  \end{aligned}$$ ◻

</div>

An interpretation of this result is that $(R_1, \dots, R_n)$ is a
permutation of the vector $(1, \dots, n)$, with equal probability of any
permutation.

## The Null Hypothesis

<div class="prop">

**Proposition 18**. *If $H_0$ is true, then
$(R_1, \dots, R_n) \perp \!\!\! \perp(Q_1, \dots, Q_n)$.*

</div>

<div class="proof">

*Proof.* Under $H_0$, $X_i \perp \!\!\! \perp Y_i$, so

$$
\mathbb{P}\left[X = x \land Y = y\right] = \mathbb{P}\left[X=x\right]\mathbb{P}\left[Y=y\right]
$$

Since the empirical cdfs $F_n$ and $G_n$ of $X_i$ and $Y_j$,
respectively, are monotonically increasing, we can apply them to every
value to get the expression

$$
\begin{split}
      &\mathbb{P}\left[nF_n(X) = nF_n(x) \land nG_n(Y) = nG_n(y)\right] \\
      &\quad = \mathbb{P}\left[nF_n(X)=nF_n(x)\right]\mathbb{P}\left[nG_n(Y)=nG_n(y)\right].
      \end{split}
$$

The rank $R_i = n F_n(X_i)$, and, similarly,
$Q_j = n G_n(Y_j)$. The non-random values $nF_n(x)$ and $nG_n(y)$ are
arbitrary integers in $(0, \dots, n)$, which we denote by $k$ and
$\ell$. Therefore,

$$
\begin{aligned}
    \mathbb{P}\left[R_i = k \land Q_j = \ell\right] &= \mathbb{P}\left[R_i = k\right]\mathbb{P}\left[Q_j = \ell\right] \\
    \implies R &\perp \!\!\! \perp Q. \qedhere
  \end{aligned}$$ ◻

</div>

## Conclusion Under the Null Hypothesis

Under $H_0$, we have just shown that $R \perp \!\!\! \perp Q$, so the
joint distribution of the $2n$ random variables
$R_1, \dots, R_n, Q_1, \dots, Q_n$ is a discrete uniform distribution on
$[0, n]$, and does not depend on the distributions of the $X$’s or
$Y$’s.

## The Test Statistic

Consider the test statistic

$$
T_n \coloneqq \frac{\displaystyle \sum_{i=1}^{n}(R_i - \overline{R}_n)(Q_i - \overline{Q}_n)}{\displaystyle \sqrt{\sum_{i=1}^{n} (R_i - \overline{R}_n)^2 \sum_{i=1}^{n} (Q_i - \overline{Q}_n)^2}}.
$$

This test statistic is the empirical correlation between the two
samples. If $H_0$ is true, then $T_n \to 0$. We will now show that $T_n$
has a much simpler expression.

### Simplifying the Rank Averages

<div id="prop:rn1" class="prop">

**Proposition 19**. *$$\overline{R}_n= \overline{Q}_n= \frac{n+1}{2}$$*

</div>

<div class="proof">

*Proof.* Because $R \perp \!\!\! \perp Q$ and $R, Q$ are each
permutations of $(1, \dots, n)$, the first equality is true.

$$
\begin{aligned}
{3}
    \overline{R}_n&\coloneqq \frac{1}{n}\sum_{i=1}^{n} R_i \\
           &= \frac{1}{n}\sum_{i=1}^{n} i &\quad&\text{(permutation of $(1, \dots, n)$)} \\
           &= \frac{1}{n}\sum_{i=1}^{n} Q_i &\quad&\text{($R \perp \!\!\! \perp Q$)} \\
    \implies \overline{R}_n&= \overline{Q}_n.
  \end{aligned}
$$

Recognize the average as a geometric series

$$
\begin{aligned}
{3}
    \overline{R}_n&= \frac{1}{n}\sum_{i=1}^{n} \\
           &= \frac{1}{n} \frac{n(n+1)}{2} &\quad&\text{(geometric series)} \\
   \implies \overline{R}_n= \overline{Q}_n&= \frac{n+1}{2}. \tag*{\qedhere}
  \end{aligned}$$ ◻

</div>

<div id="prop:rn2" class="prop">

**Proposition 20**.
*$$\sum_{i=1}^{n} (R_i - \overline{R}_n)^2 = \sum_{i=1}^{n} (Q_i - \overline{Q}_n)^2 = \frac{n(n^2 - 1)}{12}$$*

</div>

<div class="proof">

*Proof.* The first equality is true because $R, Q$ are each permutations
of $(1, \dots, n)$. The sum evaluates to

$$
\begin{aligned}
{3}
    \sum_{i=1}^{n} (R_i - \overline{R}_n)^2 &= \sum_{i=1}^{n} (R_i^2 - 2\overline{R}_nR_i + \overline{R}_n^2) \\
    &= \sum_{i=1}^{n} \left(i^2 - 2 i \frac{n+1}{2} + \left(\frac{n+1}{2}\right)^2 \right) \\
    &= \sum_{i=1}^{n} i^2 - (n+1)\sum_{i=1}^{n} i + \frac{(n+1)^2}{4} \sum_{i=1}^{n} 1 \\
    \intertext{Using the fact that $\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6}$,
      and the previous result,}
    &= \frac{n(n+1)(2n+1)}{6} - (n+1) \frac{n(n+1)}{2} + \frac{n(n+1)^2}{4} \\
    &= \frac{2n^3 + 3n^2 + n}{6} - \frac{n^3 + 2n^2 + n}{2} + \frac{n^3 + 2n^2 + n}{4}  \\
    &= \frac{1}{12} (4n^3 + 6n^2 + 2n - 6n^3 - 12n^2 - 6n + 3n^3 + 6n^2 + 3n) \\
    &= \frac{1}{12} (n^3 - n) \\
    &= \frac{n(n^2 - 1)}{12} \tag*{\qedhere}
  \end{aligned}$$ ◻

</div>

### The Test Statistic Simplified

<div class="prop">

**Proposition 21**. *The test statistic can be written as

$$
T_n = \frac{12}{n(n^2 - 1)} \sum_{i=1}^{n}R_iQ_i - \frac{3(n+1)}{n-1}.$$*

</div>

<div class="proof">

*Proof.* Plugging the expressions from
Propositions <a href="#prop:rn1" data-reference-type="ref" data-reference="prop:rn1">Proposition 19</a>
and <a href="#prop:rn2" data-reference-type="ref" data-reference="prop:rn2">Proposition 20</a>
into the definition of $T_n$,

$$
\begin{aligned}
    T_n &\coloneqq \frac{\displaystyle \sum_{i=1}^{n}(R_i - \overline{R}_n)(Q_i - \overline{Q}_n)}{\displaystyle \sqrt{\sum_{i=1}^{n} (R_i - \overline{R}_n)^2 \sum_{i=1}^{n} (Q_i - \overline{Q}_n)^2}} \\
    &= \frac{\displaystyle \sum_{i=1}^{n}\left(R_i - \frac{n+1}{2}\right)\left(Q_i - \frac{n+1}{2}\right)}{\displaystyle \sqrt{\frac{n(n^2 - 1)}{12} \frac{n(n^2 - 1)}{12}}} \\
    &= \frac{\displaystyle \sum_{i=1}^{n}\left(R_iQ_i 
                                - \frac{n+1}{2}(R_i + Q_i) 
                                + \left(\frac{n+1}{2}\right)^2 \right)}{\displaystyle \frac{n(n^2 - 1)}{12}} \\
    &= \frac{12}{n(n^2 - 1)} \left[ \sum_{i=1}^{n}R_iQ_i 
    + \left(\frac{n+1}{2}\right)^2  \sum_{i=1}^{n} 1
    - \frac{n+1}{2}\sum_{i=1}^{n}(R_i + Q_i) \right]
  \end{aligned}
$$

The first term matches the first term of the
proposition. The remaining two terms simplify to

$$
\begin{aligned}
    &\phantom{=} \frac{12}{n(n^2 - 1)} \left[ \left(\frac{n+1}{2}\right)^2  \sum_{i=1}^{n} 1
    - \frac{n+1}{2}\sum_{i=1}^{n}(R_i + Q_i) \right]  \\
    &= \frac{12}{n(n^2 - 1)} \left[ \frac{n(n+1)^2}{4} - \frac{n+1}{2}\sum_{i=1}^{n}2i \right] \\
    &= \frac{12}{n(n^2 - 1)} \left[ \frac{n(n+1)^2}{4} - (n+1)\frac{n(n+1)}{2} \right] \\
    &= \frac{12}{n(n+1)(n-1)} \left[ \frac{n(n+1)^2}{4} - \frac{n(n+1)^2}{2} \right] \\
    &= \frac{12}{n-1} \left[ \frac{n+1}{4} - \frac{n+1}{2} \right] \\
    &= \frac{12}{n-1} \left[ -\frac{n+1}{4} \right] \\
    &= \frac{3(n+1)}{n-1} \\
    \implies T_n &= \frac{12}{n(n^2 - 1)} \sum_{i=1}^{n}R_iQ_i - \frac{3(n+1)}{n-1}. \qedhere
  \end{aligned}$$ ◻

</div>

## The Distribution of the Test Statistic is Pivotal

<div class="prop">

**Proposition 22**. *If $H_0 \colon X \perp \!\!\! \perp Y$ is true,
then

$$
T_n \sim S_n = \frac{12}{n(n^2 - 1)} \sum_{i=1}^{n} R_i' Q_i' - \frac{3(n+1)}{n-1},
$$

where $(R_1', \dots, R_n')$ and $(Q_1', \dots, Q_n')$ are ranks of two
i.i.d. samples of $\mathcal{U}\left(\left[ 0, 1 \right]\right)$.*

</div>

<div class="proof">

*Proof.* Using the previous arguments, the proof is as follows:

-   By the argument in
    §<a href="#subsec:3" data-reference-type="ref" data-reference="subsec:3">4.3</a>,
    the distributions of $R_i', Q_i'$ do not depend on the distributions
    of the underlying random variables.

-   By the argument in
    §<a href="#subsec:4" data-reference-type="ref" data-reference="subsec:4">4.4</a>,
    the vector $(R_1', \dots, R_n')
          \perp \!\!\! \perp(Q_1', \dots, Q_n')$ under $H_0$, so their
    distribution is known, and is the same discrete uniform distribution
    as $R, Q$.

-   By the argument in
    §<a href="#subsec:6" data-reference-type="ref" data-reference="subsec:6">4.6</a>,
    $T_n = S_n(R, Q)$.

Therefore, by the Continuous Mapping Theorem, $T_n \sim S_n$. ◻

</div>

## Computing Quantiles of the Test Statistic

Let $\alpha \in (0, 1)$.
Algorithm <a href="#alg:sn_q" data-reference-type="ref" data-reference="alg:sn_q">[alg:sn_q]</a>
approximates the $(1-\alpha)$-quantile of $S_n$, $q_\alpha$.

$M, n \in \mathbb{N}$. $\alpha \in (0, 1)$. $q_\alpha \in [0, 1]$.
$S_v \gets$ empty array of size $M$ $R \gets$ random permutation of
$(1, \dots, n)$. $Q \gets$ random permutation of $(1, \dots, n)$.
$S_v^{(i)} \gets \frac{12}{n(n^2 - 1)} \langle R, Q \rangle - \frac{3(n+1)}{n-1}$
$S_{vs} \gets$ $j \gets \ceil*{M(1 - \alpha)}$ $S_{vs}^{(j)}$

## A Non-Asymptotic Hypothesis Test

Redefine the hypotheses in terms of the test statistic

$$
\begin{aligned}
  H_0 &\colon T_n = 0 \\
  H_1 &\colon T_n \ne 0\end{aligned}
$$

with

$$
T_n = \frac{12}{n(n^2 - 1)} \sum_{i=1}^{n}R_iQ_i - \frac{3(n+1)}{n-1}.
$$

Let $\alpha \in (0, 1)$. The hypothesis test is then

$$
\delta_\alpha = \mathbbm{1}\!\left\{|T_n| \ge \hat{q}_{\alpha/2}^{(n, M)}\right\}
$$

where $\hat{q}_{\alpha/2}^{(n, M)}$ is the estimated
$(1 - \alpha)$-quantile of $T_n$. The $p$-value for this test is

$$
\begin{aligned}
  \text{$p$-value} &\coloneqq \mathbb{P}\left[|Z| \ge |T_n|\right] \\
  &\approx \frac{\#\{j=1,\dots,M \colon |S_v^{(j)}| \ge |T_n|\}}{M}\end{aligned}
$$

where $S_v^{(j)}$ is the $j^{\text{th}}$ sample of $S_n$, as computed in
Algorithm <a href="#alg:sn_q" data-reference-type="ref" data-reference="alg:sn_q">[alg:sn_q]</a>.

A plot of the distribution of

$$
\frac{S_n^M - \overline{S}_n^M}{\sqrt{\operatorname{Var}\left(S_n^M\right)}}
$$

is shown in
Figure <a href="#fig:Sn" data-reference-type="ref" data-reference="fig:Sn">4</a>
in comparison to a standard normal. The underlying samples are each
taken from an arbitrary $\operatorname{Exp}(\lambda=1)$ to demonstrate
the independence of the test statistic distribution on the underlying
sample distribution. This test statistic is indeed normally distributed,
although the parameters of the distribution are not immediately apparent
from our derivation. Since the asymptotic distribution of the test
statistic is not readily found in theory, we rely on simulation via
Algorithm <a href="#alg:sn_q" data-reference-type="ref" data-reference="alg:sn_q">[alg:sn_q]</a>
to estimate the quantiles.

<figure>
<embed src="indep_dist.pdf" id="fig:Sn" style="width:95.0%" /><figcaption aria-hidden="true">Plot of the distribution of the test statistic <span class="math inline">\(S_n\)</span>, normalized to compare with a standard normal distribution.</figcaption>
</figure>

[^1]: Note that the two samples may have different sizes (if $n \ne m$).
