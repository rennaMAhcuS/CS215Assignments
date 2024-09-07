# Normal Sampling

## Task A

We are given the random variable $Y$, where $Y:= F_X(X)$, where $X$ is a continuous
random variable with its CDF as $F_X: \mathbb{R} \xrightarrow{} [0, 1]$.

We proceed by finding the CDF $F_Y$ of $Y$.

$F_Y(Y) = \text{P}(Y \le y) = \text{P}(F_X(X) \le y).$

As we know that $F_X$ is a continuous, a monotonically increasing and an invertible
function, we can write

$F_X(X) \le y \implies X \le F_X^{-1}(y)$

$F_Y(Y) = P(X \le F_X^{-1}(y))$

Now, by the definition of $F_X$,

$P(X \le F_X^{-1}(y)) = F_X(F_X^{-1}(y)) = y$

We have,

$F_Y(Y) = y$

## 