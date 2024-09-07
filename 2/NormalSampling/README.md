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

## Task B

To construct an algorithm $\mathcal{A}$ that transforms a uniformly random variable
$Y$ over $[0, 1]$ into a random variable with the same cumulative distribution function
(CDF) as $X$, we can use the **Inverse Transform Sampling** method.

### Solution Using Inverse Transform Sampling:

1. **Compute the Inverse CDF**:  
   Let $F_X(u)$ be the CDF of the target random variable $X$.  
   Find the inverse of this CDF, denoted as $F_X^{-1}(u)$.

2. **Generate a Uniform Random Variable**:  
   The input to the algorithm $\mathcal{A}$ is a sample $y$ drawn from the uniform
   distribution over $[0, 1]$.

3. **Transform Using the Inverse CDF**:  
   Set the output of the algorithm as $X = \mathcal{A}(y) = F_X^{-1}(y)$.

### Explanation:

By the properties of the inverse transform sampling, if $Y$ is uniform over
$[0, 1]$, then $F_X^{-1}(Y)$ will have the CDF $F_X(u)$.
Consequently, the random variable produced by $\mathcal{A}(y)$ will have the same
distribution as $X$.
The PDF follows from differentiating the CDF $F_X(u)$.