# Linear Regression  

Given a dataset of input–output pairs $(x_i, y_i)$, we aim to find the **best linear function** that predicts $y$ from $x$:  

$$
y_i = \theta^T \mathbf{x}_i + n_i, \quad n_i \sim \mathcal{N}(0, \sigma^2)
$$

Hence, the conditional likelihood for a single observation is:  

$$
p_\theta(y_i | x_i) = \mathcal{N}(y_i \,|\, \theta^T \mathbf{x}_i, \sigma^2)
$$

---

## Joint Distribution  

Given i.i.d. examples, the joint likelihood over all examples is:  

$$
p_\theta(y|X) = \prod_{i=1}^n p_\theta(y_i | x_i)
$$

Taking the logarithm gives the **log-likelihood**:  

$$
\log p_\theta(y|X) = -\frac{1}{2\sigma^2} \|y - X\theta\|^2 + \text{const.}
$$

---

## Objective Function (MLE Formulation)  

The **maximum likelihood estimation (MLE)** problem becomes:  

$$
\theta^* = \arg\max_\theta \log p_\theta(y|X)
\quad \Leftrightarrow \quad
\theta^* = \arg\min_\theta \|y - X\theta\|^2
$$

Thus, the **least-squares objective** is:  

$$
L(y, \hat y) = \frac{1}{2} \|y - X\theta\|^2
$$

and the **closed-form solution** is:  

$$
\theta^* = (X^T X)^{-1} X^T y
$$

---

### 💡 Key Observations

- The Gaussian noise assumption leads to the **least squares (L2)** cost.  
- MLE finds the parameters $\theta$ that **maximize the likelihood** of observed $y$ given $x$.  
- The least squares objective is **convex** in $\theta$, ensuring a **unique global minimum**.  

**❓ How to estimate prediction uncertainty?**  
From the Gaussian model, the **variance of predictions** can be estimated as:  

$$
\mathrm{Var}(\hat y) = \sigma^2 (X^T X)^{-1}
$$

---

# Weighted Least Squares (WLS)

**Definition:** When measurement uncertainties differ across observations, i.e. $n_i \sim \mathcal{N}(0, \sigma_i^2)$, we use **Weighted Least Squares**.  

The corresponding optimization problem becomes:  

$$
\theta^* = \arg\min_\theta \sum_{i=1}^n 
\frac{1}{\sigma_i^2} (y_i - \theta^T x_i)^2
$$

In matrix form:  

$$
\theta^* = (X^T Q X)^{-1} X^T Q y, \quad 
Q = \mathrm{diag}\!\left(\tfrac{1}{\sigma_1^2}, \ldots, \tfrac{1}{\sigma_n^2}\right)
$$

### 💡 Key Observations
- Each data point is **weighted by its precision** $(1/\sigma_i^2)$.  
- Measurements with lower uncertainty contribute **more** to the fit.  
- This is equivalent to performing MLE with **heteroscedastic Gaussian noise**.

---

# Nonlinear Regression  

Linear regression assumes that $y$ depends **linearly** on both the inputs and parameters.  
In many real-world cases, this is too restrictive.  

**Idea:** Keep the parameters linear but apply a **nonlinear transformation** to the inputs.

$$
y_i = \sum_{k=1}^m \theta_k \, \phi_k(x_i)
$$

This yields the **Basis Function Model (BFM)**, which generalizes linear regression.

---

## Basis Function Model (BFM)

- The **basis functions** $\phi_k(\cdot)$ can be **fixed** (e.g., polynomial, Fourier, wavelet) or **learnable** (e.g., neural networks).  
- The model remains **linear in parameters** $\theta$, hence convex and solvable in closed form.

### Solution
Let $\Phi$ be the design matrix:
$$
\Phi = [\phi_1(x), \phi_2(x), \ldots, \phi_m(x)]
$$
Then:
$$
\theta^* = (\Phi^T \Phi)^{-1} \Phi^T y
$$

---

## Polynomial Regression  

Define each basis as a polynomial term:
$$
\phi_k(x_i) = x_i^k, \quad k = 0, \ldots, n
$$

Then:
$$
\Phi = [1, x, x^2, \ldots, x^n]
$$

and the closed-form solution is identical:
$$
\theta^* = (\Phi^T \Phi)^{-1} \Phi^T y
$$

### 💡 Key Observations
- Polynomial regression maps the input to a **higher-dimensional feature space**.  
- It can model nonlinear relationships but may lead to **overfitting** at high degrees.

---

## Radial Basis Function (RBF) Regression  

A **Radial Basis Function** measures the similarity between a point $x$ and a center $c$:  

$$
\phi(x, c) = \exp(-\gamma \|x - c\|^2)
$$

- The closer $x$ is to $c$, the higher the influence.  
- Centers $c$ can be **learned** via optimization or **selected** from training samples (memory-based methods).  

### 💡 Key Observations
- RBFs allow **localized**, nonlinear fitting.  
- With enough centers, they can **approximate any smooth function**.

---

# Robust Regression  

Ordinary Least Squares (OLS) is sensitive to **outliers** due to the squared error term.  
To mitigate this, we use **robust loss functions** that reduce the influence of large residuals.

Two main approaches:
1. Assume a different conditional distribution $p_\theta(y|x)$ (e.g., **Laplace** → L1 loss).  
2. Modify the loss function directly (e.g., **Huber Loss**).  

### Huber Loss  

$$
L_\delta(y, \hat y) =
\begin{cases}
\frac{1}{2}(y - \hat y)^2, & \text{if } |y - \hat y| \le \delta \\
\delta(|y - \hat y| - \frac{1}{2}\delta), & \text{otherwise}
\end{cases}
$$

- Quadratic near zero (like MSE), linear for large residuals (like MAE).  
- Less sensitive to outliers while remaining differentiable.

---

# Locally Estimated Scatterplot Smoothing (LOESS)

Instead of fitting one global model, **LOESS** fits many **local linear or quadratic regressions** around each point, weighting nearby points more heavily.

---

### Step-by-Step Procedure

1. **Estimate local coefficients** $\hat{\theta}$ via **weighted least squares (WLS)** around the target point $x_0$:  

   $$
   \hat{\theta} = \arg\min_\theta \sum_{i=1}^n \alpha_i \, (y_i - x_i^T \theta)^2
   $$

2. **Compute weights** $\alpha_i$ for each neighboring point using a **kernel function**. Common kernels include:

   **Tricube kernel:**
   $$
   \alpha_i = 
   \begin{cases} 
   (1 - |d_i|^3)^3, & |d_i| < 1 \\
   0, & |d_i| \ge 1
   \end{cases}, 
   \quad d_i = \frac{|x_i - x_0|}{h}
   $$

   **Exponential kernel:**
   $$
   \alpha_i = \exp\Big(-\frac{|x_i - x_0|}{h}\Big)
   $$

   - $h$ is the **bandwidth** or span, controlling how far neighboring points influence the local fit.  
   - $d_i$ is the **normalized distance** from the target point.

3. **Update predictions** using the local coefficients:  

   $$
   \hat{y}_0 = \hat{\theta}^T x_0
   $$

4. **Repeat** for all points in the dataset (optionally iterate until the estimates stabilize).

---

### 💡 Key Observations

- Captures **local trends** without assuming a global linearity.  
- Choice of kernel affects **smoothness and locality**:
  - **Tricube kernel**: compact support, only nearby points influence the fit.  
  - **Exponential kernel**: all points influence the fit, but distant points contribute less.  
- More flexible than global regression but **computationally expensive** for large datasets.
---

# Overfitting and Regularization  

When the model fits noise instead of true patterns, it **overfits**.  
To mitigate this, we add a **regularization term**, which can also be derived from a **Bayesian perspective**.

---

## Bayesian View of Linear Regression

Assume the standard linear regression model:

$$
y = X \theta + n, \quad n \sim \mathcal{N}(0, \sigma^2 I)
$$

The **likelihood** of the data is:

$$
p(y | X, \theta) = \mathcal{N}(y \,|\, X\theta, \sigma^2 I) \propto \exp\Big(-\frac{1}{2\sigma^2} \|y - X\theta\|^2 \Big)
$$

---

### Ridge Regression (L2 Regularization)

Assume a **Gaussian prior** on the parameters:

$$
\theta \sim \mathcal{N}(0, \tau^2 I) \quad \Rightarrow \quad 
p(\theta) \propto \exp\Big(-\frac{1}{2\tau^2} \|\theta\|^2 \Big)
$$

The **posterior** is proportional to:

$$
p(\theta | y, X) \propto p(y | X, \theta) \, p(\theta)
$$

Taking negative log posterior gives:

$$
-\log p(\theta | y, X) = \frac{1}{2\sigma^2} \|y - X\theta\|^2 + \frac{1}{2\tau^2} \|\theta\|^2 + \text{const.}
$$

Comparing with the **regularized LS objective**, set:

$$
\lambda = \frac{\sigma^2}{\tau^2} \quad \Rightarrow \quad 
L(y, \hat y) = \|y - X\theta\|^2 + \lambda \|\theta\|^2
$$

Closed-form solution:

$$
\theta^* = (X^T X + \lambda I)^{-1} X^T y
$$

**💡 Bayesian interpretation:** L2 regularization is equivalent to placing a **Gaussian prior on parameters**.

---

### Lasso Regression (L1 Regularization)

Assume a **Laplace (double-exponential) prior** on parameters:

$$
p(\theta) \propto \exp(-\lambda \|\theta\|_1)
$$

The posterior negative log gives:

$$
-\log p(\theta | y, X) = \|y - X\theta\|^2 + \lambda \|\theta\|_1 + \text{const.}
$$

- Encourages **sparsity**: many parameters are exactly zero.  
- No closed-form solution; typically solved using **convex optimization** (e.g., coordinate descent).  

**💡 Bayesian interpretation:** L1 regularization corresponds to a **Laplace prior** on the parameters.

---

### Summary

| Regularization | Prior on θ      | Effect on parameters      |
|----------------|----------------|---------------------------|
| Ridge (L2)     | Gaussian       | Shrinks coefficients      |
| Lasso (L1)     | Laplace        | Shrinks and sparsifies    |


---

# Recursive Least Squares (RLS)

RLS is an **online version of Weighted Least Squares** that updates the parameter estimate $\theta$ as new data arrives.

---

## Problem Formulation

Suppose at time $t-1$, we have an estimate $\theta_{t-1}$ obtained from previous data.  
At time $t$, we receive a new sample $(x_t, y_t)$.  

The goal is to minimize the **exponentially weighted least squares** cost:

$$
J_t(\theta) = \sum_{i=1}^{t} \lambda^{t-i} (y_i - x_i^T \theta)^2, \quad 0 < \lambda \le 1
$$

where $\lambda$ is the **forgetting factor** that gives less weight to older observations.

---

## Derivation of Update Formulas

1. **Define the covariance matrix**:

Let the **weighted covariance** of inputs up to time $t-1$ be:

$$
P_{t-1} = \left(\sum_{i=1}^{t-1} \lambda^{t-1-i} x_i x_i^T \right)^{-1}
$$

2. **Define the prediction error** for the new sample:

$$
e_t = y_t - x_t^T \theta_{t-1}
$$

3. **Compute the gain vector** \(K_t\) to adjust $\theta$:

We want an update of the form:

$$
\theta_t = \theta_{t-1} + K_t \, e_t
$$

The optimal gain \(K_t\) that **minimizes $J_t(\theta)$** is derived from the weighted LS solution:

$$
K_t = \frac{P_{t-1} x_t}{\lambda + x_t^T P_{t-1} x_t}
$$

4. **Update the covariance matrix** \(P_t\):

After incorporating the new sample:

$$
P_t = \frac{1}{\lambda} \Big( P_{t-1} - K_t x_t^T P_{t-1} \Big)
$$

---

## RLS Update Equations (Summary)

$$
\begin{aligned}
K_t &= \frac{P_{t-1} x_t}{\lambda + x_t^T P_{t-1} x_t} \\[1em]
\theta_t &= \theta_{t-1} + K_t \, (y_t - x_t^T \theta_{t-1}) \\[0.5em]
P_t &= \frac{(I - K_t x_t^T) P_{t-1}}{\lambda}
\end{aligned}
$$

---

### 💡 Key Observations

- Each update **incorporates one new sample** efficiently.  
- $\lambda$ controls how fast old information is "forgotten"; $\lambda = 1$ gives standard LS.  
- The derivation comes directly from **minimizing the exponentially weighted least squares cost**, applying the **matrix inversion lemma** to avoid recomputing the inverse from scratch.

