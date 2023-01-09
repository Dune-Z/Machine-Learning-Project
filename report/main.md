## Notes

Denote the item, the user, and the parent item of a transaction $t$ by $i$, $u$, and $p$, respectively. We model the true size of the item $i$ as
$$
\mathbf{v}_i = \mathbf{v}_p + \epsilon_i\mathbf{d}_p,
$$
where $\mathbf{v}_p$ is the true size of the parent item, $\epsilon_i$ is the bias of item size relative to the parent item, and $\mathbf{d}_p$ represents the influence of size bias. We want to learn $\mathbf{v}_p$ and $\mathbf{d}_p$ for each parent item $p$. A simple approach for estimating $\mathbf{v}_p$ is to use the average of the true sizes of all users that have rent the parent item with $\epsilon_i = 0$. However, this approach is not accurate since the item can either be too small or too large for some users. To address this issue, we solve the following ordinal regression problem to learn $\mathbf{v}_p$ and $\mathbf{d}_p$:

$$
\begin{aligned}
\min_{\mathbf{w}, b_1, b_2, \mathbf{v}_p, \mathbf{d}_p} & \sum_{t\in \mathcal{T}} \ell(y_t, f(\mathbf{w}, \mathbf{v}_i)),\\
\text{s.t.} \quad & \mathbf{d}_p \ge \mathbf{0}.
\end{aligned}
\tag{1}
$$
where we define $f(\mathbf{w}, \mathbf{v}_i) = \mathbf{w}^\top(\mathbf{v}_i - \mathbf{v}_u)$ as the fitting score between the item and the user, and $\ell(y_t, f(\mathbf{w}, \mathbf{v}_i))$ is the total loss of 2 binary logistic classifiers:
$$
\ell(y_t, f(\mathbf{w}, \mathbf{v}_i)) = 
\begin{cases}
-\log (1+e^{f(\mathbf{w}, \mathbf{v}_i)-b_1}), & \text{if}\ \ y_t = \mathtt{small};\\
-\log (1+e^{-f(\mathbf{w}, \mathbf{v}_i)+b_1}) - \log (1+e^{f(\mathbf{w}, \mathbf{v}_i)-b_2}), & \text{if}\ \ y_t = \mathtt{true2size};\\
-\log (1+e^{-f(\mathbf{w}, \mathbf{v}_i)+b_2}), & \text{if}\ \ y_t = \mathtt{large}.
\end{cases}
$$
Note that we add the constraint $\mathbf{d}_p \ge \mathbf{0}$ in Prob. (1) to capture the monotonicity of the true size $\mathbf{v}_i$ with respect to the size bias $\epsilon_i$.
