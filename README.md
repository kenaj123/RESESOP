Goal: Find solution f of multiple linear inverse problems $A_i f = g_i$.

Setting: Only noisy versions of g_i are available $||g_i - g_i^\delta|| < \delta_i$ (L2-norm).
         Further, there may only be access to inexact versions $A_i^\eta$ of forward operators: $||A_i - A_i^\eta|| \leq \eta_i$ (operator norm).

In this repository we provide some implementation of the RESESOP-Kaczmarz method presented in the article:
S. Blanke, B. Hahn and A. Wald; 
Inverse problems with inexact forward operator: Iterative regularization and application in dynamic imaging;
Inverse Problems, 36 (2020). 

Second, we present a differentiable loss function that can be used to train a Deep Image Prior, while taking into account the discrepancy between the inexact and exact forward operators $A^\eta$ and $A$, respectively. As a recall, the DIP approach seeks for a neural network $\varphi_\theta$ that maps a given random prior $z$ to the solution $f$ of the inverse problem(s) $A_i f = g^\delta_i$. Since only inexact versions $A^\eta_i$ of $A_i$ are available, we propose to train $\varphi_\theta$ by minimizing the following loss-function:

$\frac{1}{n} \sum_{i=1}^n \Big\vert \vert A_i^\eta \varphi_\theta(z) - g_i^\delta \vert^2 - c_i \Big\vert^2$

where $c\in \mathbb{R}^n_{\geq 0}$ is some discrepancy term describing the model uncertainty between $A_i$ and $A_i^\eta$. Ideally, $c_i^2$ should be close to $\vert A_i f - g_i^\delta \vert^2$.

Both implementations have been used in the article:
J. Gödeke and G. Rigaud;
Imaging based on Compton scattering: model uncertainty and data-driven reconstruction methods;
Inverse Problems, 39 (2023).
