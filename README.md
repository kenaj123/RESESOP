Implementation of the RESESOP-Kaczmarz method presented in the article:
S. Blanke, B. Hahn and A. Wald; 
Inverse problems with inexact forward operator: Iterative regularization and application in dynamic imaging;
Inverse Problems, 36 (2020).

Goal of RESESOP-Kaczmarz: Find solution f of multiple linear inverse problems $A_i f = g_i$.
Setting: Only noisy versions of g_i are available $||g_i - g_i^\delta|| < \delta_i$ (L2-norm).
         Further, there may only be access to inexact versions $A_i^\eta$ of forward operators: $||A_i - A_i^\eta|| \leq \eta_i$ (operator norm).

This implementation has been used in the article:
J. Gödeke and G. Rigaud;
Imaging based on Compton scattering: model uncertainty and data-driven reconstruction methods;
Inverse Problems, 39 (2023).

Furthermore, a Deep Image Prior approach is implemented, for solving inverse problems with inexact forward operators:
A neural network $\varphi_\theta$ is considered that should map a random input $z$ to the solution $f$ of the inverse problems $A_i f = g_i$.
Since only an inexact versions $A^\eta_i$ of $A_i$ are available, we propose to train $\varphi_\theta$ to minimize the following loss-function:
$\sum_i \Vert \vert A_i^\eta \varphi_\theta(z) - g^\delta \vert^2 - c \Vert^2,$ where $c$ is some discrepancy term describing the model uncertainty between $A$ and $A^\eta$.

Also this implementation has been used in the article:
J. Gödeke and G. Rigaud;
Imaging based on Compton scattering: model uncertainty and data-driven reconstruction methods;
Inverse Problems, 39 (2023).
