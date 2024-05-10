Implementation of the RESESOP-Kaczmarz method presented in the article:
S. Blanke, B. Hahn and A. Wald; 
Inverse problems with inexact forward operator: Iterative regularization and application in dynamic imaging;
Inverse Problems, 36 (2020).

Goal of RESESOP-Kaczmarz: Find solution f of multiple linear inverse problems $A_i f = g_i$.
Setting: Only noisy versions of g_i are available $||g_i - g_i^\delta|| < \delta_i$ (L2-norm).
         Only inexact version of forward operators available: $||A_i - A_i^\eta|| \leq \eta_i$ (operator norm).

This implementation has been used in the article:
J. Gödeke and G. Rigaud;
Imaging based on Compton scattering: model uncertainty and data-driven reconstruction methods;
Inverse Problems, 39 (2023).
