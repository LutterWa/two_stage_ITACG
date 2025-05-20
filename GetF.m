function [L,D,B]=GetF(state)  % 速度坐标系下的升力计算公式
% 定义常量
S = 0.057;
g=9.80665;

state=num2cell(state);
[t, x, y, z, v, gamma, psi, alpha, beta, m]=deal(state{:});

% 动力学方程
rho=1.225;
clalpha = 49.056;
cd0 = 0.2604;
cdalpha=29.072;

cd = cd0 + cdalpha * alpha ^ 2;  % 阻力系数
cl = clalpha * alpha;  % 升力系数
cb = clalpha * beta;   % 侧向力系数

Q=0.5*rho*v^2;  % 动压

D = -cd * Q * S;  % 阻力
L = cl * Q * S;  % 升力
B = cb * Q * S;  % 侧向力
end