function [L,D,B]=GetF(state, atm, Ma2)  % 速度坐标系下的升力计算公式
% 定义常量
S = 0.057;
g=9.80665;

state=num2cell(state);
[t, x, y, z, v, gamma, psi, alpha, beta, m]=deal(state{:});

% 动力学方程
rho=interp1(atm(:,1), atm(:,2), min(max(y,0),80000));%大气密度
sonic=interp1(atm(:,1), atm(:,3), min(max(y,0),80000));%音速
ma=v/sonic;%马赫数

cd0 = interp1(Ma2(:,1), Ma2(:,3), max(min(ma, 1.5),0.4));
cdalpha = interp1(Ma2(:,1), Ma2(:,4), max(min(ma, 1.5),0.4));
clalpha = interp1(Ma2(:,1), Ma2(:,2), max(min(ma, 1.5),0.4));

cd = cd0 + cdalpha * alpha ^ 2;  % 阻力系数
cl = clalpha * alpha;  % 升力系数
cb = clalpha * beta;   % 侧向力系数

Q=0.5*rho*v^2;  % 动压

D = -cd * Q * S;  % 阻力
L = cl * Q * S;  % 升力
B = cb * Q * S;  % 侧向力
end