function dstate=dery(state)
%微分方程的右函数
g=9.80665;
s=num2cell(state);
[t, x, y, z, v, gamma, psi, alpha, beta, m] = deal(s{:});
%% 运动学方程
dx=v*cos(gamma)*cos(psi);%x北向
dy=v*sin(gamma);%y天向
dz=-v*cos(gamma)*sin(psi);%z东向

%% 动力学方程
[L,D,B] = GetF(state);  % 计算气动力

dv = D/m-g*sin(gamma);  % 速度标量
dgamma=(L-m*g*cos(gamma))/(m*v);  % 弹道倾角
dpsi = -B/(m*v*cos(gamma));  % 弹道偏角

%其他方程
dt=1;
dalpha=0;
dbeta=0;
dm=0;

dstate=[dt,dx,dy,dz,dv,dgamma,dpsi,dalpha,dbeta,dm];
end
