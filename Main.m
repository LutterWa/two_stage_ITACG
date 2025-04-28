% 清除环境变量
clear
clc

% 常量
RAD = 180/pi;
S = 0.057;
g=9.80665;
atm = load('atm2.txt');
Ma2 = [[0.4, 49.056, 0.2604, 29.072];
    [0.6, 49.468, 0.2635, 29.242];
    [0.8, 50.801, 0.2682, 30.351];
    [0.9, 51.372, 0.2776, 31.735];
    [1.0, 51.878, 0.2804, 33.014];
    [1.1, 52.468, 0.2797, 32.801];
    [1.2, 51.531, 0.2784, 32.656];
    [1.3, 51.224, 0.2771, 32.593];
    [1.4, 50.732, 0.2768, 32.442];
    [1.5, 50.321, 0.2707, 32.218]];

% 参数范围
velocity_values = 400:100:600;  % 速度
gamma_values = (-30:30:30) / RAD;  % 弹道倾角
psi_values = (-30:30:30) / RAD;  % 弹道偏角
x_values = -10000:-5000:-15000;  % 射程
y_values = 5000:5000:10000;  % 高度
z_values = -2000:2000:2000;  % 侧向
qgt_values = (-80:20:-40) / RAD;  % 俯仰方向期望落角
qpt_values = (-20:20:20) / RAD;  % 偏航方向期望落角
d_values = 600:900:6000; % 交班点到落点距离

disp(length(velocity_values)* ...
    length(gamma_values)* ...
    length(psi_values)* ...
    length(x_values)* ...
    length(y_values)* ...
    length(z_values)* ...
    length(qgt_values)* ...
    length(qpt_values)* ...
    length(d_values));  % 采样弹道条数

% 遍历所有组合
for v0 = velocity_values
    for gamma0 = gamma_values
        for psi0 = psi_values
            for x0 = x_values
                for y0 = y_values
                    for z0 = z_values
                        for qgt = qgt_values
                            for qpt = qpt_values
                                % 遍历 d 值
                                parfor i = 1:length(d_values)
                                    d = d_values(i) * -80 / round(qgt * RAD);
                                    xtd = -d * cos(qgt)*cos(qpt);
                                    ytd = -d * sin(qgt);
                                    ztd = d * cos(qgt)*sin(qpt);

                                    % 初始化状态
                                    restate = [];

                                    state = [0, x0, y0, z0, v0, gamma0, psi0, 0, 0, 84.6]; % 使用当前初始条件
                                    s=num2cell(state);
                                    [t, x, y, z, v, gamma, psi, alpha, beta, m] = deal(s{:});

                                    % 初始计算
                                    r = [xtd, ytd, ztd] - [x, y, z];

                                    R = norm(r);
                                    q = [atan2( r(2), norm([r(1), r(3)]) );
                                        -atan2( r(3), r(1) )];
                                    eta = [gamma; psi] - q;  % 导弹速度前置角
                                    Rdot = -v * cos( eta(1) ) * cos( eta(2) );
                                    qdot = [-v * sin( eta(1) );
                                        v * cos( eta(1) ) * sin( eta(2) ) / cos( q(1) )] / R;

                                    while (t < 800)
                                        if (y < 0)  % || Rdot >= 0
                                            break;
                                        end

                                        % 状态变量更新
                                        s=num2cell(state);
                                        [t, x, y, z, v, gamma, psi, ~, ~, m] = deal(s{:});

                                        % 起点到交班点
                                        if abs(x) > abs(xtd)
                                            xt = xtd;
                                            yt = ytd;
                                            zt = ztd;

                                            r = [xt, yt, zt] - [x, y, z];
                                            R = norm(r);
                                            q = [atan2( r(2), norm([r(1), r(3)]) );
                                                -atan2( r(3), r(1) )];
                                            eta = [gamma; psi] - q;  % 导弹速度前置角
                                            Rdot = -v * cos( eta(1) ) * cos( eta(2) );
                                            qdot = [-v * sin( eta(1) );
                                                v * cos( eta(1) ) * sin( eta(2) ) / cos( q(1) )] / R;
                                            tgo = R / v;

                                            acg = 4 * v * qdot(1) + 2 * v * (q(1) - qgt) / tgo + g * cos(gamma);
                                            acp = 4 * v * cos(gamma) * qdot(2) + 2 * v * cos(gamma) * (qpt - q(2)) / tgo;

                                            restate = [restate; [state, 1]];
                                        else  % 交班点到落点
                                            xt = 0;
                                            yt = 0;
                                            zt = 0;

                                            r = [xt, yt, zt] - [x, y, z];
                                            R = norm(r);
                                            q = [atan2( r(2), norm([r(1), r(3)]) );
                                                -atan2( r(3), r(1) )];
                                            eta = [gamma; psi] - q;  % 导弹速度前置角
                                            Rdot = -v * cos( eta(1) ) * cos( eta(2) );
                                            qdot = [-v * sin( eta(1) );
                                                v * cos( eta(1) ) * sin( eta(2) ) / cos( q(1) )] / R;

                                            acg = 3 * v * qdot(1) + g * cos(gamma);
                                            acp = 3 * v * cos(gamma) * qdot(2);

                                            restate = [restate; [state, 0]];
                                        end

                                        % 动力学方程
                                        rho=interp1(atm(:,1), atm(:,2), min(max(y,0),80000));%大气密度
                                        sonic=interp1(atm(:,1), atm(:,3), min(max(y,0),80000));%音速
                                        ma = v / sonic;%马赫数
                                        Q=0.5 * rho * v ^ 2;  % 动压

                                        clalpha = interp1(Ma2(:,1), Ma2(:,2), max(min(ma, 1.5),0.4));
                                        alpha = (m * acg) / (Q * S * clalpha);
                                        beta = (m * acp) / (Q * S * clalpha);

                                        if (alpha > 15 / RAD)
                                            alpha = 15 / RAD;
                                        elseif (alpha < -15 / RAD)
                                            alpha = -15 / RAD;
                                        end

                                        if (beta > 15 / RAD)
                                            beta = 15 / RAD;
                                        elseif (beta < -15 / RAD)
                                            beta = -15 / RAD;
                                        end
                                        state(8) = alpha;
                                        state(9) = beta;

                                        state = rk4(state, atm, Ma2);
                                    end
                                    % 保存结果到文件
                                    if R < 20
                                        parsave(x0, y0, z0, v0, gamma0, psi0, qgt, qpt, d, restate);
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end