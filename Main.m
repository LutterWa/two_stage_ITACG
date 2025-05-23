% 清除环境变量
clear
clc

% 常量
RAD = 180/pi;
S = 0.057;
g=9.80665;
%          x       y      z     v   gamma      psi        qd      m    d
ini_min = [-20000, 5000, -5000, 400, -30/RAD, -30/RAD, -89, -30, 84.6, 600];
ini_max = [-10000, 10000, 5000, 600,  30/RAD,  30/RAD, -30,  30, 84.6, 8000];

ktraj = 50000;

para = latin(ktraj, length(ini_min), ini_min, ini_max);

para(:,7) = round(para(:,7));
para(:,8) = round(para(:,8));
para(:,10) = round(para(:,10));

% 遍历所有组合
parfor i = 1:ktraj
    p = para(i,:);
    qgt = p(7) / RAD;
    qpt = p(8) / RAD;
    d = p(10);
    xtd = -d * cos(qgt)*cos(qpt);
    ytd = -d * sin(qgt);
    ztd = d * cos(qgt)*sin(qpt);

    % 初始化状态
    restate = [];
    state = [0, p(1), p(2), p(3), p(4), p(5), p(6), 0, 0, p(9)];  % 使用当前初始条件
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
            if abs(t-round(t, 1)) < 1e-2
                restate = [restate; [state, 1]];
            end
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
            if abs(t-round(t, 1)) < 1e-2
                restate = [restate; [state, 0]];
            end
        end

        % 动力学方程
        rho = 1.225
        Q=0.5 * rho * v ^ 2;  % 动压

        clalpha = 49.056;
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

        state = rk4(state);
    end
    % 保存结果到文件
    if R < 20
        parsave(i, p, restate);
    end
end

