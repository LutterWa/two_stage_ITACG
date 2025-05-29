clear
clc
close all

tasks = {"tgo_anal", "tgo_resnet"};
task = tasks{2};

if task == "tgo_anal"
    load("mats/tgo_analytical_predict_monte.mat")
    y = [tgo_real', v_real'];
    y_ = [tgo_pred', v_pred'];
elseif task == "tgo_resnet"
    load("mats/test_dnn.mat")
    y(:,2) = y(:,2) * 5;
    y_(:,2) = y_(:,2) * 5;
end

outputs = {'tgo', 'velocity'};
for i=1:length(outputs)
    e = y(:,i)-y_(:,i);  % 计算预测误差
    metric(i).network=outputs{i};
    metric(i).mse = mean(e.^2);  % 统计均方误差
    metric(i).rmse= sqrt(metric(i).mse);  % 统计均方根误差
    metric(i).mae = mean(abs(e));  % 统计平均绝对误差
    metric(i).mape= mean(abs(e./max(y(:,i), 0.01)))*100;  % 统计平均绝对百分比误差
    metric(i).r2  = 1-sum(e.^2)/sum((mean(y(:,i))-y(:,i)).^2);  % 统计r2
end

figure(1),
ax1 = subplot(2,1,1); hold on
plot(y(:,1))
plot(y_(:,1))
title('剩余飞行时间预测结果');
ax2 = subplot(2,1,2);
plot(y(:,1)-y_(:,1))
title('剩余飞行时间预测误差');
linkaxes([ax1, ax2], 'x');

figure(2), hold on
ax1 = subplot(2,1,1); hold on
plot(y(:,2))
plot(y_(:,2))
title('飞行速度预测结果');
ax2 = subplot(2,1,2);
plot(y(:,2)-y_(:,2))
title('飞行速度预测误差');
linkaxes([ax1, ax2], 'x');