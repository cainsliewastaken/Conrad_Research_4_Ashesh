model_one = load('predicted_Directstep_1024_lead1_RMSE.mat');
model_two = load('predicted_PEC4step_1024_lead1_RMSE.mat');
model_three = load('predicted_PEC4step_1024_lead1_tendency_RMSE.mat');

model_four = load('predicted_Directstep_1024_FNO_lead1_V2_RMSE.mat');
model_five = load('predicted_PEC4step_1024_FNO_lead1_RMSE.mat');
model_six = load('predicted_PEC4step_1024_FNO_lead1_tendency_RMSE.mat');



t_final = 100;
t_range = linspace(0, t_final, 10000);

figure(3)
clf

xx = linspace(0,t_final,length(model_one.RMSE));
plot(xx, model_one.RMSE,'-black','DisplayName','Direct Step MLP');

hold on
xx = linspace(0,t_final,length(model_two.RMSE));
plot(xx, model_two.RMSE,'DisplayName','PEC4 Step MLP ');

xx = linspace(0,t_final,length(model_three.RMSE));
plot(xx, model_three.RMSE,'DisplayName','PEC4 Step MLP spectral loss')

xx = linspace(0,t_final,length(model_four.RMSE));
plot(xx, model_four.RMSE,'DisplayName','Direct Step FNO ')

xx = linspace(0,t_final,length(model_five.RMSE));
plot(xx, model_five.RMSE,'DisplayName','PEC Step FNO ');

xx = linspace(0,t_final,length(model_six.RMSE));
plot(xx, model_six.RMSE,'DisplayName','PEC Step FNO spectral loss')
legend(Location='northwest', FontSize=8)
axis([-.5 100 -.5 20])
hold off



model_one = load('predicted_implicit_PEC4step_1024_lead50_RMSE.mat');
model_two = load('predicted_PEC4step_1024_lead50_tendency_RMSE.mat');

model_three = load('predicted_implicit_PEC4step_1024_lead100_RMSE.mat');
model_four = load('predicted_PEC4step_1024_lead100_tendency_RMSE.mat');

% model_three = load('GNO_predicted_PEC4step_1024_lead1_RMSE.mat');
% model_four = load('GNO_predicted_PEC4step_1024_lead1_tendnecy_epoch52_RMSE.mat');

% figure(4)
% clf
% hold on
% t_final = 100;
% t_range = linspace(0, t_final, 10000);
% 
% xx = linspace(0,t_final,length(model_one.RMSE));
% plot(xx, model_one.RMSE,'DisplayName','Implicit PEC4 step dt 50','LineWidth',2);
% 
% xx = linspace(0,t_final,length(model_two.RMSE));
% plot(xx, model_two.RMSE,':','DisplayName','PEC4 step dt 50','LineWidth',2);
% 
% xx = linspace(0,t_final,length(model_three.RMSE));
% plot(xx, model_three.RMSE,'DisplayName','Implicit PEC4 step dt 100','LineWidth',2);
% 
% xx = linspace(0,t_final,length(model_four.RMSE));
% plot(xx, model_four.RMSE,"--",'DisplayName','PEC4 step dt 100','LineWidth',2);
% legend(Location='northwest')
% axis([-.01, 2, -.2, 1])
% hold off
