model_one = load('MLP_predicted_Directstep_RMSE.mat');
model_two = load('MLP_predicted_PECstep_RMSE.mat');
model_four = load('MLP_predicted_PECstep_tendency_RMSE.mat');
% 
% model_one = load('FNO_predicted_Directstep_RMSE.mat');
% model_two = load('FNO_predicted_PECstep_RMSE.mat');
% model_four = load('FNO_predicted_PECstep_tendency_RMSE.mat');



figure(3)
clf
plot(model_one.RMSE,'-black','DisplayName','Direct Step','LineWidth',2);
hold on
plot(model_two.RMSE,'DisplayName','PEC Step');
plot(model_four.RMSE,'DisplayName','PEC Step spectral loss')
legend(Location='northwest')
% axis([1 300 -.5 5])
hold off


model_one = load('predicted_implicit_PEC4step_1024_lead50_spectralloss_RMSE.mat');
model_two = load('predicted_PEC4step_1024_lead50_tendency_RMSE.mat');

model_three = load('predicted_implicit_PEC4step_1024_lead100_spectralloss_RMSE.mat');
model_four = load('predicted_PEC4step_1024_lead100_tendency_RMSE.mat');


figure(4)
clf
hold on
t_final = 100;
t_range = linspace(0, t_final, 10000);

xx = linspace(0,t_final,length(model_one.RMSE));
plot(xx, model_one.RMSE,'DisplayName','Implicit PEC4 step dt 50','LineWidth',2);

xx = linspace(0,t_final,length(model_two.RMSE));
plot(xx, model_two.RMSE,':','DisplayName','PEC4 step dt 50','LineWidth',2);

xx = linspace(0,t_final,length(model_three.RMSE));
plot(xx, model_three.RMSE,'DisplayName','Implicit PEC4 step dt 100','LineWidth',2);

xx = linspace(0,t_final,length(model_four.RMSE));
plot(xx, model_four.RMSE,"--",'DisplayName','PEC4 step dt 100','LineWidth',2);
legend(Location='northwest')
axis([-.01, 2, -.2, 1])
hold off
