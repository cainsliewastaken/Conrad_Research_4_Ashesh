model_one = load('MLP_predicted_Directstep_RMSE.mat');
model_two = load('MLP_predicted_PECstep_RMSE.mat');

model_four = load('MLP_predicted_PECstep_tendnecy_RMSE.mat');

% model_one = load('FNO_predicted_Directstep_RMSE.mat');
% model_two = load('FNO_predicted_PECstep_RMSE.mat');
% 
% model_four = load('FNO_predicted_PECstep_tendency_RMSE.mat');

figure(3)
clf
plot(model_one.RMSE,'-black','DisplayName','Direct Step','LineWidth',2);
hold on
plot(model_two.RMSE,'DisplayName','PEC Step');
plot(model_four.RMSE,'DisplayName','PEC Step spectral loss')
legend(Location='northwest')
% axis([1 300 -.5 5])