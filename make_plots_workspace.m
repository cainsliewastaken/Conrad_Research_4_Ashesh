% direct_step = load('predicted_Directstep_1024_lead1_skip100.mat');
% % direct_step = load('predicted_PECstep_1024_cascade_lead1_skip100.mat');
% PEC_step = load('predicted_PECstep_1024_lead1_skip100.mat');
% 
% 
% direct_step_FNO = load('predicted_Directstep_1024_FNO_lead1_skip100.mat');
% PEC_step_FNO = load('predicted_PECstep_1024_FNO_lead1_skip100.mat');

direct_step = load('predicted_Directstep_1024_tendency_lead1_skip100.mat');
PEC_step = load('predicted_PECstep_1024_tendency_lead1_skip100.mat');

direct_step_FNO = load('predicted_Directstep_1024_FNO_tendency_lead1_mode128_skip100.mat');
PEC_step_FNO = load('predicted_PECstep_1024_FNO_tendency_lead1_skip100.mat');



figure(1)
clf
set(0, 'DefaultAxesFontSize', 20)

loglog([0:511],direct_step.Truth_FFT_x(1,1:512),'Linewidth',4, 'DisplayName','Truth');
hold on
loglog([0:511],PEC_step.pred_FFT_x(1,1:512),'Linewidth',2, 'DisplayName','PEC Net');

loglog([0:511],direct_step.pred_FFT_x(1,1:512),'Linewidth',2,'DisplayName','Direct Net');

loglog([0:511],PEC_step_FNO.pred_FFT_x(1,1:512),'Linewidth',2,'DisplayName','PEC FNO');

loglog([0:511],direct_step_FNO.pred_FFT_x(1,1:512),'Linewidth',2, 'DisplayName','Direct FNO');
xticks(xticks*100)

legend(Location='southwest')
title('Fspec X')


figure(2)
clf
set(0, 'DefaultAxesFontSize', 20)
loglog([0:511],PEC_step.Truth_FFT_dt(1,1:512),'-r','Linewidth',4);
hold on;
loglog([0:511],PEC_step.pred_FFT_dt(1,1:512),'Linewidth',2);
loglog([0:511],direct_step.pred_FFT_dt(1,1:512),'-k','Linewidth',2);
loglog([0:511],direct_step_FNO.pred_FFT_dt(1,1:512),'-r','Linewidth',2);
loglog([0:511],PEC_step_FNO.pred_FFT_dt(1,1:512),'-b','Linewidth',2);

legend('Truth','PEC Net','Direct Net','Direct FNO','PEC FNO', Location='southwest')
title('Fspec dt')

figure(3)
clf
plot(direct_step.RMSE,'-black','DisplayName','Direct Step','LineWidth',4);
hold on
plot(PEC_step.RMSE,'DisplayName','PEC Step');
plot(direct_step_FNO.RMSE,'DisplayName','Direct Step FNO')
plot(PEC_step_FNO.RMSE,'DisplayName','PEC Step FNO')
legend(Location='northwest')
% axis([1 10 -.5 5])


Truth = direct_step.Truth;
prediction = direct_step_FNO.prediction;


set(0, 'DefaultAxesFontSize', 20)

fig4 = figure(4);
clf
x = linspace(-50,50,1024);
subplot(2,2,1)
plot(x,prediction(1,:),'b','Linewidth',2);hold on
plot(x,Truth(1,:),'r','Linewidth',2)
title(['Time Step ' num2str(1)])

subplot(2,2,2)
plot(x,prediction(10,:),'b','Linewidth',2);hold on
plot(x,Truth(10,:),'r','Linewidth',2)
title(['Time Step ' num2str(1000)])

subplot(2,2,3)
plot(x,prediction(100,:),'b','Linewidth',2);hold on
plot(x,Truth(100,:),'r','Linewidth',2)
title(['Time Step ' num2str(10000)])


subplot(2,2,4)
plot(x,prediction(1000,:),'b','Linewidth',2);hold on
plot(x,Truth(1000,:),'r','Linewidth',2)
title(['Time Step ' num2str(100000)])
fig4.Position = [550 200 1000 600]; 
sgtitle("Direct Step at multiple time values")
Lgnd = legend('PECstep','Truth');
Lgnd.Position(1) = 0.01;
Lgnd.Position(2) = .85;