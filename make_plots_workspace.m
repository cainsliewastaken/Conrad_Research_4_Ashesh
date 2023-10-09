direct_step = load('predicted_directstep_1024_lead1_skip100.mat');
PEC_step = load('predicted_PECstep_1024_lead1_skip100.mat');



set(0, 'DefaultAxesFontSize', 20)

figure(1)

loglog([0:511],PEC_step.pred_FFT_dx(1,1:512),'-k','Linewidth',2);hold on;
loglog([0:511],PEC_step.Truth_FFT_dx(1,1:512),'-r','Linewidth',2);
loglog([0:511],direct_step.pred_FFT_dx(1,1:512),'-k','Linewidth',2);
legend('PEC Net','Truth','Direct Net', Location='southeast')

set(0, 'DefaultAxesFontSize', 20)


figure(2)

loglog([0:511],PEC_step.pred_FFT_dt(1,1:512),'-k','Linewidth',2);hold on;
loglog([0:511],PEC_step.Truth_FFT_dt(1,1:512),'-r','Linewidth',2);
loglog([0:511],direct_step.pred_FFT_dt(1,1:512),'-k','Linewidth',2);
legend('PEC Net','Truth','Direct Net', Location='southeast')

set(0, 'DefaultAxesFontSize', 20)

figure(2)

plot(direct_step.RMSE);
hold on
plot(PEC_step.RMSE);
legend('Direct RMSE', 'PEC RMSE')





% 
% set(0, 'DefaultAxesFontSize', 20)
% 
% figure(2)
% 
% subplot(2,2,1)
% plot(x,prediction(1,:),'b','Linewidth',2);hold on
% plot(x,Truth(1,:),'r','Linewidth',2)
% title(['Time Step' num2str(1)])
% 
% subplot(2,2,2)
% plot(x,prediction(10,:),'b','Linewidth',2);hold on
% plot(x,Truth(10,:),'r','Linewidth',2)
% title(['Time Step' num2str(10)])
% 
% subplot(2,2,3)
% plot(x,prediction(100,:),'b','Linewidth',2);hold on
% plot(x,Truth(100,:),'r','Linewidth',2)
% title(['Time Step' num2str(100)])
% 
% 
% subplot(2,2,4)
% plot(x,prediction(1000,:),'b','Linewidth',2);hold on
% plot(x,Truth(1000,:),'r','Linewidth',2)
% title(['Time Step' num2str(1000)])

