direct_step = load('predicted_directstep_1024_lead1_skip100.mat');
PEC_step = load('predicted_PECstep_1024_lead1_skip100.mat');



truth_dt = truth(2:end,:)-truth(1:end-1,:);
prediction_dt = prediction(2:end,:)-prediction(1:end-1,:);

pred_dt_fft = abs(fft(prediction_dt,[],2));
truth_dt_fft = abs(fft(truth_dt,[],2));



set(0, 'DefaultAxesFontSize', 20)

figure(1)

loglog([0:511],PEC_step.pred_FFT(1,1:512),'-k','Linewidth',2);hold on;
loglog([0:511],PEC_step.Truth_FFT(1,1:512),'-r','Linewidth',2);

legend('PEC Net','Truth', Location='southeast')

set(0, 'DefaultAxesFontSize', 20)

figure(12)

loglog([0:511],direct_step.pred_FFT(1,1:512),'-k','Linewidth',2);hold on;
loglog([0:511],direct_step.Truth_FFT(1,1:512),'-r','Linewidth',2);

legend('Direct Step Net','Truth', Location='southeast')



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

