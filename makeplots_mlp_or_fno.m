model_one = load('predicted_Directstep_1024_lead1.mat');
model_two = load('predicted_PECstep_1024_lead1.mat');

% model_three = load('predicted_Directstep_1024_lead1_tendency.mat');
model_four = load('predicted_PECstep_1024_tendency_lead1.mat');

% model_one = load('predicted_Directstep_1024_FNO_lead1.mat');
% model_two = load('predicted_PECstep_1024_FNO_lead1.mat');
% 
% % model_three = load('predicted_Directstep_1024_FNO_lead1_tendency.mat');
% model_four = load('predicted_PECstep_1024_FNO_lead1_tendency.mat');



figure(1)
clf
set(0, 'DefaultAxesFontSize', 20)

loglog([0:511],model_one.Truth_FFT_x(1,1:512),'Linewidth',4, 'DisplayName','Truth');
hold on
loglog([0:511],model_two.pred_FFT_x(1,1:512),'Linewidth',2, 'DisplayName','PEC Net');

loglog([0:511],model_one.pred_FFT_x(1,1:512),'Linewidth',2,'DisplayName','Direct Net');

loglog([0:511],model_four.pred_FFT_x(1,1:512),'Linewidth',2,'DisplayName','PEC spectral loss');

% loglog([0:511],model_three.pred_FFT_x(1,1:512),'Linewidth',2, 'DisplayName','Direct spectral loss');
legend(Location='southwest')
title('Fspec X')
saveas(figure(1), 'X_FFT_MLP')


figure(2)
clf
set(0, 'DefaultAxesFontSize', 20)
loglog([0:511],model_two.Truth_FFT_dt(1,1:512),'-r','Linewidth',4);
hold on;
loglog([0:511],model_two.pred_FFT_dt(1,1:512),'Linewidth',2);
loglog([0:511],model_one.pred_FFT_dt(1,1:512),'-k','Linewidth',2);
% loglog([0:511],model_three.pred_FFT_dt(1,1:512),'-r','Linewidth',2);
loglog([0:511],model_four.pred_FFT_dt(1,1:512),'-b','Linewidth',2);

legend('Truth','PEC Net','Direct Net','PEC spectral loss', Location='southwest')
title('Fspec dt')

saveas(figure(2), 'dt_FFT_MLP')


figure(3)
clf
plot(model_one.RMSE,'-black','DisplayName','Direct Step','LineWidth',2);
hold on
plot(model_two.RMSE,'DisplayName','PEC Step');
% plot(model_three.RMSE,'DisplayName','Direct Step spectral loss')
plot(model_four.RMSE,'DisplayName','PEC Step spectral loss')
legend(Location='northwest')
axis([1 300 -.5 5])


Truth = model_one.Truth;
prediction = model_one.prediction;


set(0, 'DefaultAxesFontSize', 20)

saveas(figure(3), 'RMSE_MLP')


fig4 = figure(4);
clf
x = linspace(-50,50,1024);
subplot(2,2,1)
plot(x,prediction(1,:),'b','Linewidth',2);hold on
plot(x,Truth(1,:),'r','Linewidth',2)
title(['Time Step ' num2str(1)])

subplot(2,2,2)
plot(x,prediction(1000,:),'b','Linewidth',2);hold on
plot(x,Truth(1000,:),'r','Linewidth',2)
title(['Time Step ' num2str(1000)])

subplot(2,2,3)
plot(x,prediction(10000,:),'b','Linewidth',2);hold on
plot(x,Truth(10000,:),'r','Linewidth',2)
title(['Time Step ' num2str(10000)])


subplot(2,2,4)
plot(x,prediction(100000,:),'b','Linewidth',2);hold on
plot(x,Truth(100000,:),'r','Linewidth',2)
title(['Time Step ' num2str(100000)])
fig4.Position = [550 200 1000 600]; 
sgtitle("Direct Step at multiple time values")
Lgnd = legend('Direct Step','Truth');
Lgnd.Position(1) = 0.01;
Lgnd.Position(2) = .85;
hold off

% saveas(figure(4), 'RMSE_MLP')



figure(5);
clf
hold on
histogram(model_one.prediction(:,:),10, Normalization="pdf")
histogram(model_one.Truth(:,:),10, Normalization="pdf")
title("Direct step output PDF")
legend("Prediction","Truth")
hold off

figure(6);
clf
hold on
histogram(model_two.prediction(:,:),10, Normalization="pdf")
histogram(model_two.Truth(:,:),10, Normalization="pdf")
title("PEC step output PDF")
legend("Prediction","Truth")
hold off

% figure(7);
% clf
% hold on
% histogram(model_three.prediction(:,:), Normalization="pdf")
% histogram(model_three.Truth(:,:), Normalization="pdf")
% title("Direct step spectral loss output PDF")
% legend("Prediction","Truth")
% hold off

figure(8);
clf
hold on
histogram(model_four.prediction(:,:), 10, Normalization="pdf")
histogram(model_four.Truth(:,:),10, Normalization="pdf")
title("PEC step spectral loss output PDF")
legend("Prediction","Truth")
hold off
