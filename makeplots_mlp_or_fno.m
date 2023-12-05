model_one = load('predicted_Directstep_1024_lead1_skip100.mat');
model_two = load('predicted_PECstep_1024_lead1_skip100.mat');

model_three = load('predicted_Directstep_1024_lead1_tendency_skip100.mat');
model_four = load('predicted_PECstep_1024_tendency_lead1_skip100.mat');

% model_one = load('predicted_Directstep_1024_FNO_lead1_skip100.mat');
% model_two = load('predicted_PECstep_1024_FNO_lead1_skip100.mat');
% 
% model_three = load('predicted_Directstep_1024_FNO_lead1_tendency_skip100.mat');
% model_four = load('predicted_PECstep_1024_FNO_lead1_tendency_lambda10_skip100.mat');


% 
% figure(1)
% clf
% set(0, 'DefaultAxesFontSize', 20)
% 
% loglog([0:511],model_one.Truth_FFT_x(1,1:512),'Linewidth',4, 'DisplayName','Truth');
% hold on
% loglog([0:511],model_two.pred_FFT_x(1,1:512),'Linewidth',2, 'DisplayName','PEC Net');
% 
% loglog([0:511],model_one.pred_FFT_x(1,1:512),'Linewidth',2,'DisplayName','Direct Net');
% 
% loglog([0:511],model_four.pred_FFT_x(1,1:512),'Linewidth',2,'DisplayName','PEC spectral loss');
% 
% loglog([0:511],model_three.pred_FFT_x(1,1:512),'Linewidth',2, 'DisplayName','Direct spectral loss');
% legend(Location='southwest')
% title('Fspec X')
% 
% % temp = fft((model_three.prediction(2:end,:)-model_three.prediction(1:end-1,:))/1e-3);
% 
% figure(2)
% clf
% set(0, 'DefaultAxesFontSize', 20)
% loglog([0:511],model_two.Truth_FFT_dt(1,1:512),'-r','Linewidth',4);
% hold on;
% loglog([0:511],model_two.pred_FFT_dt(1,1:512),'Linewidth',2);
% loglog([0:511],model_one.pred_FFT_dt(1,1:512),'-k','Linewidth',2);
% loglog([0:511],model_three.pred_FFT_dt(1,1:512),'-r','Linewidth',2);
% loglog([0:511],model_four.pred_FFT_dt(1,1:512),'-b','Linewidth',2);
% 
% legend('Truth','PEC Net','Direct Net','Direct spectral loss','PEC spectral loss', Location='southwest')
% title('Fspec dt')
% 
% figure(3)
% clf
% plot(model_one.RMSE,'-black','DisplayName','Direct Step','LineWidth',4);
% hold on
% plot(model_two.RMSE,'DisplayName','PEC Step');
% plot(model_three.RMSE,'DisplayName','Direct Step spectral loss')
% plot(model_four.RMSE,'DisplayName','PEC Step spectral loss')
% legend(Location='northwest')
% % axis([1 150 -.5 5])
% 
% 
% Truth = model_one.Truth;
% prediction = model_one.prediction;
% 
% 
% set(0, 'DefaultAxesFontSize', 20)
% 
% fig4 = figure(4);
% clf
% x = linspace(-50,50,1024);
% subplot(2,2,1)
% plot(x,prediction(1,:),'b','Linewidth',2);hold on
% plot(x,Truth(1,:),'r','Linewidth',2)
% title(['Time Step ' num2str(1)])
% 
% subplot(2,2,2)
% plot(x,prediction(10,:),'b','Linewidth',2);hold on
% plot(x,Truth(10,:),'r','Linewidth',2)
% title(['Time Step ' num2str(1000)])
% 
% subplot(2,2,3)
% plot(x,prediction(100,:),'b','Linewidth',2);hold on
% plot(x,Truth(100,:),'r','Linewidth',2)
% title(['Time Step ' num2str(10000)])
% 
% 
% subplot(2,2,4)
% plot(x,prediction(1000,:),'b','Linewidth',2);hold on
% plot(x,Truth(1000,:),'r','Linewidth',2)
% title(['Time Step ' num2str(100000)])
% fig4.Position = [550 200 1000 600]; 
% sgtitle("Direct Step at multiple time values")
% Lgnd = legend('Direct Step','Truth');
% Lgnd.Position(1) = 0.01;
% Lgnd.Position(2) = .85;

figure(5);
clf
histogram(model_one.prediction(:,:), Normalization="pdf")
histogram(model_one.Truth(:,:), Normalization="pdf")
title("Direct step output PDF")

figure(6);
clf
histogram(model_two.prediction(:,:), Normalization="pdf")
title("PEC step output PDF")

figure(7);
clf
histogram(model_three.prediction(:,:), Normalization="pdf")
title("Direct step spectral loss output PDF")

figure(8);
clf
histogram(model_four.prediction(:,:), Normalization="pdf")
title("PEC step spectral loss output PDF")
