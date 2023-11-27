% Use these 6 models to compare MLP models
model_one = load('MLP_KS_Directstep_lead1_jacs.mat');
model_two = load('MLP_KS_PECstep_lead1_jacs.mat');

model_three = load('MLP_KS_Directstep_lead1_tendency_jacs.mat');
model_four = load('MLP_KS_PECstep_lead1_tendency_jacs.mat');

model_five = load('MLP_KS_Directstep_lead1_UNTRAINED_jacs.mat');
model_six = load('MLP_KS_PECstep_lead1_UNTRAINED_jacs.mat');


% Use these 6 models to compare FNO models
model_one = load('FNO_KS_Directstep_lead1_jacs.mat');
model_two = load('FNO_KS_PECstep_lead1_jacs.mat');

model_three = load('FNO_KS_Directstep_lead1_tendency_jacs.mat');
model_four = load('FNO_KS_PECstep_lead1_tendency_lambda10_jacs.mat');

model_five = load('FNO_KS_Directstep_lead1_UNTRAINED_jacs.mat');
model_six = load('FNO_KS_PECstep_lead1_UNTRAINED_jacs.mat');


[v_direct, e_direct] = eig(squeeze(model_one.Jacobian_mats(1,:,:)));
[e_direct, ind_d] = sort(diag(e_direct));

[v_PEC, e_PEC] = eig(squeeze(model_two.Jacobian_mats(1,:,:)));
[e_PEC, ind_p] = sort(diag(e_PEC));

[v_direct_sl, e_direct_sl] = eig(squeeze(model_three.Jacobian_mats(1,:,:)));
[e_direct_sl, ind_d_sl] = sort(diag(e_direct_sl));

[v_PEC_sl, e_PEC_sl] = eig(squeeze(model_four.Jacobian_mats(1,:,:)));
[e_PEC_sl, ind_p_sl] = sort(diag(e_PEC_sl));



[v_direct_un, e_direct_un] = eig(squeeze(model_five.Jacobian_mats(1,:,:)));
[e_direct_un, ind_d_un] = sort(diag(e_direct_un));

[v_PEC_un, e_PEC_un] = eig(squeeze(model_six.Jacobian_mats(1,:,:)));
[e_PEC_un, ind_p_un] = sort(diag(e_PEC_un));



figure(1)
clf
theta = linspace(-pi,pi,100);
x=cos(theta)+1*1i*sin(theta);
set(0, 'DefaultAxesFontSize', 20)
plot(x,'r','Linewidth',2);
hold on;

plot(e_direct,'co','MarkerSize',10,'MarkerFaceColor','c','DisplayName','Direct MLP');
plot(e_PEC,'ro','MarkerSize',10,'MarkerFaceColor','r','DisplayName','PEC MLP');
plot(e_direct_sl,'ro','MarkerSize',10);
plot(e_PEC_sl,'go','MarkerSize',10);

legend('Unit Circle','Direct Basic','PEC Basic','Direct Spectral loss','PEC Spectral loss',fontsize=10)
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')



figure(12)
clf
set(0, 'DefaultAxesFontSize', 20)
% plot(x,'r','Linewidth',2);
hold on;


plot(e_direct,'co','MarkerSize',10,'MarkerFaceColor','c');
plot(e_direct_sl,'ro','MarkerSize',10);
legend('Direct Basic','Direct Spectral loss',fontsize=10)

% legend('Unit Circle','Direct Basic','Direct Spectral loss',fontsize=10)
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')



figure(13)
clf
set(0, 'DefaultAxesFontSize', 20)
% plot(x,'r','Linewidth',2);
hold on;

plot(e_PEC,'ro','MarkerSize',10,'MarkerFaceColor','r');

plot(e_PEC_sl,'bo','MarkerSize',10);

% legend('Unit Circle','PEC Basic','PEC Spectral loss',fontsize=10)
legend('PEC Basic','PEC Spectral loss',fontsize=10)

xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')

% figure(14)
% clf
% set(0, 'DefaultAxesFontSize', 20)
% plot(x,'r','Linewidth',2);
% hold on;
% 
% plot(e_direct_FNO_un,'ro','MarkerSize',10,'MarkerFaceColor','r');
% 
% plot(e_PEC_FNO_un,'bo','MarkerSize',10);
% 
% legend('Unit Circle','Direct Untrained','PEC Untrained',fontsize=10)
% % legend('Direct Untrained','PEC Untrained',fontsize=10)
% 
% xlabel('$Re(\lambda)$','Interpreter','latex')
% ylabel('$Im(\lambda)$','Interpreter','latex')




figure(100)
clf
% h = histogram(abs(e_direct),50);
% p = histcounts(abs(e_direct),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct),100, Normalization="pdf")
hold on
histogram(abs(e_direct_un), 50,Normalization="pdf")
MarchenkoPasturLaw(100, 1024, 1024, abs(e_direct_un))
legend('Direct Eigvals','Direct untrained','Marchenko Pastur',fontsize=10)

figure(101)
clf
% h = histogram(abs(e_PEC),50);
% p = histcounts(abs(e_PEC),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC-1)/1e-3,100, Normalization="pdf")
hold on
histogram(abs(e_PEC_un-1)/1e-3,100, Normalization="pdf")
MarchenkoPasturLaw(101, 1024, 1024, abs(e_PEC_un-1)/1e-3)
legend('PEC Eigvals','PEC untrained','Marchenko Pastur',fontsize=10)

figure(102)
clf
% h = histogram(abs(e_direct),50);
% p = histcounts(abs(e_direct),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct_sl), 10,Normalization="pdf")
hold on
histogram(abs(e_direct_un),10, Normalization="pdf")
MarchenkoPasturLaw(102, 1024, 1024, abs(e_direct_un))
legend('Direct Spectral loss Eigvals','Direct untrained','Marchenko Pastur',fontsize=10)


figure(103)
clf
% h = histogram(abs(e_PEC),50);
% p = histcounts(abs(e_PEC),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC_sl-1)/1e-3,100, Normalization="pdf")
hold on
histogram(abs(e_PEC_un-1)/1e-3, 100,Normalization="pdf")
MarchenkoPasturLaw(103, 1024, 1024, abs(e_PEC_un-1)/1e-3)
legend('PEC Spectral loss Eigvals','PEC untrained','Marchenko Pastur',fontsize=10)

figure(3)
clf
% h = histogram(real(e_PEC),50);
% p = histcounts(real(e_PEC),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct),100,Normalization="pdf")
MarchenkoPasturLaw(3, 1024, 1024, abs(e_direct_un))
legend('Direct Basic Eigvals','Marchenko Pastur',fontsize=10)

% 
% 
% figure(4)
% clf
% % h = histogram(real(e_PEC),50);
% % p = histcounts(real(e_PEC),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC-1)/1e-3, 100, Normalization="pdf")
% MarchenkoPasturLaw(4, 1024, 1024, abs(e_PEC_un-1)/1e-3)
% 
% legend('PEC Basic Eigvals','Marchenko Pastur',fontsize=10)



% figure(5)
% clf
% % h = histogram(abs(e_direct_FNO),50);
% % p = histcounts(abs(e_direct_FNO),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_sl),100, Normalization="pdf")
% MarchenkoPasturLaw(5, 1024, 1024, abs(e_direct_un))
% 
% legend('Direct Spectral loss Eigvals','Marchenko Pastur',fontsize=10)


figure(6)
clf
hold on
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC-1)/1e-3, 100, Normalization="pdf")
histogram(abs(e_PEC_sl-1)/1e-3, 100, Normalization="pdf")
histogram(abs(e_PEC_un-1)/1e-3, 100, Normalization="pdf")
MarchenkoPasturLaw(6, 1024, 1024, abs(e_PEC_un-1)/1e-3)

legend('PEC Basic Eigvals','PEC Spectral loss Eigvals','PEC Untrained Eigvals','Marchenko Pastur',fontsize=10)
hold off

figure(7)
clf
hold on
% h = histogram(abs(e_direct_FNO),50);
% p = histcounts(abs(e_direct_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct), 100, Normalization="pdf")
histogram(abs(e_direct_sl), 100, Normalization="pdf")
histogram(abs(e_direct_un), 100, Normalization="pdf")
MarchenkoPasturLaw(7, 1024, 1024, abs(e_direct_un))

legend('Direct Basic Eigvals','Direct Spectral loss Eigvals','Direct Untrained Eigvals','Marchenko Pastur',fontsize=10)
hold off

% figure(8)
% clf
% % h = histogram(real(e_PEC_FNO),50);
% % p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_un-1)/1e-3, 100, Normalization="pdf")
% MarchenkoPasturLaw(8, 1024, 1024, abs(e_PEC_un-1)/1e-3)
% 
% legend('PEC Untrained Eigvals','Marchenko Pastur',fontsize=10)

% 
% figure(9)
% clf
% % h = histogram(real(e_PEC_FNO),50);
% % p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_un),100, Normalization="pdf")
% MarchenkoPasturLaw(9, 1024, 1024, abs(e_direct_un))
% 
% legend('Direct Untrained Eigvals','Marchenko Pastur',fontsize=10)


% figure(10)
% clf
% % h = histogram(real(e_PEC_FNO),50);
% % p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_un-1)/1e-3, 100, Normalization="pdf")
% MarchenkoPasturLaw(10, 1024, 1024, abs(e_PEC_un-1)/1e-3)
% legend('PEC Basic Eigvals','Marchenko Pastur',fontsize=10)
