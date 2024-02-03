% Use these 6 models to compare MLP models
model_one = load('MLP_KS_Directstep_lead1_jacs.mat');
model_two = load('MLP_KS_PECstep_lead1_jacs.mat');

% model_three = load('MLP_KS_Directstep_lead1_tendency_jacs.mat');
model_four = load('MLP_KS_PECstep_lead1_tendency_jacs.mat');

model_five = load('MLP_KS_Directstep_lead1_UNTRAINED_jacs.mat');
model_six = load('MLP_KS_PECstep_lead1_UNTRAINED_jacs.mat');


% Use these 6 models to compare FNO models
model_seven = load('FNO_KS_Directstep_lead1_large_jacs.mat');
model_eight = load('FNO_KS_PECstep_lead1_large_jacs.mat');

% model_nine = load('FNO_KS_Directstep_lead1_tendency_large_jacs.mat');
model_ten = load('FNO_KS_PECstep_lead1_tendency_large_jacs.mat');

model_eleven = load('FNO_KS_Directstep_lead1_UNTRAINED_jacs.mat');
model_twelve = load('FNO_KS_PECstep_lead1_UNTRAINED_jacs.mat');

% Next models are the implicit methods
model_thirteen = load('KS_PEC4step_implicit_lead100_jacs.mat');
model_fourteen = load('KS_PEC4step_implicit_lead50_jacs.mat');

% MLP eigenvalue calculations
direct_MLP = squeeze(model_one.Jacobian_mats(1,:,:));
% direct_MLP = (direct_j*direct_j')/1024)
[v_direct_MLP, e_direct_MLP] = eig(direct_MLP);
[e_direct_MLP, ind_d_MLP] = sort(diag(e_direct_MLP));

PEC_MLP = squeeze(model_two.Jacobian_mats(1,:,:));
% PEC_MLP = (PEC_MLP*PEC_MLP')/1024;
[v_PEC_MLP, e_PEC_MLP] = eig(PEC_MLP);
[e_PEC_MLP, ind_p_MLP] = sort(diag(e_PEC_MLP));

% [v_direct_sl, e_direct_sl] = eig(squeeze(model_three.Jacobian_mats(1,:,:)));
% [e_direct_sl, ind_d_sl] = sort(diag(e_direct_sl));

PEC_sl_MLP = squeeze(model_four.Jacobian_mats(1,:,:));
% PEC_sl_MLP = (PEC_sl_MLP*PEC_sl_MLP')/1024;
[v_PEC_sl_MLP, e_PEC_sl_MLP] = eig(PEC_sl_MLP);
[e_PEC_sl_MLP, ind_p_sl_MLP] = sort(diag(e_PEC_sl_MLP));

direct_un_MLP= squeeze(model_five.Jacobian_mats(1,:,:));
% direct_un_MLP = (direct_un_j*direct_un_j')/1024
[v_direct_un_MLP, e_direct_un_MLP] = eig(direct_un_MLP);
[e_direct_un_MLP, ind_d_un_MLP] = sort(diag(e_direct_un_MLP));

PEC_un_MLP = squeeze(model_six.Jacobian_mats(1,:,:));
% PEC_un_MLP = (PEC_un_MLP*PEC_un_MLP')/1024;
[v_PEC_un_MLP, e_PEC_un_MLP] = eig(PEC_un_MLP);
[e_PEC_un_MLP_MLP, ind_p_un_MLP] = sort(diag(e_PEC_un_MLP));

% FNO eigenvalue calculations
direct_FNO = squeeze(model_seven.Jacobian_mats(1,:,:));
% direct_FNO = (direct_j*direct_j')/1024);
[v_direct_FNO, e_direct_FNO] = eig(direct_FNO);
[e_direct_FNO, ind_d_FNO] = sort(diag(e_direct_FNO));

PEC_FNO = squeeze(model_eight.Jacobian_mats(1,:,:));
% PEC_FNO = (PEC_FNO*PEC_FNO')/1024;
[v_PEC_FNO, e_PEC_FNO] = eig(PEC_FNO);
[e_PEC_FNO, ind_p] = sort(diag(e_PEC_FNO));

% [v_direct_sl, e_direct_sl] = eig(squeeze(model_three.Jacobian_mats(1,:,:)));
% [e_direct_sl, ind_d_sl] = sort(diag(e_direct_sl));

PEC_sl_FNO = squeeze(model_ten.Jacobian_mats(1,:,:));
% PEC_sl_FNO = (PEC_sl_FNO*PEC_sl_FNO')/1024;
[v_PEC_sl_FNO, e_PEC_sl_FNO] = eig(PEC_sl_FNO);
[e_PEC_sl_FNO, ind_p_sl_FNO] = sort(diag(e_PEC_sl_FNO));

direct_un__FNO= squeeze(model_eleven.Jacobian_mats(1,:,:));
% direct_un__FNO = (direct_un_j*direct_un_j')/1024
[v_direct_un_FNO, e_direct_un_FNO] = eig(direct_un__FNO);
[e_direct_un_FNO, ind_d_un_FNO] = sort(diag(e_direct_un_FNO));

PEC_un_FNO = squeeze(model_twelve.Jacobian_mats(1,:,:));
% PEC_un_FNO = (PEC_un_FNO*PEC_un_FNO')/1024;
[v_PEC_un_FNO, e_PEC_un_FNO] = eig(PEC_un_FNO);
[e_PEC_un_FNO, ind_p_un_FNO] = sort(diag(e_PEC_un_FNO));


% Implicit eigenvalue calculations
PEC4_Implicit_lead100_MLP = squeeze(model_thirteen.Jacobian_mats(1,:,:));
% PEC4_Implicit_lead100_MLP = (PEC4_Implicit_lead100_MLP*PEC4_Implicit_lead100_MLP')/1024;
[v_PEC4_Implicit_lead100_MLP, e_PEC4_Implicit_lead100_MLP] = eig(PEC4_Implicit_lead100_MLP);
[e_PEC4_Implicit_lead100_MLP, ind_PEC4_Implicit_lead100_MLP] = sort(diag(e_PEC4_Implicit_lead100_MLP));

PEC4_Implicit_lead50_MLP = squeeze(model_fourteen.Jacobian_mats(1,:,:));
% PEC4_Implicit_lead50_MLP = (PEC4_Implicit_lead50_MLP*PEC4_Implicit_lead50_MLP')/1024;
[v_PEC4_Implicit_lead50_MLP, e_PEC4_Implicit_lead50_MLP] = eig(PEC4_Implicit_lead50_MLP);
[e_PEC4_Implicit_lead50_MLP, ind_PEC4_Implicit_lead50_MLP] = sort(diag(e_PEC4_Implicit_lead50_MLP));


figure(1)
clf
theta = linspace(-pi,pi,10000);
x=cos(theta)+1*1i*sin(theta);
set(0, 'DefaultAxesFontSize', 20)
plot(x,'r','Linewidth',2);
hold on;

plot(e_direct_MLP,'co','MarkerSize',5,'MarkerFaceColor','c','DisplayName','Direct MLP');
plot(e_PEC_MLP,'ro','MarkerSize',5,'MarkerFaceColor','r','DisplayName','PEC MLP');
plot(e_PEC_sl_MLP,'go','MarkerSize',5);


plot(e_direct_FNO,'mo','MarkerSize',5,'DisplayName','Direct MLP');
plot(e_PEC_FNO,'ko','MarkerSize',5,'DisplayName','PEC MLP');
plot(e_PEC_sl_FNO,'bo','MarkerSize',5);


legend('Unit Circle','Direct Basic MLP','PEC Basic MLP','PEC Spectral loss MLP','Direct Basic FNO','PEC Basic FNO','PEC Spectral loss FNO',fontsize=10)
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')


figure(13)
clf
% set(0, 'DefaultAxesFontSize', 20)
% plot(x,'r','Linewidth',2);
hold on;

plot(e_PEC_MLP,'ro','MarkerSize',5);
plot(e_PEC_sl_MLP,'go','MarkerSize',5);



% plot(e_PEC_un_MLP,'co','MarkerSize',5);

plot(e_PEC_FNO,'ko','MarkerSize',5);
plot(e_PEC_sl_FNO,'bo','MarkerSize',5);
% plot(e_PEC_un_FNO,'ko','MarkerSize',5);

% legend('PEC Basic MLP','PEC Spectral loss MLP','PEC Untrained MLP','PEC Basic FNO','PEC Spectral loss FNO','PEC Untrained FNO',fontsize=10)

axis manual
plot(x,'r','Linewidth',2);
legend('PEC Basic MLP','PEC Spectral loss MLP','PEC Basic FNO','PEC Spectral loss FNO','Unit circle',fontsize=10)


xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')


figure(14)
clf
set(0, 'DefaultAxesFontSize', 20)
hold on;

plot(e_PEC4_Implicit_lead50_MLP,'ro','MarkerSize',5);

plot(e_PEC4_Implicit_lead100_MLP,'bo','MarkerSize',5);

axis manual
plot(x,'k','Linewidth',1);
legend('Implicit PEC4 lead 50','Implicit PEC4 lead 100','Unit Circle',fontsize=10)


xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')

% 
% 
% 
% figure(100)
% clf
% h = histogram(abs(e_direct_MLP),50);
% p = histcounts(abs(e_direct_MLP),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_MLP),1000)
% hold on
% histogram(abs(e_direct_un_MLP), 25)
% MarchenkoPasturLaw(100, 1024, 1024, abs(e_direct_un_MLP))
% legend('Direct Eigvals','Direct untrained','Marchenko Pastur',fontsize=10)
% 
% 
% 
% figure(101)
% clf
% h = histogram(abs(e_PEC_MLP),50);
% p = histcounts(abs(e_PEC_MLP),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_MLP),100, Normalization="pdf")
% hold on
% histogram(abs(e_PEC_un_MLP),100, Normalization="pdf")
% MarchenkoPasturLaw(101, 1024, 1024, abs(e_PEC_un_MLP))
% legend('PEC Eigvals','PEC untrained','Marchenko Pastur',fontsize=10)
% 
% 
% figure(102)
% clf
% % h = histogram(abs(e_direct),50);
% % p = histcounts(abs(e_direct),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_sl), 100,Normalization="pdf")
% hold on
% histogram(abs(e_direct_un_MLP),100, Normalization="pdf")
% MarchenkoPasturLaw(102, 1024, 1024, abs(e_direct_un_MLP))
% legend('Direct Spectral loss Eigvals','Direct untrained','Marchenko Pastur',fontsize=10)
% 
% 
% figure(103)
% clf
% h = histogram(abs(e_PEC_MLP),50);
% p = histcounts(abs(e_PEC_MLP),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_sl_MLP),100, Normalization="pdf")
% hold on
% histogram(abs(e_PEC_un_MLP), 100,Normalization="pdf")
% MarchenkoPasturLaw(103, 1024, 1024, abs(e_PEC_un_MLP))
% legend('PEC Spectral loss Eigvals','PEC untrained','Marchenko Pastur',fontsize=10)
% 
% 
% figure(3)
% clf
% h = histogram(real(e_PEC_MLP),50);
% p = histcounts(real(e_PEC_MLP),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_MLP),100,Normalization="pdf")
% MarchenkoPasturLaw(3, 1024, 1024, abs(e_direct_un_MLP))
% legend('Direct Basic Eigvals','Marchenko Pastur',fontsize=10)
% 
% 
% 
% figure(4)
% clf
% h = histogram(real(e_PEC_MLP),50);
% p = histcounts(real(e_PEC_MLP),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_MLP), 100, Normalization="pdf")
% MarchenkoPasturLaw(4, 1024, 1024, abs(e_PEC_un_MLP))
% 
% legend('PEC Basic Eigvals','Marchenko Pastur',fontsize=10)
% 
% 
% 
% figure(5)
% clf
% % h = histogram(abs(e_direct_FNO),50);
% % p = histcounts(abs(e_direct_FNO),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_sl),100, Normalization="pdf")
% MarchenkoPasturLaw(5, 1024, 1024, abs(e_direct_un_MLP))
% 
% legend('Direct Spectral loss Eigvals','Marchenko Pastur',fontsize=10)
% 
% 
% figure(6)
% clf
% hold on
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_MLP), 100, Normalization="pdf")
% histogram(abs(e_PEC_sl_MLP), 100, Normalization="pdf")
% histogram(abs(e_PEC_un_MLP), 100, Normalization="pdf")
% MarchenkoPasturLaw(6, 1024, 1024, abs(e_PEC_un_MLP))
% 
% legend('PEC Basic Eigvals','PEC Spectral loss Eigvals','PEC Untrained Eigvals','Marchenko Pastur',fontsize=10)
% hold off
% 
% 
% 
% figure(62)
% clf
% hold on
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_MLP), 1000)
% histogram(abs(e_PEC_sl_MLP), 1000)
% histogram(abs(e_PEC_un_MLP), 100, Normalization="pdf")
% MarchenkoPasturLaw(6, 1024, 1024, abs(e_PEC_un_MLP))
% 
% legend('PEC Basic Eigvals','PEC Spectral loss Eigvals',fontsize=10)
% hold off
% 
% 
% 
% 
% 
% figure(7)
% clf
% hold on
% % h = histogram(abs(e_direct_FNO),50);
% % p = histcounts(abs(e_direct_FNO),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_MLP), 100, Normalization="pdf")
% histogram(abs(e_direct_sl), 100, Normalization="pdf")
% histogram(abs(e_direct_un_MLP), 100, Normalization="pdf")
% MarchenkoPasturLaw(7, 1024, 1024, abs(e_direct_un_MLP))
% 
% legend('Direct Basic Eigvals','Direct Spectral loss Eigvals','Direct Untrained Eigvals','Marchenko Pastur',fontsize=10)
% hold off
% 
% figure(8)
% clf
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_un_MLP), 100, Normalization="pdf")
% MarchenkoPasturLaw(8, 1024, 1024, abs(e_PEC_un_MLP))
% 
% legend('PEC Untrained Eigvals','Marchenko Pastur',fontsize=10)
% 
% 
% figure(9)
% clf
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
% plot(binCenters(1:end-1), p)
% histogram(abs(e_direct_un_MLP),100, Normalization="pdf")
% MarchenkoPasturLaw(9, 1024, 1024, abs(e_direct_un_MLP))
% 
% legend('Direct Untrained Eigvals','Marchenko Pastur',fontsize=10)
% 
% 
% figure(10)
% clf
% % h = histogram(real(e_PEC_FNO),50);
% % p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% % binCenters = h.BinEdges + (h.BinWidth/2);
% %plot(binCenters(1:end-1), p)
% histogram(abs(e_PEC_un_MLP), 100, Normalization="pdf")
% MarchenkoPasturLaw(10, 1024, 1024, abs(e_PEC_un_MLP))
% legend('PEC Basic Eigvals','Marchenko Pastur',fontsize=10)
