direct_step_jacs = load('MLP_KS_Directstep_lead1_jacs.mat');
PEC_step_jacs = load('MLP_KS_PECstep_lead1_jacs.mat');

direct_step_jacs_FNO = load('FNO_KS_Directstep_lead1_jacs.mat');
PEC_step_jacs_FNO = load('FNO_KS_PECstep_lead1_jacs.mat');

 
% direct_step_jacs = load('MLP_KS_Directstep_tendency_lead1_jacs.mat');
% PEC_step_jacs = load('MLP_KS_PECstep_tendency_lead1_jacs.mat');
% 
% direct_step_jacs_FNO = load('FNO_KS_Directstep_tendency_lead1_jacs.mat');
% PEC_step_jacs_FNO = load('FNO_KS_PECstep_tendency_lead1_jacs.mat');


[v_direct, e_direct] = eig(squeeze(direct_step_jacs.Jacobian_mats(1,:,:)));
[e_direct, ind_d] = sort(diag(e_direct));

[v_PEC, e_PEC] = eig(squeeze(PEC_step_jacs.Jacobian_mats(1,:,:)));
[e_PEC, ind_p] = sort(diag(e_PEC));

[v_direct_FNO, e_direct_FNO] = eig(squeeze(direct_step_jacs_FNO.Jacobian_mats(1,:,:)));
[e_direct_FNO, ind_d_FNO] = sort(diag(e_direct_FNO));

[v_PEC_FNO, e_PEC_FNO] = eig(squeeze(PEC_step_jacs_FNO.Jacobian_mats(1,:,:)));
[e_PEC_FNO, ind_p_FNO] = sort(diag(e_PEC_FNO));




direct_step_jacs_untrained = load('Model_output\MLP_KS_Directstep_tendency_lead1_UNTRAINED_jacs.mat');
PEC_step_jacs_untrained = load('MLP_KS_PECstep_lead1_UNTRAINED_jacs.mat');
 
direct_step_jacs_FNO_untrained = load('Model_output\FNO_KS_Directstep_lead1_UNTRAINED_jacs.mat');
PEC_step_jacs_FNO_untrained =load('Model_output\FNO_KS_PECstep_tendency_lead1_UNTRAINED_jacs.mat');


[v_direct_un, e_direct_un] = eig(squeeze(direct_step_jacs_untrained.Jacobian_mats(1,:,:)));
[e_direct_un, ind_d_un] = sort(diag(e_direct_un));

[v_PEC_un, e_PEC_un] = eig(squeeze(PEC_step_jacs_untrained.Jacobian_mats(1,:,:)));
[e_PEC_un, ind_p_un] = sort(diag(e_PEC_un));

[v_direct_FNO_un, e_direct_FNO_un] = eig(squeeze(direct_step_jacs_FNO_untrained.Jacobian_mats(1,:,:)));
[e_direct_FNO_un, ind_FNO_d] = sort(diag(e_direct_FNO_un));

[v_PEC_FNO_un, e_PEC_FNO_un] = eig(squeeze(PEC_step_jacs_FNO_untrained.Jacobian_mats(1,:,:)));
[e_PEC_FNO_un, ind_FNO_p] = sort(diag(e_PEC_FNO_un));

% 
% 
% figure(1)
% clf
% theta = linspace(-pi,pi,100);
% x=cos(theta)+1*1i*sin(theta);
% set(0, 'DefaultAxesFontSize', 20)
% plot(x,'r','Linewidth',2);
% hold on;
% 
% plot(e_direct,'co','MarkerSize',10,'MarkerFaceColor','c','DisplayName','Direct MLP');
% plot(e_PEC,'ro','MarkerSize',10,'MarkerFaceColor','r','DisplayName','PEC MLP');
% plot(e_direct_FNO,'go','MarkerSize',10);
% plot(e_PEC_FNO,'ro','MarkerSize',10);
% 
% legend('Unit Circle','Direct MLP','PEC MLP','Direct FNO','PEC FNO',fontsize=10)
% xlabel('$Re(\lambda)$','Interpreter','latex')
% ylabel('$Im(\lambda)$','Interpreter','latex')
% 
% 
% 
% figure(12)
% clf
% set(0, 'DefaultAxesFontSize', 20)
% plot(x,'r','Linewidth',2);
% hold on;
% 
% plot(e_direct,'co','MarkerSize',10,'MarkerFaceColor','c');
% 
% plot(e_direct_FNO,'ro','MarkerSize',10);
% 
% legend('Unit Circle','Direct MLP','Direct FNO',fontsize=10)
% xlabel('$Re(\lambda)$','Interpreter','latex')
% ylabel('$Im(\lambda)$','Interpreter','latex')
% 
% 
% 
% 
% figure(13)
% clf
% set(0, 'DefaultAxesFontSize', 20)
% % plot(x,'r','Linewidth',2);
% hold on;
% 
% plot(e_PEC,'ro','MarkerSize',10,'MarkerFaceColor','r');
% 
% plot(e_PEC_FNO,'bo','MarkerSize',10);
% 
% % legend('Unit Circle','PEC MLP','PEC FNO',fontsize=10)
% legend('PEC MLP','PEC FNO',fontsize=10)
% 
% xlabel('$Re(\lambda)$','Interpreter','latex')
% ylabel('$Im(\lambda)$','Interpreter','latex')
% 
% 
% 
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
% legend('Unit Circle','Direct FNO Untrained','PEC FNO Untrained',fontsize=10)
% % legend('Direct FNO Untrained','PEC FNO Untrained',fontsize=10)
% 
% xlabel('$Re(\lambda)$','Interpreter','latex')
% ylabel('$Im(\lambda)$','Interpreter','latex')




figure(100)
clf
% h = histogram(abs(e_direct),50);
% p = histcounts(abs(e_direct),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct), Normalization="pdf")
hold on
histogram(abs(e_direct_un), Normalization="pdf")
MarchenkoPasturLaw(100, 1024, 1024, abs(e_direct))
legend('Direct Eigvals','Direct untrained','Marchenko Pastur',fontsize=10)

figure(101)
clf
% h = histogram(abs(e_PEC),50);
% p = histcounts(abs(e_PEC),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC-1)/1e-3, Normalization="pdf")
hold on
histogram(abs(e_PEC_un-1)/1e-3, Normalization="pdf")
legend('PEC Eigvals','PEC untrained','Marchenko Pastur',fontsize=10)
MarchenkoPasturLaw(101, 1024, 1024, abs(e_PEC-1)/1e-3)



figure(102)
clf
% h = histogram(abs(e_direct),50);
% p = histcounts(abs(e_direct),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct_FNO), Normalization="pdf")
hold on
histogram(abs(e_direct_FNO_un), Normalization="pdf")
MarchenkoPasturLaw(102, 1024, 1024, abs(e_direct_FNO))

legend('Direct FNO Eigvals','Direct FNO untrained',fontsize=10)


figure(103)
clf
% h = histogram(abs(e_PEC),50);
% p = histcounts(abs(e_PEC),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC_FNO-1)/1e-3, Normalization="pdf")
hold on
histogram(abs(e_PEC_FNO_un-1)/1e-3, Normalization="pdf")
MarchenkoPasturLaw(103, 1024, 1024, abs(e_PEC_FNO-1)/1e-3)
legend('PEC FNO Eigvals','PEC FNO untrained','Marchenko Pastur',fontsize=10)






figure(3)
clf
% h = histogram(real(e_PEC),50);
% p = histcounts(real(e_PEC),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct), Normalization="pdf")
MarchenkoPasturLaw(3, 1024, 1024, abs(e_direct))

legend('Direct Eigvals','Marchenko Pastur',fontsize=10)



figure(4)
clf
% h = histogram(real(e_PEC),50);
% p = histcounts(real(e_PEC),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC-1)/1e-3, Normalization="pdf")
MarchenkoPasturLaw(4, 1024, 1024, abs(e_PEC-1)/1e-3)

legend('PEC Eigvals','Marchenko Pastur',fontsize=10)



figure(5)
clf
% h = histogram(abs(e_direct_FNO),50);
% p = histcounts(abs(e_direct_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct_FNO), Normalization="pdf")
MarchenkoPasturLaw(5, 1024, 1024, abs(e_direct_FNO))

legend('Direct Eigvals FNO','Marchenko Pastur',fontsize=10)


figure(6)
clf
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC_FNO-1)/1e-3, Normalization="pdf")
MarchenkoPasturLaw(6, 1024, 1024, abs(e_PEC_FNO-1)/1e-3)

legend('PEC Eigvals FNO','Marchenko Pastur',fontsize=10)


figure(7)
clf
% h = histogram(abs(e_direct_FNO),50);
% p = histcounts(abs(e_direct_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct_FNO_un), Normalization="pdf")
MarchenkoPasturLaw(7, 1024, 1024, abs(e_direct_FNO_un))

legend('Direct Eigvals FNO Untrained','Marchenko Pastur',fontsize=10)


figure(8)
clf
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC_FNO_un-1)/1e-3, Normalization="pdf")
MarchenkoPasturLaw(8, 1024, 1024, abs(e_PEC_FNO_un-1)/1e-3)

legend('PEC Eigvals FNO Untrained','Marchenko Pastur',fontsize=10)


figure(9)
clf
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_direct_un), Normalization="pdf")
MarchenkoPasturLaw(9, 1024, 1024, abs(e_direct_un))

legend('Direct MLP Eigvals Untrained','Marchenko Pastur',fontsize=10)


figure(10)
clf
% h = histogram(real(e_PEC_FNO),50);
% p = histcounts(real(e_PEC_FNO),50,'Normalization','pdf');
% binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC_un-1)/1e-3, Normalization="pdf")
MarchenkoPasturLaw(10, 1024, 1024, abs(e_PEC_un-1)/1e-3)
legend('PEC MLP Eigvals Untrained','Marchenko Pastur',fontsize=10)


