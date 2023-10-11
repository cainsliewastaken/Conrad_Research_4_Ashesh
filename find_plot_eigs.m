direct_step_jacs = load('MLP_KS_Directstep_tendency_lead1_jacs.mat');
PEC_step_jacs = load('MLP_KS_PECstep_tendency_lead1_jacs.mat');

[v_direct, e_direct] = eig(squeeze(direct_step_jacs.Jacobian_mats(2,:,:)));
[e_direct, ind_d] = sort(diag(e_direct));

[v_PEC, e_PEC] = eig(squeeze(PEC_step_jacs.Jacobian_mats(1,:,:)));
[e_PEC, ind_p] = sort(diag(e_PEC));

direct_step_jacs_FNO = load('FNO_KS_Directstep_tendency_lead1_jacs.mat');
PEC_step_jacs_FNO = load('FNO_KS_PECstep_tendency_lead1_jacs.mat');

[v_direct_FNO, e_direct_FNO] = eig(squeeze(direct_step_jacs_FNO.Jacobian_mats(2,:,:)));
[e_direct_FNO, ind_d_FNO] = sort(diag(e_direct_FNO));

[v_PEC_FNO, e_PEC_FNO] = eig(squeeze(PEC_step_jacs_FNO.Jacobian_mats(1,:,:)));
[e_PEC_FNO, ind_p_FNO] = sort(diag(e_PEC_FNO));





theta = linspace(-pi,pi,100);
x=cos(theta)+1*1i*sin(theta);

figure(1)
clf
set(0, 'DefaultAxesFontSize', 20)
plot(x,'r','Linewidth',2);
hold on;

plot(e_direct,'co','MarkerSize',10,'MarkerFaceColor','c');
plot(e_PEC,'ro','MarkerSize',10,'MarkerFaceColor','r');

plot(e_direct_FNO,'co','MarkerSize',10);
plot(e_PEC_FNO,'ro','MarkerSize',10);


legend('Unit Circle','Direct MLP','PEC MLP','Direct FNO','PEC FNO',fontsize=10)
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')

figure(2)
clf
h = histogram(abs(e_direct),50);
p = histcounts(abs(e_direct),50,'Normalization','pdf');
binCenters = h.BinEdges + (h.BinWidth/2);
histogram(abs(e_direct))
legend('Direct Eigvals',fontsize=10)


figure(4)
h = histogram(real(e_PEC),50);
p = histcounts(real(e_PEC),50,'Normalization','pdf');
binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC))
legend('PEC Eigvals',fontsize=10)
