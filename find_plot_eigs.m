direct_step_jacs = load('FNO_KS_Directstep_lead1.mat');
Euler_step_jacs = load('FNO_KS_Eulerstep_lead1.mat');
PEC_step_jacs = load('FNO_KS_PECstep_lead1.mat');


[v_direct, e_direct] = eig(squeeze(direct_step_jacs.Jacobian_mats(2,:,:)));
[e_direct, ind_d] = sort(diag(e_direct));


[v_Euler, e_Euler] = eig(squeeze(Euler_step_jacs.Jacobian_mats(1,:,:)));
[e_Euler, ind_e] = sort(diag(e_Euler));



[v_PEC, e_PEC] = eig(squeeze(PEC_step_jacs.Jacobian_mats(1,:,:)));
[e_PEC, ind_p] = sort(diag(e_PEC));

theta = linspace(-pi,pi,100);
x=cos(theta)+1*1i*sin(theta);

figure(1)
clf
set(0, 'DefaultAxesFontSize', 20)
plot(x,'r','Linewidth',2);
hold on;

plot(e_Euler,'ks', 'MarkerSize',10, 'MarkerFaceColor','k');
plot(e_direct,'co','MarkerSize',10,'MarkerFaceColor','c');
plot(e_PEC,'ro','MarkerSize',10,'MarkerFaceColor','r');

legend('Unit Circle','Euler','Direct','PEC',fontsize=10)
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')

figure(2)
clf
h = histogram(abs(e_direct),50);
p = histcounts(abs(e_direct),50,'Normalization','pdf');
binCenters = h.BinEdges + (h.BinWidth/2);
histogram(abs(e_direct))
legend('Direct Eigvals',fontsize=10)

figure(3)
h = histogram(real(e_Euler),50);
p = histcounts(real(e_Euler),50,'Normalization','pdf');
binCenters = h.BinEdges + (h.BinWidth/2);
%plot(h, p)
legend('Euler Eigvals',fontsize=10)

figure(4)
h = histogram(real(e_PEC),50);
p = histcounts(real(e_PEC),50,'Normalization','pdf');
binCenters = h.BinEdges + (h.BinWidth/2);
%plot(binCenters(1:end-1), p)
histogram(abs(e_PEC))
legend('PEC Eigvals',fontsize=10)
