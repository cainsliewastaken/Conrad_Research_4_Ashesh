Direct_struct = load('MLP_KS_Directstep_lead1_jacs_all_chkpts.mat');
PEC_struct = load('MLP_KS_PECstep_lead1_jacs_all_chkpts.mat');

direct_mat = cell2mat(struct2cell(Direct_struct));
PEC_mat = cell2mat(struct2cell(PEC_struct));
e_Direct_full = zeros(13,1024);
e_PEC_full = zeros(13,1024);

for i=1:31
    [~, e_direct] = eig(squeeze(direct_mat(i,:,:)));
    e_Direct_full(i,:) = sort(diag(e_direct));
    
    [~, e_PEC] = eig(squeeze(PEC_mat(i,:,:)));
    e_PEC_full(i,:) = sort(diag(e_PEC));
end

figure(1)
clf
theta = linspace(-pi,pi,100);
x=cos(theta)+1*1i*sin(theta);
set(0, 'DefaultAxesFontSize', 20)
plot(x,'Linewidth',1,'Color',[.7 .7 .7]);
hold on;

plot(e_Direct_full(1,:),'ko','MarkerSize',5,'MarkerFaceColor','k');
plot(e_Direct_full(31,:),'ro','MarkerSize',5);
% legend('Direct Epoch 0','Direct Epoch 60',fontsize=10)

legend('Unit Circle','Direct Epoch 0','Direct Epoch 60',fontsize=10)
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')
title("Direct step eigenvalues")

figure(2)
clf
set(0, 'DefaultAxesFontSize', 20)
hold on;

plot(e_PEC_full(1,:),'ko','MarkerSize',5,'MarkerFaceColor','k');
plot(e_PEC_full(31,:),'ro','MarkerSize',5);
legend('PEC Epoch 0','PEC Epoch 60',fontsize=10)

xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')
title("PEC step eigenvalues")

hold off
figure(3)
plot(max(abs(e_Direct_full(:,:)),[],2))
xlabel('Epoch')
ylabel('$|\lambda|$','Interpreter','latex')
title("Direct step")

figure(4)
plot(max(abs(e_PEC_full(:,:)),[],2))
xlabel('Epoch')
ylabel('$|\lambda|$','Interpreter','latex')
title('PEC step max eigenvalue')
