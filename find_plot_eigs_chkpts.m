Direct_struct = load('MLP_KS_Directstep_lead1_multi_jacs.mat');
PEC_struct = load('MLP_KS_PECstep_lead1_multi_jacs.mat');
PEC_specloss_struct = load('MLP_KS_PECstep_lead1_tendency_multi_jacs.mat');

Direct_struct_FNO = load('FNO_KS_Directstep_lead1_multi_jacs.mat');
PEC_struct_FNO = load('FNO_KS_PECstep_lead1_multi_jacs.mat');
PEC_specloss_struct_FNO = load('FNO_KS_PECstep_lead1_tendency_multi_jacs.mat');

direct_mat = cell2mat(struct2cell(Direct_struct));
PEC_mat = cell2mat(struct2cell(PEC_struct));
PEC_specloss_mat = cell2mat(struct2cell(PEC_specloss_struct));

direct_mat_FNO = cell2mat(struct2cell(Direct_struct_FNO));
PEC_mat_FNO = cell2mat(struct2cell(PEC_struct_FNO));
PEC_specloss_mat_FNO = cell2mat(struct2cell(PEC_specloss_struct_FNO));

num_jacs = size(direct_mat);
num_jacs  = num_jacs(1);

e_Direct_full = zeros(num_jacs,1024);
e_Direct_vals = [];

e_PEC_full = zeros(num_jacs,1024);
e_PEC_vals = [];

e_PEC_specloss_full = zeros(num_jacs,1024);
e_PEC_specloss_vals = [];

e_Direct_full_FNO = zeros(num_jacs,1024);
e_Direct_vals_FNO = [];

e_PEC_full_FNO = zeros(num_jacs,1024);
e_PEC_vals_FNO = [];

e_PEC_specloss_full_FNO = zeros(num_jacs,1024);
e_PEC_specloss_vals_FNO = [];

for i=1:num_jacs
    [~, e_direct] = eig(squeeze(direct_mat(i,:,:)));
    % e_Direct_full(i,:) = sort(diag(e_direct));
    e_Direct_vals = [e_Direct_vals; diag(e_direct)];

    [~, e_PEC] = eig(squeeze(PEC_mat(i,:,:)));
    % e_PEC_full(i,:) = sort(diag(e_PEC));
    e_PEC_vals = [e_PEC_vals; diag(e_PEC)];

    [~, e_PEC_specloss] = eig(squeeze(PEC_specloss_mat(i,:,:)));
    % e_PEC_specloss_full(i,:) = sort(diag(e_PEC));
    e_PEC_specloss_vals = [e_PEC_specloss_vals; diag(e_PEC_specloss)];

    [~, e_direct_FNO] = eig(squeeze(direct_mat_FNO(i,:,:)));
    % e_Direct_full_FNO(i,:) = sort(diag(e_direct_FNO));
    e_Direct_vals_FNO = [e_Direct_vals_FNO; diag(e_direct_FNO)];

    [~, e_PEC_FNO] = eig(squeeze(PEC_mat_FNO(i,:,:)));
    % e_PEC_full_FNO(i,:) = sort(diag(e_PEC_FNO));
    e_PEC_vals_FNO = [e_PEC_vals_FNO; diag(e_PEC_FNO)];

    [~, e_PEC_specloss_FNO] = eig(squeeze(PEC_specloss_mat_FNO(i,:,:)));
    % e_PEC_specloss_full_FNO(i,:) = sort(diag(e_PEC_FNO));
    e_PEC_specloss_vals_FNO = [e_PEC_specloss_vals_FNO; diag(e_PEC_specloss_FNO)];
end

% figure(1)
% clf
% theta = linspace(-pi,pi,100);
% x=cos(theta)+1*1i*sin(theta);
% set(0, 'DefaultAxesFontSize', 20)
% plot(x,'Linewidth',1,'Color',[.7 .7 .7]);
% hold on;
% 
% plot(e_Direct_full(1,:),'ko','MarkerSize',5,'MarkerFaceColor','k');
% plot(e_Direct_full(41,:),'ro','MarkerSize',5);
% % legend('Direct Epoch 0','Direct Epoch 60',fontsize=10)
% 
% legend('Unit Circle','Direct Epoch 0','Direct Epoch 60',fontsize=10)
% xlabel('$Re(\lambda)$','Interpreter','latex')
% ylabel('$Im(\lambda)$','Interpreter','latex')
% title("Direct step eigenvalues")
% 
% figure(2)
% clf
% set(0, 'DefaultAxesFontSize', 20)
% hold on;
% 
% plot(e_PEC_full(1,:),'ko','MarkerSize',5,'MarkerFaceColor','k');
% plot(e_PEC_full(41,:),'ro','MarkerSize',5);
% legend('PEC Epoch 0','PEC Epoch 60',fontsize=10)
% 
% xlabel('$Re(\lambda)$','Interpreter','latex')
% ylabel('$Im(\lambda)$','Interpreter','latex')
% title("PEC step eigenvalues")

% hold off
% figure(3)
% plot(max(abs(e_Direct_full(:,:)),[],2))
% xlabel('Epoch')
% ylabel('$|\lambda|$','Interpreter','latex')
% title("Direct step")
% 
% figure(4)
% plot(max(abs(e_PEC_full(:,:)),[],2))
% xlabel('Epoch')
% ylabel('$|\lambda|$','Interpreter','latex')
% title('PEC step max eigenvalue')


figure(5)
clf
hold on
histogram(abs(e_Direct_vals),100, Normalization="pdf")
histogram(abs(e_Direct_vals_FNO),100, Normalization="pdf")

% MarchenkoPasturLaw(5, 1024, 1024, abs(e_Direct_vals))

legend('Direct Eigvals','Direct FNO Eigvals',fontsize=10)
hold off

figure(6)
clf
hold on
histogram(abs( (e_PEC_vals-1) /1e-3),100, Normalization="pdf")
histogram(abs( (e_PEC_vals_FNO-1) /1e-3),100, Normalization="pdf")

% MarchenkoPasturLaw(6, 1024, 1024, abs(e_PEC_vals))

legend('PEC Eigvals','PEC FNO Eigvals',fontsize=10)
hold off

figure(7)
clf
hold on
histogram(abs( (e_PEC_specloss_vals-1) /1e-3),100, Normalization="pdf")
histogram(abs( (e_PEC_specloss_vals_FNO-1) /1e-3),100, Normalization="pdf")

% MarchenkoPasturLaw(7, 1024, 1024, abs( (e_PEC_specloss_vals-1) /1e-3) )

legend('PEC spectral loss Eigvals','PEC FNO spectral loss Eigvals',fontsize=10)
hold off

figure(8)
clf
hold on
histogram(abs(e_Direct_vals),100, Normalization="pdf")
histogram(abs( (e_PEC_vals-1) /1e-3),100, Normalization="pdf")
histogram(abs( (e_PEC_specloss_vals-1) /1e-3),100, Normalization="pdf")

histogram(abs(e_Direct_vals_FNO),100, Normalization="pdf")
histogram(abs( (e_PEC_vals_FNO-1) /1e-3),100, Normalization="pdf")
histogram(abs( (e_PEC_specloss_vals_FNO-1) /1e-3),100, Normalization="pdf")

% MarchenkoPasturLaw(8, 1024, 1024, abs(e_PEC_vals))

legend('Direct Eigvals (unscaled)','PEC Eigvals','PEC spectral loss Eigvals','Direct Eigvals FNO (unscaled)','PEC FNO Eigvals','PEC FNO spectral loss Eigvals',fontsize=10)
hold off


figure(10)
clf
hold on
histogram(abs((e_PEC_vals-1)/1e-3),100, Normalization="pdf")
histogram(abs((e_PEC_specloss_vals-1)/1e-3),100, Normalization="pdf")

histogram(abs((e_PEC_vals_FNO-1)/1e-3),100, Normalization="pdf")
histogram(abs((e_PEC_specloss_vals_FNO-1)/1e-3),100, Normalization="pdf")

% MarchenkoPasturLaw(10, 1024, 1024, abs(e_PEC_vals))

legend('PEC Eigvals','PEC spectral loss Eigvals','PEC FNO Eigvals','PEC FNO spectral loss Eigvals',fontsize=10)
hold off

figure(11)
clf
set(0, 'DefaultAxesFontSize', 20)
hold on;

plot(e_Direct_vals,'co','MarkerSize',5,'MarkerFaceColor','c');
plot(e_PEC_vals,'ro','MarkerSize',5,'MarkerFaceColor','r');
plot(e_PEC_specloss_vals,'bo','MarkerSize',5,'MarkerFaceColor','b');

plot(e_Direct_vals_FNO,'go','MarkerSize',5,'MarkerFaceColor','g');
plot(e_PEC_vals_FNO,'ko','MarkerSize',5,'MarkerFaceColor','k');
plot(e_PEC_specloss_vals_FNO,'mo','MarkerSize',5,'MarkerFaceColor','m');

legend('Direct Step','PEC','PEC Spectral loss','Direct Step FNO','PEC FNO','PEC FNO Spectral loss',fontsize=10)

xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')
hold off

figure(12)
clf
set(0, 'DefaultAxesFontSize', 20)
hold on;

plot(e_Direct_vals,'co','MarkerSize',5,'MarkerFaceColor','c');

plot(e_Direct_vals_FNO,'ro','MarkerSize',5,'MarkerFaceColor','r');

legend('Direct Step','Direct FNO Step',fontsize=10)

xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')
hold off

figure(13)
clf
set(0, 'DefaultAxesFontSize', 20)
hold on;

plot(e_PEC_vals,'ro','MarkerSize',3,'MarkerFaceColor','r');
plot(e_PEC_specloss_vals,'bo','MarkerSize',3,'MarkerFaceColor','b');

plot(e_PEC_vals_FNO,'go','MarkerSize',3);
plot(e_PEC_specloss_vals_FNO,'ko','MarkerSize',3);


legend('PEC','PEC Spectral loss','PEC FNO','PEC FNO Spectral loss',fontsize=10)

xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')
hold off

