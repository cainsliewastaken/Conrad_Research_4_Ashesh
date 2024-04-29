model_one = load('FNO_KS_PEC4step_lead1_tendency_jacs_all_chkpts.mat');

model_RMSE = load('predicted_PEC4step_1024_DON_lead1_RMSE.mat');

model_name = 'PEC4 Step DON ';

% MLP eigenvalue calculations
model_mat = squeeze(model_one.Jacobian_mats_epoch_60(1,:,:));
% direct_MLP = (direct_MLP*direct_MLP')/1024;
[model_evec, model_e_val] = eig(model_mat);
[model_e_val, model_ind] = sort(diag(model_e_val));



figure(2)
clf
subplot(2,2,1)
theta = linspace(-pi,pi,10000);
x=cos(theta)+1*1i*sin(theta);
set(0, 'DefaultAxesFontSize', 20)
hold on;

plot(model_e_val,'co','MarkerSize',5,'MarkerFaceColor','c','DisplayName',model_name);
axis manual
plot(x,'r','Linewidth',2,'DisplayName','Unit Circle');

legend(fontsize=10)
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')

subplot(2,2,2)
t_final = 100;
t_range = linspace(0, t_final, 10000);


xx = linspace(0,t_final,length(model_RMSE.RMSE));
plot(xx, model_RMSE.RMSE,'-black','DisplayName',model_name);
legend(Location='northwest', FontSize=8)
xlabel('t')
ylabel('RMSE')
axis([-.5 100 -.5 20])
hold off

subplot(2,2,3)
histogram(abs(model_e_val), 100, Normalization="pdf")
histogram(abs((model_e_val-1)/1e-3), 100, Normalization="pdf")
xlabel('|\lambda|', 'Interpreter','tex')
legend(model_name, fontsize=10)




