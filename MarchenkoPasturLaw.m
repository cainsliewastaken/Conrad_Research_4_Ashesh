function MarchenkoPasturLaw(fig, N, T, eigs)
% Marchenko Pastur Distribution

% In Random Matrix Theory, MP law gives the probability density function
% of singular values of large rectangular random matrices;
% when the dimensions of matrix tend to infinity.

% This contribution illustrates the PDF of matrix Y(N,N)=(T^-1)X*X^T, 
% where X is random matrix whose entries X_i,j are independent 
% and identically distributed random variables with zero mean
% and variance s^2. The program is applicable for both uniform and random
% distributions.

% Ref :
% Marchenko,V. A., Pastur, L. A. (1967) "Distribution of eigenvalues for some sets of
% random matrices", Mat. Sb. (N.S.), 72(114):4, 507–536

% (c) Youssef KHMOU, Applied Mathematics ,30 January,2015.

% Ratio of matrix dimensions
Q=N/T;

% number of points for measurement.
n=2500;

% sigma = 1;
% sigma = sqrt(var(eigs));
% sprintf("%d",sigma)
% 
lmax = max(eigs);
sigma = sqrt(lmax/(1 + (1/sqrt(Q)))^2);
% sprintf("%d",sigma)

a=(sigma*(1-sqrt(Q))).^2;
b=(sigma*(1+sqrt(Q))).^2;


lambda=linspace(a,b,n);
% Normalization
% Theoretical pdf
ft=@(lambda,a,b,c) sqrt((b-lambda).*(lambda-a))./(2*pi*lambda*Q*sigma.^2);
F=ft(lambda,a,b,Q);
% Processing numerical pdf
F(isnan(F))=0;
% F=F/trapz(F);
% F=F/sum(F);




figure(fig)
hold on;
plot(lambda,F,'g','LineWidth',2);
hold off;
end
