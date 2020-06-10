%% First initalization
clear all; close all;
rng(1091);
load('F16traindata_reconstructed.mat','Z_K1_K','Cm');
X= Z_K1_K'; Y = Cm;
test_fraction = 0.3;
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);
n_epochs = 50;
goal = 0;
min_grad = 1e-10;
mu = 0.0001;
alpha = 10;
mu_max = 1e10;
max_fails = 3;
n_neurons = 30;
RBFcenters = 1; W_init =1;
RBFnet_lin = train_RBF_lineregress(Xtrain,Ytrain,n_neurons,RBFcenters,W_init);
Y_est_test_lin1 = simRBF(Xtest,RBFnet_lin.IW',RBFnet_lin.LW',RBFnet_lin.centers);
[RBFnet_lm,error1] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
Y_est_test_lm1 = simRBF(Xtest,RBFnet_lm.IW',RBFnet_lm.LW',RBFnet_lm.centers);
plot_title = 'Error map using RBF network (linear regression)';
IOmap_error(Xtest,Ytest,Y_est_test_lin1,plot_title,1);
plot_title = 'Error map using RBF network (LM algorithm)';
IOmap_error(Xtest,Ytest,Y_est_test_lm1,plot_title,2);

%% Second initialization
rng(374);
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);
RBFnet_lin = train_RBF_lineregress(Xtrain,Ytrain,n_neurons,RBFcenters,W_init);
Y_est_test_lin2 = simRBF(Xtest,RBFnet_lin.IW',RBFnet_lin.LW',RBFnet_lin.centers);
[RBFnet_lm,error2] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
Y_est_test_lm2 = simRBF(Xtest,RBFnet_lm.IW',RBFnet_lm.LW',RBFnet_lm.centers);
plot_title = 'Error map using RBF network (linear regression)';
IOmap_error(Xtest,Ytest,Y_est_test_lin2,plot_title,3);
plot_title = 'Error map using RBF network (LM algorithm)';
IOmap_error(Xtest,Ytest,Y_est_test_lm2,plot_title,4);
%% Third initialization
rng(4959);
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);
RBFnet_lin = train_RBF_lineregress(Xtrain,Ytrain,n_neurons,RBFcenters,W_init);
Y_est_test_lin3 = simRBF(Xtest,RBFnet_lin.IW',RBFnet_lin.LW',RBFnet_lin.centers);
[RBFnet_lm,error3] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
Y_est_test_lm3 = simRBF(Xtest,RBFnet_lm.IW',RBFnet_lm.LW',RBFnet_lm.centers);
plot_title = 'Error map using RBF network (linear regression)';
IOmap_error(Xtest,Ytest,Y_est_test_lin3,plot_title,5);
plot_title = 'Error map using RBF network (LM algorithm)';
IOmap_error(Xtest,Ytest,Y_est_test_lm3,plot_title,6);

%% LM sensitivity to centers and weights
rng(105);
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);
RBFcenters = [1 2 3];
W = [1 2];
MSE = zeros(n_epochs,length(W)*length(RBFcenters));
k=1;
for i =1:length(W)
    for j =1:length(RBFcenters)
        [RBFnet_lm,error3] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters(j),W(i),n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
        MSE(:,k) = error3(:,2);
        k=k+1;
    end
end
figure(7);
axis = 1:n_epochs;
semilogy(axis,MSE);
xlabel('Epochs [-]');
ylabel('MSE [-]');
title('MSE convergence for different initialization parameters using LM');
legend('Random weights,kmeans centers','Random weights,random centers','Random weights,uniform centers',...
        'Same weights,kmeans centers','Same weights,random centers','Same weights,uniform centers');