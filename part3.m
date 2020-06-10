%% Define data sets
clear all;
rng(105); %Seed 105
load('F16traindata_reconstructed.mat','Z_K1_K','Cm');
close all;
X= Z_K1_K'; Y = Cm;
test_fraction = 0.3;
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);
%% Linear regression RBF (part 3.1)
%Set network parameters (no training parameters required)
n_neurons = 30;
RBFcenters = 3; %Leave 1 to use kmeans centers, 2 for uniformly distributed centers.
W_init = 2;
RBFnet_lin = train_RBF_lineregress(Xtrain,Ytrain,n_neurons,RBFcenters,W_init);
Y_est_train = simRBF(Xtrain,RBFnet_lin.IW',RBFnet_lin.LW',RBFnet_lin.centers);
Y_est_test = simRBF(Xtest,RBFnet_lin.IW',RBFnet_lin.LW',RBFnet_lin.centers);

%Mean square error
mse_train = immse(Ytrain,Y_est_train);
mse_test = immse(Ytest,Y_est_test);


plot_title = 'C_m estimation using RBF network (linear regression)';
IOmap(Xtest,Ytest,Y_est_test,plot_title,1);
plot_title = 'Error map using RBF network (linear regression)';
IOmap_error(Xtest,Ytest,Y_est_test,plot_title,2);

%% Levenberg-Marquardt RBF (part 3.2-3.3)
Network parameters
n_neurons = 30;
RBFcenters = 1; W_init = 1;
n_epochs = 50;
goal = 0;
min_grad = 1e-10;
mu = 0.0001;
alpha = 10;
mu_max = 1e10;
max_fails = 3;
%Network training
[RBFnet_lm,error] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
Y_est_train = simRBF(Xtrain,RBFnet_lm.IW',RBFnet_lm.LW',RBFnet_lm.centers);
Y_est_test = simRBF(Xtest,RBFnet_lm.IW',RBFnet_lm.LW',RBFnet_lm.centers);
 
figure(4);
axis = 1:n_epochs;
semilogy(axis,error(:,1),axis,error(:,2));
xlabel('Epochs [-]'); ylabel('MSE [-]');
title('MSE performance of RBF network (LM algorithm)');
legend('MSE for train data', 'MSE for test data');
 
plot_title = 'C_m estimation using RBF network (LM algorithm)';
IOmap(Xtest,Ytest,Y_est_test,plot_title,5);
plot_title = 'Error map using RBF network (LM algorithm)';
IOmap_error(Xtest,Ytest,Y_est_test,plot_title,6);
