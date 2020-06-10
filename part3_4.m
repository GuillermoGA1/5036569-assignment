%% Influence of learning rate in LM algorithm (part 3.4)
% Define data sets
clear all;
rng(105);
load('F16traindata_reconstructed.mat','Z_K1_K','Cm');
close all;
X= Z_K1_K'; Y = Cm;
test_fraction = 0.3;
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);

%Network parameters
n_neurons = 30;
RBFcenters = 3; W_init = 1;
n_epochs = 50;
goal = 0;
min_grad = 1e-10;
mu_max = 1e10;
max_fails = 3;

var_rate = 10;
fix_rate = 1;
mu = [1 0.0001 1e-6];
convergence_fix = zeros(n_epochs,length(mu));
convergence_var = zeros(n_epochs,length(mu));

%Iterate for several damping values
for i=1:length(mu)
[~,error_fix] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu(i),fix_rate,mu_max,max_fails);
[~,error_var] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu(i),var_rate,mu_max,max_fails);
convergence_fix(:,i)=error_fix(:,1);
convergence_var(:,i)= error_var(:,1);
end
 
figure(4);
axis = 1:n_epochs;
semilogy(axis,convergence_fix,axis,convergence_var);
xlabel('Epochs [-]');
ylabel('MSE [-]');
title('Effect of learning rate in MSE convergence');
legend('Fix 1','Fix 0.0001', 'Fix 1e-6','Adaptive 1','Adaptive 0.0001','Adaptive 1e-6');