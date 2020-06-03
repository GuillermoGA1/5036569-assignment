%% Optimizing number of neurons
% Define data sets
clear all;
rng(1);
load('F16traindata_reconstructed.mat','Z_K1_K','Cm');
close all;
X= Z_K1_K'; Y = Cm;
test_fraction = 0.3;
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);

%Network parameters
n_neurons = 1;
RBFcenters = 1; W_init = 1;
n_epochs = 35;
goal = 0;
min_grad = 1e-10;
mu_max = 1e10;
max_fails = 4;
alpha = 10;
mu = 0.001;

%Arrays to store
test_mse = [];
train_mse = [];
N_NEURONS = [];
i = 0;
stop = 0;
fails = 0;
test_fails = 4;

%Iterate for several damping values
while ~stop && n_neurons <= 120
    i= i+1;
[RBF_net_lm,error] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
e1 = nonzeros(error(:,1));
e2 = nonzeros(error(:,2));
N_NEURONS = horzcat(N_NEURONS, n_neurons);
train_mse = horzcat(train_mse,e1(end));
test_mse = horzcat(test_mse,e2(end));
n_neurons = n_neurons + 1;

if i > 1 && test_mse(1,i) > test_mse(1,i-1)
    fails = fails +1 ;
end

if fails >= 1 && test_mse(1,i) < test_mse(1,i-1)
    fails = 0;
end

if fails >= test_fails
    stop = 1;
end
end
 
%Do previous net
n_neurons = 20;
[RBFnet1,error1] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
Y_est_test_1 = simRBF(Xtest,RBFnet1.IW',RBFnet1.LW',RBFnet1.centers);
plot_title = 'C_m estimation using RBF (random # neurons)';
IOmap(Xtest,Ytest,Y_est_test_1,plot_title,1);
plot_title = 'Error map using RBF (random # neurons)';
IOmap_error(Xtest,Ytest,Y_est_test_1,plot_title,2);

%Choose optimal number of neurons
[~,index] =min(test_mse);
n_neurons = N_NEURONS(index);
[RBF_optimal,error_optimal] = train_RBF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
Y_est_test_optimal = simRBF(Xtest,RBF_optimal.IW',RBF_optimal.LW',RBF_optimal.centers);
plot_title = 'C_m estimation using RBF (optimal # neurons)';
IOmap(Xtest,Ytest,Y_est_test_optimal,plot_title,3);
plot_title = 'Error map using RBF (optimal # neurons)';
IOmap_error(Xtest,Ytest,Y_est_test_optimal,plot_title,4);

figure(5);
axis = 1:n_epochs;
semilogy(axis,error1(:,2),axis,error_optimal(:,2));
xlabel('Epochs [-]'); ylabel('MSE [-]');
title('MSE performance for different number of neurons)');
legend('Random number of neurons', 'Optimal number of neurons');

figure(6);
semilogy(N_NEURONS,test_mse,N_NEURONS,train_mse);
xlabel('Number of neurons [-]');
ylabel('MSE [-]');
title('Effect of number of neurons in net performance');
legend('Test MSE','Train MSE');



