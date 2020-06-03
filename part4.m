%% Define data sets
clear all;
rng(1);
load('F16traindata_reconstructed.mat','Z_K1_K','Cm');
close all;
X= Z_K1_K'; Y = Cm;
test_fraction = 0.25;
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
Xtrain = X(~idx,:);     Ytrain = Y(~idx,:);
Xtest  = X(idx,:);      Ytest  = Y(idx,:);

%Normalize datasets
Xtestmean = mean(Xtest); Ytestmean = mean(Ytest);
Xtrainmean = mean(Xtrain); Ytrainmean = mean(Ytrain);
Xteststd = std(Xtest); Yteststd = std(Ytest);
Xtrainstd = std(Xtrain); Ytrainstd = std(Ytrain);
Xtrain = normalize(Xtrain);
Ytrain = normalize(Ytrain);
Xtest = normalize(Xtest);
Ytest = normalize(Ytest);

%% Back propagation FF (part 4.1)
%Set network parameters 
n_neurons = 50;
input_bias = 1;
output_bias = 1;
n_epochs = 100;
goal = 0;
min_grad = 1e-10;
eta = 1e-6;
alpha = 5;
eta_max = 1e10;
max_fails = 3;
W_init = 1;

[FFnet_backprop,y_train_bp,y_test_bp] = train_FF_backprop(Xtrain,Ytrain,Xtest,Ytest,n_neurons,...
    input_bias, output_bias,W_init,n_epochs,goal,min_grad,eta,alpha,eta_max,max_fails);

Y_est_train_bp = simFF(Xtrain,FFnet_backprop.w_vector,input_bias,output_bias,n_neurons);
Y_est_test_bp = simFF(Xtest,FFnet_backprop.w_vector,input_bias,output_bias,n_neurons);

%% Levenberg-Marquadt FF (4.2)
%Set network parameters 
mu = 0.001;
alpha = 10;
mu_max = 1e10;
max_fails = 5;

[FFnet_lm,y_train_lm,y_test_lm] = train_FF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,...
    input_bias, output_bias,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);

Y_est_train_lm = simFF(Xtrain,FFnet_lm.w_vector,input_bias,output_bias,n_neurons);
Y_est_test_lm = simFF(Xtest,FFnet_lm.w_vector,input_bias,output_bias,n_neurons);


%% PLOTS
%Revert normalization
Xtest = Xteststd .* Xtest + Xtestmean;
Ytest = Yteststd .* Ytest + Ytestmean;
Xtrain = Xtrainstd .* Xtrain + Xtrainmean;
Ytrain = Ytrainstd .* Ytrain + Ytrainmean;
Y_est_test_bp = Yteststd .* Y_est_test_bp + Ytestmean;
Y_est_train_bp = Ytrainstd .* Y_est_train_bp + Ytrainmean;
Y_est_test_lm = Yteststd .* Y_est_test_lm + Ytestmean;
Y_est_train_lm = Ytrainstd .* Y_est_train_lm + Ytrainmean;
y_train_bp = Ytrainstd .* y_train_bp + Ytrainmean;
y_test_bp = Yteststd .* y_test_bp + Ytestmean;
y_train_lm = Ytrainstd .* y_train_lm + Ytrainmean;
y_test_lm = Yteststd .* y_test_lm + Ytestmean;

error_bp_train =[];
error_bp_test = [];
for k=1:size(y_test_bp,2)
    error_bp_train = [error_bp_train;immse(Ytrain,y_train_bp(:,k))];
    error_bp_test = [error_bp_test;immse(Ytest,y_test_bp(:,k))];   
end

error_lm_train =[];
error_lm_test = [];
for k=1:size(y_test_lm,2)
    error_lm_train = [error_lm_train;immse(Ytrain,y_train_lm(:,k))];
    error_lm_test = [error_lm_test;immse(Ytest,y_test_lm(:,k))];   
end

plot_title = 'C_m estimation using FF network (back propagation)';
IOmap(Xtest,Ytest,Y_est_test_bp,plot_title,1);
plot_title = 'Error map using FF network (back propagation)';
IOmap_error(Xtest,Ytest,Y_est_test_bp,plot_title,2);

figure(3);
axis = 1:n_epochs;
semilogy(axis,error_bp_train,axis,error_bp_test);
xlabel('Epochs [-]'); ylabel('MSE [-]');
title('MSE performance of FF network (Back propagation)');
legend('MSE for train data', 'MSE for test data');

plot_title = 'C_m estimation using FF network (LM algorithm)';
IOmap(Xtest,Ytest,Y_est_test_lm,plot_title,4);

plot_title = 'Error map using FF network (LM algorithm)';
IOmap_error(Xtest,Ytest,Y_est_test_lm,plot_title,5);

figure(6);
axis = 1:n_epochs;
semilogy(axis,error_lm_train,axis,error_lm_test);
xlabel('Epochs [-]'); ylabel('MSE [-]');
title('MSE performance of FF network (LM algorithm)');
legend('MSE for train data', 'MSE for test data');


