%Define data sets
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

%Normalize datasets
Xtestmean = mean(Xtest); Ytestmean = mean(Ytest);
Xtrainmean = mean(Xtrain); Ytrainmean = mean(Ytrain);
Xteststd = std(Xtest); Yteststd = std(Ytest);
Xtrainstd = std(Xtrain); Ytrainstd = std(Ytrain);
Xtrain = normalize(Xtrain);
Ytrain = normalize(Ytrain);
Xtest = normalize(Xtest);
Ytest = normalize(Ytest);

%Set network parameters (back propagation
rand_neurons = 30;
training_algo = 'lm'; %change to 'bp' for back propapgation
n_neurons = 0;
input_bias = -1;
output_bias = -1;
n_epochs = 50;
goal = 0;
min_grad = 1e-10;
eta = 0.0001; mu = 0.0001;
alpha = 10;
eta_max = 1e10; mu_max = 1e10;
max_fails = 3;
W_init = 1;
z= 0;
MSE = [];
N_NEURONS = [];

while n_neurons < 100
    n_neurons = n_neurons+5;
    if training_algo == 'lm'
            [FFnet,y_train,y_test,~] = train_FF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
    elseif training_algo == 'bp'
            [FFnet,y_train,y_test] = train_FF_backprop(Xtrain,Ytrain,Xtest,Ytest,n_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,eta,alpha,eta_max,max_fails);
    else
        disp('Invalid selection of training algorithm');
        break
    end
        
     error = [];
     Ytest2 = Yteststd .* Ytest + Ytestmean;
     y_test = Yteststd .* y_test + Ytestmean;
     for z=1:size(y_test,2)
         error = [error;immse(Ytest2,y_test(:,z))];
     end
     mse = error(end,1);
     MSE = [MSE mse];
     N_NEURONS = [N_NEURONS n_neurons];
                   
end
%Plot effect of number of neurons
N_values = length(MSE);
step = n_neurons/N_values;
xaxis =0:step:n_neurons;
xaxis = xaxis(2:end);
figure(1)
plot(xaxis,MSE);
xlabel('Number of neurons [-]');
ylabel('MSE [-]');
title('Effect of number of neurons in net performance');

%Choose optimal number of neurons
[~,index] = min(MSE);
opt_neurons = N_NEURONS(index);
if training_algo == 'lm'
    [FFnet_optimal,~,y_test_opt,~] = train_FF_lm(Xtrain,Ytrain,Xtest,Ytest,opt_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
    [FFnet_rand,~,y_test_rand,~] = train_FF_lm(Xtrain,Ytrain,Xtest,Ytest,rand_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails);
else
    [FFnet_optimal,~,y_test_opt] = train_FF_backprop(Xtrain,Ytrain,Xtest,Ytest,opt_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,eta,alpha,eta_max,max_fails);
    [FFnet_rand,~,y_test_rand] = train_FF_backprop(Xtrain,Ytrain,Xtest,Ytest,rand_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,eta,alpha,eta_max,max_fails);
end

error_opt = []; error_rand = [];
y_test_opt = Yteststd .* y_test_opt + Ytestmean;
y_test_rand = Yteststd .* y_test_rand + Ytestmean;
for z=1:size(y_test_opt,2)
    error_opt = [error_opt;immse(Ytest2,y_test_opt(:,z))];  
end
for z=1:size(y_test_rand,2)
    error_rand = [error_rand;immse(Ytest2,y_test_rand(:,z))];
end
Xtest = Xteststd .* Xtest + Xtestmean;


plot_title = 'C_m estimation using FF network (optimal # neurons)';
IOmap(Xtest,Ytest2,y_test_opt(:,end),plot_title,2);
plot_title = 'C_m estimation using FF network (random # neurons)';
IOmap(Xtest,Ytest2,y_test_rand(:,end),plot_title,3);

figure(4)
axis = 1:size(error_opt,1);
semilogy(axis,error_opt);
hold on;
axis = 1:size(error_rand,1);
semilogy(axis,error_rand);
xlabel('Epochs [-]'); ylabel('MSE [-]');
title('MSE performance for different number of neurons)');
legend('Optimal number of neurons','Random number of neurons');