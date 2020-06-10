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
n_neurons = 30;
input_bias = -1;
output_bias = -1;
n_epochs = 50;
goal = 0;
min_grad = 1e-10;
eta = [1 0.0001 1e-6]; mu = [1 0.0001 1e-6];
alpha = [1 10];
eta_max = 1e10; mu_max = 1e10;
max_fails = 3;
W_init = 1;

for i = 1:length(alpha)
    for j = 1:length(eta)
            [FFnet_backprop,y_train_bp,y_test_bp] = train_FF_backprop(Xtrain,Ytrain,Xtest,Ytest,n_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,eta(j),alpha(i),eta_max,max_fails);
     
            error_bp = [];
            Ytest2 = Yteststd .* Ytest + Ytestmean;
            y_test_bp = Yteststd .* y_test_bp + Ytestmean;
            for z=1:size(y_test_bp,2)
                error_bp = [error_bp;immse(Ytest2,y_test_bp(:,z))];
            end
            figure(1)
            axis = 1:size(error_bp,1);
            semilogy(axis,error_bp);
            hold on;       
    end
    
end
 Legend=cell(6,1);
 Legend{1}='Fixed 1' ;
 Legend{2}='Fixed 0.001' ;
 Legend{3}='Fixed 1e.6' ;
 Legend{4}='Adaptive 1' ;
 Legend{5}='Adaptive 0.001' ;
 Legend{6}='Adaptive 1e-6' ;
 legend(Legend);   
 
 for i = 1:length(alpha)
    for j = 1:length(mu)
            [FFnet_lm,y_train_lm,y_test_lm,~] = train_FF_lm(Xtrain,Ytrain,Xtest,Ytest,n_neurons,...
            input_bias,output_bias,W_init,n_epochs,goal,min_grad,mu(j),alpha(i),mu_max,max_fails);
            
            error_lm = [];
            Ytest2 = Yteststd .* Ytest + Ytestmean;
            y_test_lm = Yteststd .* y_test_lm + Ytestmean;
            for z=1:size(y_test_lm,2)
                error_lm = [error_lm;immse(Ytest2,y_test_lm(:,z))];
            end
            figure(2)
            axis = 1:size(error_lm,1);
            semilogy(axis,error_lm);
            hold on;    
    end
    
end
 Legend=cell(6,1);
 Legend{1}='Fixed 1' ;
 Legend{2}='Fixed 0.001' ;
 Legend{3}='Fixed 1e-6' ;
 Legend{4}='Adaptive 1' ;
 Legend{5}='Adaptive 0.001' ;
 Legend{6}='Adaptive 1e.6' ;
 legend(Legend);