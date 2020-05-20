function [theta_opt,opt_order,mse_average] = optimize_OLS(max_order,input,output)
%rng('default');
X = input; Y = output;
n_folds = 5;
test_fraction = 0.3;
mse = zeros(max_order,n_folds,3);
%% Create folds 

%Split total data into train and test data
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
X_train = X(~idx,:);     Y_train = Y(~idx,:);
X_test  = X(idx,:);      Y_test  = Y(idx,:);

fold_size = floor(size(X_train,1) / n_folds);
random = randperm(size(X_train,1),size(X_train,1));
fold_idx = zeros(fold_size, n_folds);

for i =0:(n_folds-1)
    fold_idx(:,i+1) = random(i*fold_size+1:fold_size*(i+1));
end

%Since fold indexes are for whole X and Y training data set, split them
%into indexes for train data and indexes for validation data.
cv2 = cvpartition(size(fold_idx,1),'HoldOut',test_fraction);
idx2 = cv2.test;
fold_idx_train = fold_idx(~idx2,:);     
fold_idx_validate  = fold_idx(idx2,:);      

%% Calculate OLS for every order using folds

for i =1:max_order
    for j = 1:n_folds
        model = regression_matrix(i,3);
        A_train= x2fx(X_train(fold_idx_train(:,j),:), model);
        A_validate =x2fx(X_train(fold_idx_validate(:,j),:), model);
        A_test = x2fx(X_test,model);
        theta = pinv(A_train)*Y_train(fold_idx_train(:,j));
        Y_est_train = A_train*theta;
        Y_est_validate = A_validate*theta;
        Y_est_test = A_test*theta;
        
        mse(i,j,1)= immse(Y_est_train,Y_train(fold_idx_train(:,j)));
        mse(i,j,2)= immse(Y_est_validate,Y_train(fold_idx_validate(:,j)));
        mse(i,j,3) = immse(Y_est_test,Y_test);
    end
end

%Get folds average of mse for each order
o=1:max_order;
mse_average = mean(mse,2);
[~,opt_order] = min(mse_average(:,:,3));
model_opt = regression_matrix(opt_order,3);
A_opt = x2fx(X_train, model_opt);
theta_opt =  pinv(A_opt)*Y_train;
end


        
       
        

        
       