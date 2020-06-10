function [net,y_train,y_test] = train_FF_backprop(XtrainData,YtrainData,XtestData,YtestData,n_neurons,...
    input_bias, output_bias,W_init,n_epochs,goal,min_grad,eta,alpha,eta_max,max_fails)
rng(1019);
n_inputs = size(XtrainData,2);
n_outputs = size(YtrainData,2);
N = size(XtrainData,1);
M = size(XtestData,1);
%Build net fixed fields
net.trainAlg = 'backpropagation';
net.trainFunct = {'tansig','purelin'};
net.range = [-ones(n_inputs, 1) ones(n_inputs, 1)];
net.name = 'ff';
%net.trainParam.goal = goal;
net.trainParam.min_grad = min_grad;
net.trainParam.eta = eta;
net.trainParam.eta_dec = 1/alpha;
net.trainParam.eta_inc = alpha;
net.trainParam.eta_max = eta_max;


%Initialization parameters
if W_init == 1
IW = randn(n_neurons-1,n_inputs+1)';
LW = randn(n_neurons, n_outputs);
elseif W_init == 2
IW = ones(n_neurons-1,n_inputs+1)';
LW = ones(n_neurons, n_outputs);
end
w = reshape(IW',[(n_neurons-1)*(n_inputs+1) 1]);
w = vertcat(w,LW);
epochs = 0;
validation_fails = 0;
stop = 0;

%Initialize arrays
dvj_dwij = zeros(N,n_neurons-1,4);
dE_dwij = zeros(N,n_neurons-1,4);
y_train = zeros(N,n_epochs);
y_test = zeros(M,n_epochs);
error = zeros(n_epochs,2);

while  ~stop
epochs = epochs+1;

[y_k,y_j,v_j] = simFF(XtrainData,w,input_bias,output_bias,n_neurons);
e = YtrainData - y_k;
E = immse(YtrainData,y_k);

%Derivatives wrt to output weights
dE_dyk =(2/N)*e*-1;
dyk_dvk = 1;
dvk_dwjk = y_j;
dE_dwjk = dE_dyk .* dyk_dvk .* dvk_dwjk;
dE_dwjk_sum = sum(dE_dwjk,1);

%Derivatives wrt to input weights
dvk_dyj = LW(2:end)';
dyj_dvj = 1./(cosh(v_j)).^2;
dvj_dwij(:,:,1) = input_bias*ones(N,n_neurons-1);
dvj_dwij(:,:,2) = XtrainData(:,1) .* ones(N,n_neurons-1);
dvj_dwij(:,:,3) = XtrainData(:,2) .* ones(N,n_neurons-1);
dvj_dwij(:,:,4) = XtrainData(:,3) .* ones(N,n_neurons-1);
dE_dwij(:,:,1) = dE_dyk .* dyk_dvk .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,1);
dE_dwij(:,:,2) = dE_dyk .* dyk_dvk .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,2);
dE_dwij(:,:,3) = dE_dyk .* dyk_dvk .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,3);
dE_dwij(:,:,4) = dE_dyk .* dyk_dvk .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,4);
dE_dwij_sum = sum(dE_dwij,1);

grad =-[dE_dwij_sum(:,:,1) dE_dwij_sum(:,:,2) dE_dwij_sum(:,:,3) dE_dwij_sum(:,:,4)  dE_dwjk_sum]';
w_t1 = w + eta*grad;

%Recalculate with updated weights
[y_k,~,~] = simFF(XtrainData,w_t1,input_bias,output_bias,n_neurons);
e = YtrainData - y_k;
E_t1 = immse(YtrainData,y_k);

%Learning rate update
if E_t1 <= E
    w = w_t1;
    eta = eta * alpha;
else
    z=0;
    while E_t1 >E && z<7
        eta = eta /alpha;
        w_t1 = w + eta*grad;
        [y_k,~,~] = simFF(XtrainData,w_t1,input_bias,output_bias,n_neurons);
        e = YtrainData - y_k;
        E_t1 = immse(YtrainData,y_k);  
        z=z+1;
    end
    w = w_t1;
end

%Evaluate network after every iteration to store error performance
y_est_train = simFF(XtrainData,w,input_bias,output_bias,n_neurons);
y_est_test = simFF(XtestData,w,input_bias,output_bias,n_neurons);
y_train(:,epochs) = y_est_train;
y_test(:,epochs) = y_est_test;
error(epochs,1) = immse(y_est_train,YtrainData);
error(epochs,2) = immse(y_est_test,YtestData);

%Check validation fails 
if epochs > 1 && error(epochs,2) >= error(epochs-1,2)
    validation_fails = validation_fails +1;
end
if epochs > 1 && error(epochs,2) < error(epochs-1,2)
    validation_fails = 0;
end

%Stop criteria
if validation_fails >= max_fails
    stop = 1;
    criteria = 'Maximum validation fails reached';
elseif abs(gradient(error(:,1)))< min_grad
    stop = 1;
    criteria = 'Too low gradient';
elseif epochs >= n_epochs
    stop = 1;
    criteria = 'Maximum number of epochs reached';
elseif error(epochs,1)<= goal
    stop = 1;
    criteria = 'Goal reached';
elseif eta >= eta_max
    stop = 1;
    criteria = 'Too high learning rate';
else
    stop = 0;
end

end
IW = [w(1:n_neurons-1) w(n_neurons:2*(n_neurons-1)) w(2*n_neurons-1:3*(n_neurons-1)) w(3*n_neurons-2:4*(n_neurons-1))]';
LW =  w(4*n_neurons-3:end);
error = nonzeros(error);
error = reshape(error,[],2);
n_epochs = size(error,1);
y_test = y_test(:,1:n_epochs);
y_train = y_train(:,1:n_epochs);
net.b{1,1} = IW(1,:);
net.b{2,1} = LW(1,1);
net.IW =IW(2:end,:)';
net.LW = LW(2:end)';
net.w_vector = w;
net.trainParam.goal = error(epochs,1);
net.trainParam.epochs= epochs;
disp(criteria);

end
