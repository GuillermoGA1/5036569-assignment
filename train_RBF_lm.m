function [net,error] = train_RBF_lm(XtrainData,YtrainData,XtestData,YtestData,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails)

n_inputs = size(XtrainData,2);
n_outputs = size(YtrainData,2);
N = size(XtrainData,1);
%Build net fixed fields
net.trainAlg = 'lm';
net.trainFunct = {'radbas','purelin'};
net.range = [-ones(n_inputs, 1) ones(n_inputs, 1)];
net.name = 'rbf';
%net.trainParam.goal = goal;
net.trainParam.min_grad = min_grad;
net.trainParam.mu = mu;
net.trainParam.mu_dec = 1/alpha;
net.trainParam.mu_inc = alpha;
net.trainParam.mu_max = mu_max;

%Initalize centers of activation functions
if RBFcenters == 1
    [~,net.centers] = kmeans(XtrainData,n_neurons);
elseif RBFcenters == 2
    r1 = max(XtrainData(:,1)) - min(XtrainData(:,1));
    r2 = max(XtrainData(:,2)) - min(XtrainData(:,2));
    r3 = max(XtrainData(:,2)) - min(XtrainData(:,3));
    c1 = r1 .* rand(n_neurons,1) + min(XtrainData(:,1));
    c2 = r2 .* rand(n_neurons,1) + min(XtrainData(:,2));
    c3 = r3 .* rand(n_neurons,1) + min(XtrainData(:,3));
    net.centers = [c1 c2 c3];
elseif RBFcenters == 3
    step1 = (max(XtrainData(:,1)) - min(XtrainData(:,1)))/n_neurons;
    step2 = (max(XtrainData(:,2)) - min(XtrainData(:,2)))/n_neurons;
    step3 = (max(XtrainData(:,3)) - min(XtrainData(:,3)))/n_neurons;
    c1 = min(XtrainData(:,1)):step1:max(XtrainData(:,1));
    c2 = min(XtrainData(:,2)):step2:max(XtrainData(:,2));
    c3 = min(XtrainData(:,3)):step3: max(XtrainData(:,3));
    net.centers = [c1(1:end-1)' c2(1:end-1)' c3(1:end-1)'];
    
end

%Initialization parameters
if W_init == 1
IW = randn(n_neurons,n_inputs)';
LW = randn(n_neurons, n_outputs);
elseif W_init == 2
IW = ones(n_neurons,n_inputs)';
LW = ones(n_neurons, n_outputs);
end
    
w = reshape(IW',[n_neurons*n_inputs 1]);
w = vertcat(w,LW);
epochs = 0;
validation_fails = 0;
stop = 0;

 %Arrays to store values
diff = zeros(N,n_neurons,3);
de_dwij = zeros(N,n_neurons,3);
dvj_dwij = zeros(N,n_neurons,3);
error = zeros(n_epochs,2);

while  ~stop
epochs = epochs+1;
IW = [w(1:n_neurons) w(n_neurons+1:2*n_neurons) w(2*n_neurons+1:3*n_neurons)]';
LW = w(3*n_neurons+1:end);
diff(:,:,1) = (XtrainData(:,1) - net.centers(:,1)').^2;
diff(:,:,2) = (XtrainData(:,2) - net.centers(:,2)').^2;
diff(:,:,3) = (XtrainData(:,3) - net.centers(:,3)').^2;
v_j = (IW(1,:).^2) .* diff(:,:,1) + (IW(2,:).^2) .* diff(:,:,2) + (IW(3,:).^2) .* diff(:,:,3);
y_j = exp(-v_j);
v_k = y_j * LW;
y_k = v_k;

e = YtrainData - y_k;
E = immse(YtrainData,y_k);

%Derivatives wrt to output weights
de_dwjk = -1 * y_j;

%Derivatives wrt to input weights
dvk_dyj = LW';
dyj_dvj = -y_j;
dvj_dwij(:,:,1) = 2 * IW(1,:) .* diff(:,:,1);
dvj_dwij(:,:,2) = 2 * IW(2,:) .* diff(:,:,2);
dvj_dwij(:,:,3) = 2 * IW(3,:) .* diff(:,:,3);
de_dwij(:,:,1) = -1 .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,1);
de_dwij(:,:,2) = -1 .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,2);
de_dwij(:,:,3) = -1 .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,3);

%Build jacobian
J = [de_dwij(:,:,1) de_dwij(:,:,2) de_dwij(:,:,3)  de_dwjk];
w_t1 = w - pinv(J' * J + mu*eye(4*n_neurons)) * (J'*e);

%Recalculate:
IW = [w_t1(1:n_neurons) w_t1(n_neurons+1:2*n_neurons) w_t1(2*n_neurons+1:3*n_neurons)]';
LW = w_t1(3*n_neurons+1:end);
y_k = simRBF(XtrainData,IW,LW,net.centers);
e = YtrainData - y_k;
E_t1 = immse(YtrainData,y_k);

if E_t1 <= E
    w = w_t1;
    mu = mu * 1/alpha;
else
    z = 0;
    while E_t1 > E && z<5
        mu = mu * alpha;
        w_t1 = w - pinv(J' * J + mu*eye(4*n_neurons)) * (J'*e);
        IW = [w_t1(1:n_neurons) w_t1(n_neurons+1:2*n_neurons) w_t1(2*n_neurons+1:3*n_neurons)]';
        LW = w_t1(3*n_neurons+1:end);
        y_k = simRBF(XtrainData,IW,LW,net.centers);
        e = YtrainData - y_k;
        E_t1 = immse(YtrainData,y_k);
        z= z+1;
    end
    w = w_t1;
end

%Evaluate network after every iteration to store error performance
y_est_train = simRBF(XtrainData,IW,LW,net.centers);
y_est_test = simRBF(XtestData,IW,LW,net.centers);
error(epochs,1) = immse(y_est_train,YtrainData);
error(epochs,2) = immse(y_est_test,YtestData);

%Check validation fails 
if epochs > 1 && error(epochs,2) > error(epochs-1,2)
    validation_fails = validation_fails +1;
end
if epochs > 1 && error(epochs,2) < error(epochs-1,2)
    validation_fails = 0;
end

%Stop criteria
if validation_fails >= max_fails
    stop = 1;
    for k=(epochs-validation_fails):epochs
        error(epochs,:)=zeros(1,2);
    end
    epochs = epochs -validation_fails;
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
elseif mu >= mu_max
    stop = 1;
    criteria = 'Too high learning rate';
else
    stop = 0;
end   
end

net.IW =IW';
net.LW = LW';
net.trainParam.goal = error(epochs,1);
net.trainParam.epochs= epochs;
disp(criteria);

end