function [net,error] = train_RBF_lm(XtrainData,YtrainData,XtestData,YtestData,n_neurons,RBFcenters,W_init,n_epochs,goal,min_grad,mu,alpha,mu_max,max_fails)
rng(1019);
X = [XtrainData;XtestData]; 
n_inputs = size(XtrainData,2);
n_outputs = size(YtrainData,2);
N = size(XtrainData,1);
%Build net fixed fields
net.trainAlg = 'lm';
net.trainFunct = {'radbas','purelin'};
net.range = [min(X(:,1)) max(X(:,1));min(X(:,2)) max(X(:,2));min(X(:,3)) max(X(:,3))];
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
    net.centers = [];
    for j=1:n_inputs
        r = max(XtrainData(:,j)) - min(XtrainData(:,j));
        c = r .* rand(n_neurons,1) + min(XtrainData(:,j));
    net.centers = [net.centers c];
    end
elseif RBFcenters == 3
    net.centers =[];
    for j= 1:n_inputs
        step = (max(XtrainData(:,j)) - min(XtrainData(:,j)))/n_neurons;
        c = min(XtrainData(:,j)):step:max(XtrainData(:,j));
        net.centers = [net.centers c(1:end-1)'];   
    end
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
diff = zeros(N,n_neurons,n_inputs);
de_dwij = zeros(N,n_neurons,n_inputs);
dvj_dwij = zeros(N,n_neurons,n_inputs);
error = zeros(n_epochs,2);

while  ~stop
epochs = epochs+1;
IW = [];
for j=0:n_inputs-1
    IW = [IW w(j*n_neurons+1:(j+1)*n_neurons)];
end
IW =IW';
LW = w(n_inputs*n_neurons+1:end);
v_j= zeros(N,n_neurons);
for j=1:n_inputs
    diff(:,:,j) = (XtrainData(:,j)-net.centers(:,j)').^2;
    v_j = v_j + (IW(j,:).^2) .* diff(:,:,j);
end
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
J = [];
for j=1:n_inputs
    dvj_dwij(:,:,j) = 2 * IW(j,:) .* diff(:,:,j);
    de_dwij(:,:,j) = -1 .* dvk_dyj .* dyj_dvj .* dvj_dwij(:,:,j);
    J = [J de_dwij(:,:,j)];
end
%Build jacobian
J = [J de_dwjk];
w_t1 = w - pinv(J' * J + mu*eye((n_inputs+1)*n_neurons)) * (J'*e);

%Recalculate:
IW =[];
for j=0:n_inputs-1
    IW = [IW w_t1(j*n_neurons+1:(j+1)*n_neurons)];
end
IW =IW';
LW = w_t1(n_inputs*n_neurons+1:end);
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
        w_t1 = w - pinv(J' * J + mu*eye((n_inputs+1)*n_neurons)) * (J'*e);
        IW =[];
        for j=0:n_inputs-1
            IW = [IW w_t1(j*n_neurons+1:(j+1)*n_neurons)];
        end
        IW =IW';
        LW = w_t1(n_inputs*n_neurons+1:end);
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