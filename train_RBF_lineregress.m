function net= train_RBF_lineregress(XtrainData,YtrainData,n_neurons,RBFcenters,W_init)
rng(1019);
X = XtrainData;
%Network parameters:
n_inputs = size(XtrainData,2);
n_outputs = size(YtrainData,2);
net.centers = zeros(n_neurons,n_inputs);
net.trainAlg = 'linregress';
net.trainFunct = {'radbas','purelin'};
net.range = [min(X(:,1)) max(X(:,1));min(X(:,2)) max(X(:,2));min(X(:,3)) max(X(:,3))];
net.N_centers = size(net.centers);
net.name = 'rbf';

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

%Initialize weights;
if W_init == 1
IW = randn(n_inputs,n_neurons);
elseif W_init == 2
IW = ones(n_inputs,n_neurons);
end

diff = zeros(size(XtrainData,1),n_neurons,n_inputs);
v_j= zeros(size(XtrainData,1),n_neurons);
for j=1:n_inputs
    diff(:,:,j) = (XtrainData(:,j)-net.centers(:,j)').^2;
    v_j = v_j + (IW(j,:).^2) .* diff(:,:,j);
end
%v_j = (IW(1,:).^2) .* diff1 + (IW(2,:).^2) .* diff2 + (IW(3,:).^2) .* diff3;

%Get output (activation function) of hidden layer
phi_j = exp(-v_j);

%Optimize "a" (weights for outputs of hidden layer
%using linear regression
LW = pinv(phi_j) * YtrainData;
v_k = phi_j * LW;
%Output layer
y_k = v_k;

net.IW =IW';
net.LW=LW';

end

