function net= train_RBF_lineregress(XtrainData,YtrainData,n_neurons,RBFcenters,W_init)
%Network parameters:
n_inputs = size(XtrainData,2);
n_outputs = size(YtrainData,2);
net.centers = zeros(n_neurons,n_inputs);
net.trainAlg = 'linregress';
net.trainFunct = {'radbas','purelin'};
net.range = [-ones(n_inputs, 1) ones(n_inputs, 1)];
net.N_centers = size(net.centers);
net.name = 'rbf';

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

%Initialize weights;
if W_init == 1
IW = randn(n_inputs,n_neurons);
elseif W_init == 2
IW = ones(n_inputs,n_neurons);
end

diff1 = (XtrainData(:,1)-net.centers(:,1)').^2;
diff2 = (XtrainData(:,2)-net.centers(:,2)').^2;
diff3 = (XtrainData(:,3)-net.centers(:,3)').^2;
v_j = (IW(1,:).^2) .* diff1 + (IW(2,:).^2) .* diff2 + (IW(3,:).^2) .* diff3;

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

