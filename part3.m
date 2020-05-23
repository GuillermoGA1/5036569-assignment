%% Define data sets
load('F16traindata_reconstructed.mat','Z_K1_K','Cm');
close all;
X= Z_K1_K'; Y = Cm;
train_fraction = 0.8;
test_fraction = 0.2;
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
XtrainData = X(~idx,:);     YtrainData = Y(~idx,:);
XtestData  = X(idx,:);      YtestData  = Y(idx,:);

%% Linear regression RBF (part 3.1)
%Set network parameters
%No training parameters required 
n_neurons = 100;
RBFcenters = 1; %Leave 1 to use kmeans centers, 2 for uniformly distributed centers.
RBFnet = train_RBF_lineregress(XtrainData,YtrainData,n_neurons,RBFcenters);
Y_est_train = simRBF(XtrainData,RBFnet.IW',RBFnet.LW',RBFnet.centers);
Y_est_test = simRBF(XtestData,RBFnet.IW',RBFnet.LW',RBFnet.centers);

mse_train = immse(YtrainData,Y_est_train);
mse_test = immse(YtestData,Y_est_test);

figure(1);
hold on;
scatter(XtestData(:,1),YtestData);
plot(XtestData(:,1),Y_est_test);

figure(2);
xaxis = 1:size(YtestData,1);
plot(xaxis,YtestData,xaxis,Y_est_test);
