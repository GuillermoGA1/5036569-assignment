rng('default');
IEKF;
close all
%% CALCULATE OLS
%Input order of polynomial of OLS and fractions of dataset for testing
order = 3;
train_fraction = 0.7;
test_fraction = 0.3;

X= Z_K1_K'; Y = Cm;
number_states = size(X,2);
sigma_v = [0.035  0.013 0.11];
W = diag(sigma_v);

%Split data into test and train datasets.
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
XtrainData = X(~idx,:);     YtrainData = Y(~idx,:);
XtestData  = X(idx,:);      YtestData  = Y(idx,:);

%Get OLS estimator
model = regression_matrix(order,number_states);
A_train= x2fx(XtrainData, model);
A_test = x2fx(XtestData,model);
%Compute OLS estimator
theta_train = pinv(A_train)* YtrainData;
theta_test = pinv(A_test)* YtestData;
Y_estimate_train = A_train*theta_train;
Y_estimate_test = A_test*theta_test;

%% PLOT TRAIN OUTPUT AND ESTIMATE OUTPUT FROM TRAINING DATA
%PLOT OF TRAINING DATA%
% creating triangulation (only used for plotting here)
TRIeval = delaunayn(XtrainData(:,1:2));
%   viewing angles
az = 140; el=36;
figure(1);
subplot(1,2,1);
trisurf(TRIeval, XtrainData(:,1),XtrainData(:,2),Y_estimate_train, 'EdgeColor', 'none'); 
grid on;
hold on;
plot3(XtrainData(:,1),XtrainData(:,2),YtrainData,'.k'); % note that alpha_m = alpha, beta_m = beta, y = Cm
view(az, el);
ylabel('beta [rad]');
xlabel('alpha [rad]');
zlabel('C_m [-]');
title('F16 CM(\alpha_m, \beta_m)OLS estimator (training data)');
set(gcf,'Renderer','OpenGL');
hold on;
poslight = light('Position',[0.5 .5 15],'Style','local');
hlight = camlight('headlight');
material([.3 .8 .9 25]);
minz = min(Cm);
shading interp;
lighting phong;
drawnow();

%PLOT OF TEST DATA%
subplot(1,2,2);
TRIeval = delaunayn(XtestData(:,1:2));
set(1, 'Position', [800 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'color', [1 1 1], 'PaperPositionMode', 'auto');
trisurf(TRIeval, XtestData(:,1),XtestData(:,2),Y_estimate_test, 'EdgeColor', 'none'); 
grid on;
hold on;
plot3(XtestData(:,1),XtestData(:,2),YtestData,'.k'); % note that alpha_m = alpha, beta_m = beta, y = Cm
view(az, el);
ylabel('beta [rad]');
xlabel('alpha [rad]');
zlabel('C_m [-]');
title('F16 CM(\alpha_m, \beta_m) OLS estimator (test data)');
set(gcf,'Renderer','OpenGL');
hold on;
poslight = light('Position',[0.5 .5 15],'Style','local');
hlight = camlight('headlight');
material([.3 .8 .9 25]);
minz = min(Cm);
shading interp;
lighting phong;
drawnow();

%% Optimize OLS (influence of order on erro)
%Improve OLS 
max_order = 10;
[theta_opt, optimal_order, mse] = optimize_OLS(max_order,X,Y);

figure(2);
orders = 1:max_order;
plot(orders,mse(:,:,1),orders,mse(:,:,2),orders,mse(:,:,3));
xlabel('Polynomial order [-]');
ylabel('Mean square error [-]');
title('Influence of polynomial order in estimator performance');
legend('Training data', 'Validation data', 'Test data');

%% Model error validation 

%Check first condition of BLUE, E{epsilon} = 0

model = regression_matrix(optimal_order,3);
A_optimal = x2fx(XtestData,model);
Y_estimate_optimal = A_optimal*theta_opt;
epsilon = YtestData-Y_estimate_optimal;

% Check second condition of BLUE: E{epsilon*epsilon'} = sigma^2*I
[epsilon_auto, lags] = xcorr(epsilon - mean(epsilon, 1));
epsilon_auto = epsilon_auto ./ max(epsilon_auto, 1);

% 95% confidence interval (two std. dev.)
conf_95 = sqrt(1 / size(epsilon_auto, 1)) * [-2; 2];

%PLOTS
bins = [101 101 301];
figure; hold on
histogram(epsilon(:, 1), bins(1), 'FaceColor', 'b')
xlabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
ylabel('\# Residuals [-]', 'Interpreter', 'Latex')
title('Model Residual: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
axis([-0.02 0.02 0 200])
grid on

figure; hold on
plot(lags, epsilon_auto(:, 1), 'bo-')
plot(lags, conf_95(1) * ones(1, length(lags)), 'r--')
plot(lags, conf_95(2) * ones(1, length(lags)), 'r--')
xlabel('Lag [-]', 'Interpreter', 'Latex')
ylabel('Normalized Autocorrelation $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual Normalized Autocorrelation: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)


figure;
subplot(2, 2, 1); hold on
scatter(XtestData(:, 1), epsilon(:, 1), 'r')
xlabel('\alpha [rad]');
ylabel('Residual of C_m [-]');
title('Residual vs \alpha');
grid on

subplot(2, 2, 2); hold on
scatter(XtestData(:, 2), epsilon(:, 1), 'r')
xlabel('\beta [rad]');
ylabel('Residual of C_m [-]');
title('Residual vs \beta');
grid on

subplot(2, 2, 3); hold on
scatter(XtestData(:, 3), epsilon(:, 1), 'r')
xlabel('V_m [m/s]');
ylabel('Residual of C_m [-]');
title('Residual vs V_m');
grid on

subplot(2, 2, 4); hold on
scatter(YtestData(:, 1), epsilon(:, 1), 'r')
xlabel('$C_m$ [-])');
ylabel('Residual of C_m [-]');
title('Residual vs C_m');
grid on


%% STATISTICAL VALIDATION

theta_cov_matrix = pinv(A_optimal) * (epsilon * epsilon') * A_optimal * pinv(A_optimal) / A_optimal';
sigma = diag(theta_cov_matrix);
figure;
subplot(1, 2, 1); hold on
bar(theta_opt);
xlabel('Index [-]');
ylabel('$\hat{\theta}_i$ [-]', 'Interpreter', 'Latex');
title('Parameter Value of OLS Estimator');
grid on

subplot(1, 2, 2); hold on
plot(1:size(sigma, 1), sigma, 'b')
xlabel('Index [-]');
ylabel('$\sigma^2_{\hat{\theta}_i}$ [-]', 'Interpreter', 'Latex');
title('Parameter Variance of OLS Estimator');
grid on

