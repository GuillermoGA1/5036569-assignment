rng('default');
%IEKF;
close all
%% CALCULATE OLS
%Input order of polynomial of OLS and fractions of dataset for testing
order = 5;
train_fraction = 0.7;
test_fraction = 0.3;

X= Z_K1_K'; Y = Cm;
number_states = size(X,2);

%Split data into test and train datasets.
cv = cvpartition(size(X,1),'HoldOut',test_fraction);
idx = cv.test;
XtrainData = X(~idx,:);     YtrainData = Y(~idx,:);
XtestData  = X(idx,:);      YtestData  = Y(idx,:);

%Get OLS estimator
model = regression_matrix(order,number_states);
A= x2fx(XtrainData, model);
%Compute OLS estimator
theta = pinv(A)*YtrainData;
Y_estimate = A*theta;

%% PLOT TRAIN OUTPUT AND ESTIMATE OUTPUT FROM TRAINING DATA
% creating triangulation (only used for plotting here)
TRIeval = delaunayn(XtrainData(:,1:2));

%   viewing angles
az = 140;
el = 36;
% 
% create figures
plotID = 1001;
figure(plotID);
set(plotID, 'Position', [800 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'color', [1 1 1], 'PaperPositionMode', 'auto');
trisurf(TRIeval, XtrainData(:,1),XtrainData(:,2),Y_estimate, 'EdgeColor', 'none'); 
grid on;
hold on;

% plot data points
plot3(XtrainData(:,1),XtrainData(:,2),YtrainData,'.k'); % note that alpha_m = alpha, beta_m = beta, y = Cm
view(az, el);
ylabel('beta [rad]');
xlabel('alpha [rad]');
zlabel('C_m [-]');
title('F16 CM(\alpha_m, \beta_m) raw interpolation');
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
legend('Training data', 'Validation data', 'Test data');

%% Model error validation 

%Check first condition of BLUE, E{epsilon} = 0

model = regression_matrix(optimal_order,3);
A_optimal = x2fx(XtestData,model);
Y_est_errorval = A_optimal*theta_opt;
epsilon = YtestData-Y_est_errorval;

% Check second condition of BLUE: E{epsilon*epsilon'} = sigma^2*I
[epsilon_auto, lags] = xcorr(epsilon - mean(epsilon, 1));
epsilon_auto = epsilon_auto ./ max(epsilon_auto, 1);

% 95% confidence interval (two std. dev.)
conf_95 = sqrt(1 / size(epsilon_auto, 1)) * [-2; 2];

%PLOTS
figure; hold on
plot(lags, epsilon_auto(:, 1), 'bo-')
plot(lags, conf_95(1) * ones(1, length(lags)), 'r--')
plot(lags, conf_95(2) * ones(1, length(lags)), 'r--')
xlabel('Lag [-]', 'Interpreter', 'Latex')
ylabel('Normalized Autocorrelation $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual Normalized Autocorrelation: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)


figure;
subplot(2, 2, 1); hold on
scatter(XtestData(:, 1), epsilon(:, 1), 'b')
xlabel('$\alpha$ [rad]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $\alpha$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(2, 2, 2); hold on
scatter(XtestData(:, 2), epsilon(:, 1), 'b')
xlabel('$\beta$ [rad]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $\beta$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(2, 2, 3); hold on
scatter(XtestData(:, 3), epsilon(:, 1), 'b')
xlabel('$V$ [m/s]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $V$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(2, 2, 4); hold on
scatter(YtestData(:, 1), epsilon(:, 1), 'b')
xlabel('$C_m$ [-]', 'Interpreter', 'Latex')
ylabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual vs Model Input: $C_m$ vs $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
grid on


%% STATISTICAL VALIDATION

theta_cov_matrix = pinv(A_optimal) * (epsilon * epsilon') * A_optimal * pinv(A_optimal) / A_optimal';
sigma = diag(theta_cov_matrix);
figure;
subplot(1, 2, 1); hold on
bar(theta_opt);
xlabel('Index [-]', 'Interpreter', 'Latex')
ylabel('$\hat{\theta}_i$ [-]', 'Interpreter', 'Latex')
title('Parameter Value of OLS Estimator', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

subplot(1, 2, 2); hold on
plot(1:size(sigma, 1), sigma, 'b')
xlabel('Index [-]', 'Interpreter', 'Latex')
ylabel('$\sigma^2_{\hat{\theta}_i}$ [-]', 'Interpreter', 'Latex')
title('Parameter Variance of OLS Estimator', 'Interpreter', 'Latex', 'FontSize', 12)
grid on

