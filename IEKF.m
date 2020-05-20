% Inputs:
% - U_k: input vector
% - Z_k: measurement vector
% - dt: used sample time
% - sigma_w: system noise std. dev.
% - sigma_v: measurement noise std. dev.
%
% Outputs:
% - X_HAT_K1_K1: one-step-ahead optimal state estimation vector
% - Z_K1_K: one-step-ahead measurement prediction vector
% - IEKF_COUNT: vector containing number of IEKF iterations per sample

load('F16traindata_CMabV_2020.mat');

%%%SIMULATION PARAMETERS%%%
dt              = 0.01;
N               = size(U_k,1);
epsilon         = 1e-10;
maxIterations   = 100;

%%%SET INITIAL VALUES%%%
Ex_0 = [155; 0.3; -0.15; 0.2];  % initial estimate, x_hat(0|0) == E{x_0} (or x_hat_0_0)
m = 3;  %Number of inputs: u_dot, v_dot, w_dot

% Initial estimate for covariance matrix
sigma_0  = [0.1 0.1 0.1 0.1];
P_0     = diag(sigma_0.^2);

%System noise
sigma_w = [1e-3 1e-3 1e-3 0]; 
Q = diag(sigma_w.^2);
n = length(sigma_w);

%Measurement noise
sigma_v = [0.035  0.013 0.11];
R = diag(sigma_v.^2);
nm = length(sigma_v);

G = eye(n); % noise input matrix
B = eye(m); % input matrix

%%%DEFINE ARRAYS TO STORE RESULTS%%%
X_K1_K1 = zeros(n,N);
P_K1_K1 = zeros(n,N);
SIGMA_COR = zeros(n,N);
Z_K1_K = zeros(nm,N);
IEKF_COUNT = zeros(N,1);

x_k1_k1 = Ex_0; % x(0|0)=E{x_0}
P_k1_k1 = P_0; % P(0|0)=P(0)


%%%RUN THE EXTENDED KALMAN FILTER%%%
tic;
ti = 0;
tf = dt;

%Run filter through all N samples
for k=1:N
    %Step 1: one step ahead prediction
    [t,x_k1_k] = ode45(@(t,x) get_f(x,U_k(k,:)),[ti tf],x_k1_k1);
    x_k1_k = x_k1_k(end,:)'; t = t(end);
    
    %Predict z(k+1,k)
    z_k1_k = get_h(x_k1_k, U_k(k,:));
    Z_K1_K(:,k) = z_k1_k;
    
    %Step 2: calculate jacobian of function f.
    Fx = get_Fx(x_k1_k1, U_k(k,:));
    
    %Step 3: discretize transition matrix
    [Phi,Gamma] = c2d(Fx,G,dt);
    
    %Step 4: calculate covariance matrix of state prediction error
    P_k1_k = Phi*P_k1_k1*Phi' + Gamma*Q*Gamma'; 
    P_pred = diag(P_k1_k);
    
    %ITERATIVE PART%
    %Set iteration parameters
    eta2 = x_k1_k;
    it_error = 2*epsilon;
    iterations = 0;
    
    while (it_error > epsilon)
        if (iterations >= maxIterations)
            fprintf('Terminating IEKF: exceeded max iterations (%d)\n', maxIterations);
            break
        end
        
        iterations = iterations + 1;
        
        
       %Step 5: calculate the jacobian for function h.
       eta1 = eta2;
       Hx = get_Hx(eta1, U_k(k,:)); 
         
       %Step 6: calculate Kalman gain matrix
       Inv = (Hx*P_k1_k*Hx' + R); %first get innovation matrix
       K_k1 = P_k1_k * Hx'/Inv;
       
       % Step 7: new observation state
       z_p = get_h(eta1, U_k(k,:));
       eta2 = x_k1_k + K_k1 * (Z_k(k,:)' - z_p - Hx*(x_k1_k - eta1));
       %Compute error (stop criteria)
       it_error = norm((eta2 - eta1), inf) / norm(eta1, inf);
    end
    
    IEKF_COUNT(k) = iterations;
    x_k1_k1 = eta2;
    
    %Step 8: calculate covariance matrix of state estimation error
    P_k1_k1 = (eye(n)-K_k1*Hx) *P_k1_k* (eye(n) - K_k1*Hx)' + K_k1*R*K_k1';
    sigma_cor = sqrt(diag(P_k1_k1));
    
    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % store results
    X_K1_K1(:,k) = x_k1_k1;
    P_K1_K1(:,k) = diag(P_k1_k1);
    SIGMA_COR(:,k) = sigma_cor;
end
toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = 0:0.01:0.01*(size(Z_k, 1)-1);

%Estimated measures are taken with bias
Z_k1_biased = Z_K1_K;
%Use estimated washup coefficient to get true alpha
Z_K1_K(1, :) = Z_K1_K(1, :) ./ (1 + X_K1_K1(4, :));

Z_k=Z_k';

figure(1);
subplot(3,1,1);
hold on
plot(t, Z_k(1, :), 'y','LineWidth',1.3);
hold on
plot(t, Z_k1_biased(1, :), 'm',t, Z_K1_K(1, :), 'b');
xlabel('Time [s]')
ylabel('\alpha [rad]');
title('Angle of attack');
legend({'Raw','Filtered-biased','Filtered-unbiased'},'Location','northwest');
grid on
subplot(3,1,2);
plot(t, Z_k(2, :), 'y',t, Z_K1_K(2, :), 'm');
grid on
xlabel('Time [s]')
ylabel('\beta [rad]');
title('Sideslip angle');
legend({'Raw','Filtered'},'Location','northwest');
subplot(3,1,3);
plot(t, Z_k(3, :), 'y',t, Z_K1_K(3, :), 'm');
grid on
xlabel('Time [s]')
ylabel('V_m[m/s]');
title('Aircraft velocity');
legend({'Raw','Filtered'},'Location','northwest');

figure(2);
subplot(2,1,1);
plot(t, IEKF_COUNT, 'b');
xlabel('Time [s]')
ylabel('Number of iterations');
title('Iterations required to reach desired error');
subplot(2,1,2);
plot(t,X_K1_K1(4,:));
xlabel('Time [s]')
ylabel('C_\alpha');
title('Washup coefficient estimate');

figure(3);
plot(Z_k(1, :), Z_k(2, :), 'y',Z_k1_biased(1, :),Z_k1_biased(2, :),'m',...
    Z_K1_K(1,:),Z_K1_K(2,:),'b');
grid on
xlabel('\alpha [rad]')
ylabel('\beta [rad]');
title('Angle of attack vs sideslip angle');
legend({'Raw','Filtered-biased','Filtered-unbiased'},'Location','northeast');

