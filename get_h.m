function Zn = get_h(x,U)
%Derives the observation matrix h, from system equation:
% z_n(t) = h(x(t),u(t),t)

%Assign variables from state vector to local variables
u = x(1); v = x(2); w = x(3); C=x(4);


%h is defined according to following equations;

alpha = atan(w/u)*(1+C);
beta = atan(v/sqrt(u^2+w^2));
V= sqrt(u^2 + v^2 + w^2);
Zn =[alpha beta V]';
end