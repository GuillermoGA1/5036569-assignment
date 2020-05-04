function X_dot = get_f(x,u)

%Derives the state transition matix f, corresponding to:
%x_dot(t) = f(x(t), u(t), t)
%f is trivial, meaning that x_dot can be completely defined with the input
%vector. An extra row for the fourth state is added with zero
%because the fourth state C is just a constant.

X_dot = [u 0]';
end