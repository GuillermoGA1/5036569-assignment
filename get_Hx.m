function Hx = get_Hx(x,u)
%Obtains the jacobian for function h
u=x(1); v=x(2); w =x(3); C =x(4);

Hx = [-w*(C+1)/(u^2+w^2) 0 u*(C+1)/(u^2+w^2) atan(w/u);
    -u*v/(sqrt(u^2+w^2)*(u^2+v^2+w^2)) sqrt(u^2+w^2)/(u^2+v^2+w^2) -w*v/(sqrt(u^2+w^2)*(u^2+v^2+w^2)) 0;
    u/sqrt(u^2+v^2+w^2) v/sqrt(u^2+v^2+w^2) w/sqrt(u^2+v^2+w^2) 0];
end
