function y_k = simRBF(X,IW,LW,centers)

diff1 = (X(:,1)-centers(:,1)').^2;
diff2 = (X(:,2)-centers(:,2)').^2;
diff3 = (X(:,3)-centers(:,3)').^2;
v_j = (IW(1,:).^2) .* diff1 + (IW(2,:).^2) .* diff2 + (IW(3,:).^2) .* diff3;
y_j =exp(-v_j);
v_k = y_j * LW;
y_k = v_k;

end