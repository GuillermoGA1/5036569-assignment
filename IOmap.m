function plot = IOmap(XtestData, YtestData,y_k,plottitle,plotID)
evalres     = 0.05;
minXI       = -1*ones(1,2);
maxXI       = 1*ones(1,2);
Xplot       = [XtestData(:,1), XtestData(:,2)]';

yRBF    = y_k;

%%   Plotting results
%---------------------------------------------------------
%   ... creating triangulation (only needed for plotting)
TRIeval     = delaunayn(Xplot',{'Qbb','Qc','QJ1e-6'});

%   ... viewing angles
az = 120;
el = 45;

%   ... creating figure for RBF network
figure(plotID);
trisurf(TRIeval, Xplot(1, :)', Xplot(2, :)', yRBF', 'EdgeColor', 'none'); 
hold on;
% plot data points
plot3(XtestData(:,1),XtestData(:,2),YtestData,'.k');
grid on;
view(az, el);
titstring = sprintf(plottitle);
xlabel('\alpha [rad]');
ylabel('\beta [rad]');
zlabel('C_m [-]');
title(titstring,'Interpreter','Latex');

%   ... set fancy options for plotting FF network
set(gcf,'Renderer','OpenGL');
hold on;
poslight = light('Position',[0.5 .5 15],'Style','local');
hlight = camlight('headlight');
material([.3 .8 .9 25]);
minz = min(yRBF);
shading interp;
lighting phong;
drawnow();
end
