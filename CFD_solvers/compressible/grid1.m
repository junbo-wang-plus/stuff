close all;clear all;

%%% Define constants
IL = [42 82 162]; %cylinder surface
JL = [22 42 82]; %head space
PA = -3.5;
PC = -0.5;

for k = 1
    dtheta = -pi/2/(IL(k)-1);
    dx = (PA-PC)/(JL(k)-1);
    
    %%% Boundary conditions
    x = zeros(IL(k),JL(k));
    y = zeros(IL(k),JL(k));
    
    for i = 1:IL(k)
        x(i,1) = cos((i-1)*dtheta+pi)*abs(PC);
        y(i,1) = sin((i-1)*dtheta+pi)*abs(PC);
        x(i,JL(k)) = cos((i-1)*dtheta+pi)*abs(PA);
        y(i,JL(k)) = sin((i-1)*dtheta+pi)*abs(PA);
    end
    
%     scatter(x(:,:),y(:,:),'.')
    
    for j = 2:JL(k)-1
        x(1,j) = PC+(j-1)*dx;
        y(1,j) = y(1,1);
        y(IL(k),j) = -PC-(j-1)*dx;
        x(IL(k),j) = x(IL(k),1);
    end
    
%     scatter(x(:,:),y(:,:),'.')
    
    for i = 2:IL(k)-1
        for j = 2:JL(k)-1
            dx2 = (x(i,JL(k))-x(i,1))/(JL(k)-1);
            dy2 = (y(i,JL(k))-y(i,1))/(JL(k)-1);
            x(i,j) = x(i,1)+(j-1)*dx2;
            y(i,j) = y(i,1)+(j-1)*dy2;
        end
    end
%     scatter(x(:,:),y(:,:),'.')
    
    %%% 
    
    res = 1;
    n = 0;
    
    x_old = x;
    y_old = y;
    
    maxstep = 1;
    maxres = 1e-6;
    tic
    for m = 1:maxstep
        for i = 2:JL(k)-1
            for j = 2:JL(k)-1
                dxi = (x_old(i+1,j)-x_old(i-1,j))/2; dxj = (x_old(i,j+1)-x_old(i,j-1))/2;
                dyi = (y_old(i+1,j)-y_old(i-1,j))/2; dyj = (y_old(i,j+1)-y_old(i,j-1))/2;
                J = dxi*dyj - dxj*dyi;
                P = 0; Q = 3;
                R1 = -J^2*(P*dxi + Q*dxj); R2 = -J^2*(P*dyi + Q*dyj); 
                alpha = dxj^2+dyj^2;
                beta = dxi*dxj + dyi*dyj;
                gamma = dxi^2+dyi^2;
    
                x(i,j) = 1/(-2*(alpha+gamma)) ...
                *(R1-alpha*(x_old(i+1,j)+x_old(i-1,j)) ...
                +2*beta*(1/4*(x_old(i+1,j+1)-x_old(i-1,j+1) ...
                -x_old(i+1,j-1)+x_old(i-1,j-1)))-gamma ...
                *(x_old(i,j+1)+x_old(i,j-1)));
                y(i,j) = 1/(-2*(alpha+gamma)) ...
                *(R2-alpha*(y_old(i+1,j)+y_old(i-1,j))+2 ...
                *beta*(1/4*(y_old(i+1,j+1)-y_old(i-1,j+1) ...
                -y_old(i+1,j-1)+y_old(i-1,j-1)))-gamma ...
                *(y_old(i,j+1)+y_old(i,j-1)));
                diff_x = x-x_old;
                diff_y = y-y_old;
            end
        end
                res_x = max(max(abs(diff_x)));
                res_y = max(max(abs(diff_y)));
                res = (res_x>res_y)*res_x + (res_x<res_y)*res_y;
                x_old = x;
                y_old = y;
                if res <= maxres
                    break;
                end
    end
    disp(m)
    toc
    
    %%% Cell centers
    [x_FV,y_FV] = FV(x,y,IL(k),JL(k));
    
    %%% Plot
%     pos = [500 500 1000 1000];
%     figure('Position',pos);
%     for i = 1:IL(k)
%         plot(x(i,:),y(i,:),'-','Color',"black");
%         hold on
%     end
%     
%     for i = 1:JL(k)
%         plot(x(:,i),y(:,i),'-','Color',"black");
%     end
%     
%     scatter(x_FV(:,:),y_FV(:,:),'.',"red")
%     ylim([-0.3,-PA+0.3]);
%     xlim([PA-0.3,0.3]);
% %     title("Grid #"+k+", IL="+IL(k)+", JL="+JL(k));
%     xlabel('x (m)');
%     ylabel('y (m)');
%     fontsize(gcf,30,"points")
%     ax = gca;
%     ax.PlotBoxAspectRatio = [1 1 1];
%     hold off
%     saveas(gcf,"grid"+k+".jpg")
%     %%% Save data
%     save("grid_"+k+".mat","x","y","x_FV","y_FV")
end

% pos = [500 500 1000 1000];
% figure('Position',pos);
% plot(x(IL(1),:),y(IL(1),:),'-','Color',"blue",'LineWidth',2);
% hold on
% plot(x(:,JL(1)),y(:,JL(1)),'-','Color',"green",'LineWidth',2);
% plot(x(:,1),y(:,1),'-','Color',"red",'LineWidth',2);
% plot(x(1,:),y(1,:),'-','Color',"magenta",'LineWidth',2);
% legend("Exit","Inlet","Cylinder wall","Lower wall",'Location','northwest')
% 
% ylim([-0.3,-PA+0.3]);
% xlim([PA-0.3,0.3]);
% xlabel('x (m)');
% ylabel('y (m)');
% fontsize(gcf,30,"points")
% ax = gca;
% ax.PlotBoxAspectRatio = [1 1 1];
% hold off
% saveas(gcf,"gridbc.jpg")

%%%%%%%
% function out = alpha(x,y,delta_eta_x,delta_eta_y)
%     out = delta_eta_x^2+delta_eta_y^2;
% end
% 
% function out = beta(x,y,delta_xi_x,delta_xi_y,delta_eta_x,delta_eta_y)
%     delta_xi_x = (x(i+1,j)-x(i-1,j))/2;
%     delta_eta_x = (x(i,j+1)-x(i,j-1))/2;
%     delta_xi_y = (y(i+1,j)-y(i-1,j))/2;
%     delta_eta_y = (y(i,j+1)-y(i,j-1))/2;
%     out = delta_xi_x*delta_eta_x + delta_xi_y*delta_eta_y;
% end
% 
% function out = gamma(x,y,delta_xi_x,delta_xi_y)
%     delta_xi_x = (x(i+1,j)-x(i-1,j))/2;
%     delta_xi_y = (y(i+1,j)-y(i-1,j))/2;
%     out = delta_xi_x^2+delta_xi_y^2;
% end

function [xnew,ynew] = FV(x,y,IL,JL)
    xnew = zeros(IL+1,JL+1);
    ynew = zeros(IL+1,JL+1);
    for i = 2:IL
        for j = 2:JL
            xnew(i,j) = mean([x(i-1,j-1),x(i,j-1),x(i-1,j),x(i,j)]);
            ynew(i,j) = mean([y(i-1,j-1),y(i,j-1),y(i-1,j),y(i,j)]);
        end
        xnew(i,1) = mean([x(i-1,1),x(i,1)])*2-xnew(i,2);
        ynew(i,1) = mean([y(i-1,1),y(i,1)])*2-ynew(i,2);
        xnew(i,JL+1) = mean([x(i-1,JL),x(i,JL)])*2-xnew(i,JL);
        ynew(i,JL+1) = mean([y(i-1,JL),y(i,JL)])*2-ynew(i,JL);
    end
    for j = 2:JL
        ynew(1,j) = mean([y(1,j-1),y(1,j)])*2-ynew(2,j);
        xnew(1,j) = mean([x(1,j-1),x(1,j)])*2-xnew(2,j);
        ynew(IL+1,j) = mean([y(IL,j-1),y(IL,j)])*2-ynew(IL,j);
        xnew(IL+1,j) = mean([x(IL,j-1),x(IL,j)])*2-xnew(IL,j);
    end
    xnew(1,1) = 2*x(1,1)-xnew(2,2);
    ynew(1,1) = 2*y(1,1)-ynew(2,2);
    xnew(IL+1,1) = 2*x(IL,1)-xnew(IL,2);
    ynew(IL+1,1) = 2*y(IL,1)-ynew(IL,2);
    xnew(1,JL+1) = 2*x(1,JL)-xnew(2,JL);
    ynew(1,JL+1) = 2*y(1,JL)-ynew(2,JL);
    xnew(IL+1,JL+1) = 2*x(IL,JL)-xnew(IL,JL);
    ynew(IL+1,JL+1) = 2*y(IL,JL)-ynew(IL,JL);
end