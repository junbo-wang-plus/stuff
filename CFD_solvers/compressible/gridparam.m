clear all; close all;

for k = 1:3
    load("grid_"+k+".mat")
    
    IL = length(x(:,1));
    JL = length(x(1,:));
    
    %%% params
    [dx,dy] = FV_dxdy(x,y,IL,JL);
    [DX,DY] = FV_DXDY(x,y,IL,JL);
    [S] = FV_S(dx,dy,IL,JL);   %right,top,left,bottom
    [nx,ny] = FV_n(dx,dy,S,IL,JL);
    Vol = FV_V(x,y,IL,JL);

%     save("gridparams_"+k+".mat","S","dx","dy","nx","ny","Vol","IL","JL","DX","DY")
end

%%% Functions
function [S] = FV_S(dx,dy,IL,JL)
    S = zeros(IL+1,JL+1,4); %right,top,left,bottom
    for i = 2:IL
        for j = 2:JL
            for m = 1:4
                S(i,j,m) = sqrt(dx(i,j,m)^2+dy(i,j,m)^2);
            end
        end
    end
end

function [dx,dy] = FV_dxdy(x,y,IL,JL)
    dx = zeros(IL+1,JL+1,4); %right,top,left,bottom
    dy = zeros(IL+1,JL+1,4); %right,top,left,bottom
    for i = 2:IL
        for j = 2:JL
            dx(i,j,1) = x(i,j) - x(i,j-1);
            dx(i,j,2) = x(i-1,j) - x(i,j);
            dx(i,j,3) = x(i-1,j-1) - x(i-1,j);
            dx(i,j,4) = x(i,j-1) - x(i-1,j-1);
            dy(i,j,1) = y(i,j) - y(i,j-1);
            dy(i,j,2) = y(i-1,j) - y(i,j);
            dy(i,j,3) = y(i-1,j-1) - y(i-1,j);
            dy(i,j,4) = y(i,j-1) - y(i-1,j-1);
        end
    end
end

function [DX,DY] = FV_DXDY(x,y,IL,JL)
    DX = zeros(IL+1,JL+1); %right,top,left,bottom
    DY = zeros(IL+1,JL+1); %right,top,left,bottom
    for i = 2:IL
        for j = 2:JL
            dx1 = abs(x(i,j)-x(i-1,j-1));
            dx2 = abs(x(i-1,j)-x(i,j-1));
            DX(i,j) = max(dx1,dx2);
            dy1 = abs(y(i,j)-y(i-1,j-1));
            dy2 = abs(y(i-1,j)-y(i,j-1));
            DY(i,j) = max(dy1,dy2);
        end
    end
end


function [nx,ny] = FV_n(dx,dy,S,IL,JL)
    nx = zeros(IL+1,JL+1,4);
    ny = zeros(IL+1,JL+1,4);
    for i = 2:IL
        for j = 2:JL
            for m = 1:2
                nx(i,j,m) = dy(i,j,m)/S(i,j,m);
                ny(i,j,m) = -dx(i,j,m)/S(i,j,m);
            end
            for m = 3:4
                nx(i,j,m) = -dy(i,j,m)/S(i,j,m);
                ny(i,j,m) = dx(i,j,m)/S(i,j,m);
            end
        end
    end
end

function V = FV_V(x,y,IL,JL)
    V = zeros(IL,JL);
    for i = 2:IL
        for j = 2:JL
            S1 = 1/2*abs((x(i,j)-x(i-1,j-1))*(y(i-1,j)-y(i-1,j-1)) ...
                -(x(i-1,j)-x(i-1,j-1))*(y(i,j)-y(i-1,j-1)));
            S2 = 1/2*abs((x(i,j-1)-x(i-1,j-1))*(y(i,j)-y(i-1,j-1)) ...
                -(x(i,j)-x(i-1,j-1))*(y(i,j-1)-y(i-1,j-1)));
            V(i,j) = S1+S2;
        end
    end
end


