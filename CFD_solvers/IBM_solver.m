%%% IBPM Solver %%%
clear all; close all;

%% Read small grid solution for initialization
% load("grid_params_50x50.mat") %small grid params
% load("sol_50_400.mat") % small grid solution
% dx_ini = dx; dy_ini = dy;
% ip_ini = ip; iu_ini = iu;
% iv_ini = iv; pos_ini = pos;
% q_pos_ini = q_pos; q_ini = q;
% p_ini = p;
% clear('T','w','q','p','q_prev','step','res')
%% Load main grid
load("grid_params.mat") % "nx","ny","nu","nv","np","nq","nw",...
                        % "dx","dy","iu","iv","ip","iw","pos",...
                        % "w_pos","q_pos"
%% Continuing/starting params
% load("sol_50_400.mat") % for continuing
% step = step - 1; %for continuing
c = 0.8; % Courant number
dt = c*min(dx,dy);
T = 0;
%% Flow params
uR = 0; vR = 0;
uT = 1; vT = 0;
uL = 0; vL = 0;
uB = 0; vB = 0;
Re = 400; % Reynolds number
nu = 1/Re; % viscosity
step_max = 200000;
res_max = 1e-6;
%% IBPM-specific
R = 0.2; %cylinder
L = 0.4;
x_c = 0.5;
y_c = 0.5;
% [xi,ds] = IBPM_cylinder(R,x_c,y_c,dx,dy); %body geometry
[xi,ds] = IBPM_triangle(L,x_c,y_c,dx,dy); %body geometry
nb = length(xi(:,1));
nf = 2*nb;
[H,E] = regintp(nq,nb,nf,nx,ny,iu,iv,dx,dy,ds,xi,q_pos); %reg/int operators
%% Initialize with all zeros
[q,g] = init(nx,ny,nq,np,nf,iu,iv,ip);
q_prev = q;
step = 1;
res = zeros(step_max,1);
res(1) = 1;
%% Initialize with a flow field for interpolation
% [q,p] = init2(nx,ny,nq,np,iu,iv,ip,iu_ini,iv_ini,ip_ini,q_ini,p_ini);
% q_prev = q;
% step = 1;
% res = zeros(step_max,1);
% res(1) = 1;
%% Solve
while res(step) > res_max && step < step_max
    step = step + 1; 
    [q_new,g] = FSM(q,g,q_prev,H,E,nu,nx,ny,np,nq,nf,iu,iv,ip,dx, ...
    dy,dt,uR,vR,uL,vL,uT,vT,uB,vB);
    res(step) = get_residual(q_new,q);
    q_prev = q;
    q = q_new; 
    T = T+dt;
    %%% Monitoring with residual plot
    semilogy(1:step,res(1:step)) 
    title("T="+T);
    drawnow
    %%%
end
% save("temp3_ibt_sol_"+nx+"_"+Re+".mat","Re","T","res","q","g","step","q_prev")
%%---------------------------------------------------------------%%
%% Supporting functions
%% Calculate residual
function out = get_residual(q_new,q)
    diff = abs(q_new-q);
    out = max(diff./q);
end
%% Fractional-step method
function [q_new,g_new] = FSM(q,g,q_prev,H,E,nu,nx,ny,np,nq,nf,iu,iv,ip,dx, ...
    dy,dt,uR,vR,uL,vL,uT,vT,uB,vB)
    % 1st step, unchanged
    % use q as initial guess for qF
    qF = FSM_1(q,q_prev,nu,nx,ny,nq,iu,iv,dx,dy,dt,uR,vR,uL,vL, ...
    uT,vT,uB,vB,q);
    % 2nd step
    % use g as initial guess for g_new
    g_new = FSM_2(qF,g,H,E,nx,ny,iu,iv,ip,nq,np,nf,dx,dy,uR,uL,vT,vB,dt,nu);
    % 3rd step
    q_new = FSM_3(qF,g_new,H,nx,ny,np,nq,nf,ip,iu,iv,dx,dy,dt,nu);
end

%% calculate qF, 1st step of FSM, unchanged
function x = FSM_1(q,q_prev,nu,nx,ny,nq,iu,iv,dx,dy,dt,uR,vR,uL,vL, ...
    uT,vT,uB,vB,q_ini)
    i = 0;
    eps = 1e-6;
    imax = nq*5;
    b = FSM_1_b(q,q_prev,nx,ny,iu,iv,nq,dx,dy,uR,vR,uL,vL,uT,vT,uB ...
        ,vB,dt,nu);
    x = q_ini; %initial guess
    r = b-Rtimes(x,nx,ny,iu,iv,nq,dx,dy,dt,nu); %r = b-A*x_ini
    d = r;
    delta_new = r.'*r;
    delta_0 = delta_new;

    while i<imax && delta_new>eps^2*delta_0
        Q = Rtimes(d,nx,ny,iu,iv,nq,dx,dy,dt,nu);
        alpha = delta_new/(d.'*Q);
        x = x+alpha*d;
        r = r-alpha*Q;
        delta_old = delta_new;
        delta_new = r.'*r;
        beta = delta_new/delta_old;
        d = r+beta*d;
        i = i+1;
    end
    % disp("step 1:"+i)
end
%% calculate g_new, 2nd step of FSM
function x = FSM_2(qF,g_ini,H,E,nx,ny,iu,iv,ip,nq,np,nf,dx,dy,uR,uL,vT,vB,dt,nu)
    i = 0;
    eps = 1e-6;
    imax = np*50;
    b = FSM_2_b(qF,E,nx,ny,iu,iv,ip,np,nf,dx,dy,uR,uL,vT,vB);
    x = g_ini; %initial guess
    r = b-FSM_2_A(x,H,E,nx,ny,nq,np,nf,ip,iu,iv,dx,dy,dt,nu); %r = b-A*x_ini
    d = r;
    delta_new = r.'*r;
    delta_0 = delta_new;
    
    while i<imax && delta_new>eps^2*delta_0
        Q = FSM_2_A(d,H,E,nx,ny,nq,np,nf,ip,iu,iv,dx,dy,dt,nu);
        alpha = delta_new/(d.'*Q);
        x = x+alpha*d;
        r = r-alpha*Q;
        delta_old = delta_new;
        delta_new = r.'*r;
        beta = delta_new/delta_old;
        d = r+beta*d;
        i = i+1;
    end    
    % disp("Step 2:"+i)
end

%% 3rd step of FSM
function out = FSM_3(qF,g,H,nx,ny,np,nq,nf,ip,iu,iv,dx,dy,dt,nu)
    Q = Qtimes(g,H,np,nf,nx,ny,nq,ip,iu,iv,dx,dy);
    RiQ = R_inv(Q,nx,ny,iu,iv,nq,dx,dy,dt,nu);
    out = qF - RiQ;
end
%% FSM components updated with IB stuff
%%% R (A of 1st step), unchanged
function out = Rtimes(q,nx,ny,iu,iv,nq,dx,dy,dt,nu)
    L = lap(q,nx,ny,iu,iv,nq,dx,dy);
    out = q-dt/2*nu*L;
end
%%% R inverse, unchanged
function out = R_inv(q,nx,ny,iu,iv,nq,dx,dy,dt,nu)
    L = lap(q,nx,ny,iu,iv,nq,dx,dy);
    out = q+dt/2*nu*L+(dt/2*nu)^2*lap(L,nx,ny,iu,iv,nq,dx,dy);
    % out = q;
end
%%% b of 1st step
function out = FSM_1_b(q,q_prev,nx,ny,iu,iv,nq,dx,dy,uR,vR,uL,vL,uT,vT,uB,vB,dt,nu)
    L = lap(q,nx,ny,iu,iv,nq,dx,dy);
    S = q+dt/2*nu*L;
    A = -adv(q,nx,ny,iu,iv,nq,dx,dy,uR,vR,uL,vL,uT,vT,uB,vB);
    A_prev = -adv(q_prev,nx,ny,iu,iv,nq,dx,dy,uR,vR,uL,vL,uT,vT,uB,vB);
    term2 = dt/2*(3*A-A_prev);
    L_bc = lapbc(q,nx,ny,iu,iv,nq,dx,dy,uR,vR,uL,vL,uT,vT,uB,vB);
    term3 = dt*nu*L_bc; %not sure about this one
    out = S+term2+term3;
end
%%% A of 2nd step, QT Ri Q, updated
function out = FSM_2_A(g,H,E,nx,ny,nq,np,nf,ip,iu,iv,dx,dy,dt,nu)
    Q = Qtimes(g,H,np,nf,nx,ny,nq,ip,iu,iv,dx,dy);
    RiQ = R_inv(Q,nx,ny,iu,iv,nq,dx,dy,dt,nu);
    out = QTtimes(RiQ,E,nx,ny,iu,iv,ip,np,dx,dy);
end
%%% b of 2nd step, updated
function out = FSM_2_b(qF,E,nx,ny,iu,iv,ip,np,nf,dx,dy,uR,uL,vT,vB)
    term1 = QTtimes(qF,E,nx,ny,iu,iv,ip,np,dx,dy);
    D_bc = divbc(nx,ny,ip,np,dx,dy,uR,uL,vT,vB); 
    r2 = zeros(nf,1); %no-slip
    term2 = [D_bc;-r2];
    out = term1+term2;
    % out = -out;
end
%% Basic operators
%%% Divergence
function out = div(q,nx,ny,iu,iv,ip,np,dx,dy)
    out = zeros(np,1);
    for j = 2:ny-1         
        for i = 2:nx-1
            % interior
            out(ip(i,j)) = (-q(iu(i,j))+q(iu(i+1,j)))/dx + ...
                           (-q(iv(i,j))+q(iv(i,j+1)))/dy;
        end
        % left interior, missing -q(iu(1,j))/dx
        out(ip(1,j)) = q(iu(2,j))/dx + ... 
                   (-q(iv(1,j))+q(iv(1,j+1)))/dy;
        % right interior, missing q(iu(nx+1,j))/dx
        out(ip(nx,j)) = -q(iu(nx,j))/dx + ...
                   (-q(iv(nx,j))+q(iv(nx,j+1)))/dy;
    end
    for i = 2:nx-1 
        % bottom interior, missing -q(iv(i,1))/dy
        out(ip(i,1)) = (-q(iu(i,1)) + q(iu(i+1,1)))/dx + q(iv(i,2))/dy;
        % top interior, missing q(iv(i,ny+1))/dy
        out(ip(i,ny)) = (-q(iu(i,ny)) + q(iu(i+1,ny)))/dx - q(iv(i,ny))/dy;
    end
    % bottom right, missing q(iu(nx+1,1))/dx - q(iv(nx,1))/dy
    out(ip(nx,1)) = -q(iu(nx,1))/dx+q(iv(nx,2))/dy;    
    % top left, missing -q(iu(1,ny))/dx + q(iv(1,ny+1))/dy
    out(ip(1,ny)) = q(iu(2,ny))/dx - q(iv(1,ny))/dy;
    % top right, missing q(iu(nx+1,ny))/dx + q(iv(nx,ny+1))/dy
    out(ip(nx,ny)) = -q(iu(nx,ny))/dx - q(iv(nx,ny))/dy;
end
%%% Boundary divergence
function out = divbc(nx,ny,ip,np,dx,dy,uR,uL,vT,vB)
    out = zeros(np,1);
    for j = 2:ny-1
        out(ip(1,j)) = -uL/dx; % left interior
        out(ip(nx,j)) = uR/dx; % right interior
    end
    for i = 2:nx-1
        out(ip(i,1)) = -vB/dy; % bottom interior
        out(ip(i,ny)) = vT/dy; % top interior
    end
    out(ip(nx,1)) = uR/dx - vB/dy; % bottom right
    out(ip(1,ny)) = -uL/dx + vT/dy; % top left
    out(ip(nx,ny)) = uR/dx + vT/dy; % top right
end

%%% Gradient
function out = grad(p,nx,ny,nq,ip,iu,iv,dx,dy)
    out = zeros(nq,1);
    %bottom left of u
    out(iu(2,1)) = p(ip(2,1))/dx;
    for i = 3:nx
        %bottom interior
        out(iu(i,1)) = (p(ip(i,1))-p(ip(i-1,1)))/dx;
    end
    %bottom left of v
    out(iv(1,2)) = p(ip(1,2))/dy;
    for j = 3:ny
        %left interior
        out(iv(1,j)) = (p(ip(1,j))-p(ip(1,j-1)))/dy;
    end
    %interior
    for j = 2:ny
        for i = 2:nx
            out(iu(i,j)) = (p(ip(i,j))-p(ip(i-1,j)))/dx;
            out(iv(i,j)) = (p(ip(i,j))-p(ip(i,j-1)))/dy;
        end
    end
end

%%% Laplacian
function out = lap(q,nx,ny,iu,iv,nq,dx,dy)
    out = zeros(nq,1);
    for j = 2:ny-1
        for i = 3:nx-1
            % interior for Lx
            out(iu(i,j)) = (q(iu(i-1,j))-2*q(iu(i,j))+q(iu(i+1,j)))/(dx^2)...
                + (q(iu(i,j-1))-2*q(iu(i,j))+q(iu(i,j+1)))/(dy^2);
        end
    end
    % boundary for Lx
    for j = 2:ny-1
        % right, missing q(iu(nx+1,j))
        out(iu(nx,j)) = (q(iu(nx-1,j))-2*q(iu(nx,j)))/(dx^2)...
                + (q(iu(nx,j-1))-2*q(iu(nx,j))+q(iu(nx,j+1)))/(dy^2);
        % left, missing q(iu(1,j))
        out(iu(2,j)) = (-2*q(iu(2,j))+q(iu(3,j)))/(dx^2)...
                + (q(iu(2,j-1))-2*q(iu(2,j))+q(iu(2,j+1)))/(dy^2);
    end
    for i = 3:nx-1
        % top, missing q(iu(i,ny+1))
        out(iu(i,ny)) = (q(iu(i-1,ny))-2*q(iu(i,ny))+q(iu(i+1,ny)))/(dx^2)...
                + (q(iu(i,ny-1))-2*q(iu(i,ny)))/(dy^2);
        % bottom, missing q(iu(i,1-1))
        out(iu(i,1)) = (q(iu(i-1,1))-2*q(iu(i,1))+q(iu(i+1,1)))/(dx^2)...
                + (-2*q(iu(i,1))+q(iu(i,2)))/(dy^2);
    end
        % bottom-left corner, missing q(iu(1,1)) and q(iu(2,1-1))
    out(iu(2,1)) = (-2*q(iu(2,1))+q(iu(3,1)))/(dx^2)...
                + (-2*q(iu(2,1))+q(iu(2,2)))/(dy^2);
        % bottom-right corner, missing q(iu(nx+1,1)) and q(iu(nx,1-1))
    out(iu(nx,1)) = (q(iu(nx-1,1))-2*q(iu(nx,1)))/(dx^2)...
                + (-2*q(iu(nx,1))+q(iu(nx,2)))/(dy^2);
        % top-left corner, missing q(iu(1,ny)) and q(iu(2,ny+1))
    out(iu(2,ny)) = (-2*q(iu(2,ny))+q(iu(3,ny)))/(dx^2)...
                + (q(iu(2,ny-1))-2*q(iu(2,ny)))/(dy^2);
        % top-right corner, missing q(iu(nx+1,ny)) and q(iu(nx,ny+1))
    out(iu(nx,ny)) = (q(iu(nx-1,ny))-2*q(iu(nx,ny)))/(dx^2)...
                + (q(iu(nx,ny-1))-2*q(iu(nx,ny)))/(dy^2);
    for j = 3:ny-1
        for i = 2:nx-1
            % interior for Ly
            out(iv(i,j)) = (q(iv(i-1,j))-2*q(iv(i,j))+q(iv(i+1,j)))/(dx^2)...
                + (q(iv(i,j-1))-2*q(iv(i,j))+q(iv(i,j+1)))/(dy^2);
        end
    end
    % boundary for Ly
    for j = 3:ny-1
        % right, missing q(iv(nx+1,j))
        out(iv(nx,j)) = (q(iv(nx-1,j))-2*q(iv(nx,j)))/(dx^2)...
                + (q(iv(nx,j-1))-2*q(iv(nx,j))+q(iv(nx,j+1)))/(dy^2);
        % left, missing q(iv(1-1,j))
        out(iv(1,j)) = (-2*q(iv(1,j))+q(iv(2,j)))/(dx^2)...
                + (q(iv(1,j-1))-2*q(iv(1,j))+q(iv(1,j+1)))/(dy^2);
    end
    for i = 2:nx-1
        % top, missing q(iv(i,ny+1))
        out(iv(i,ny)) = (q(iv(i-1,ny))-2*q(iv(i,ny))+q(iv(i+1,ny)))/(dx^2)...
                + (q(iv(i,ny-1))-2*q(iv(i,ny)))/(dy^2);
        % bottom, missing q(iv(i,1))
        out(iv(i,2)) = (q(iv(i-1,2))-2*q(iv(i,2))+q(iv(i+1,2)))/(dx^2)...
                + (-2*q(iv(i,2))+q(iv(i,3)))/(dy^2);
    end
        % bottom-left corner, missing q(iv(1-1,2)) and q(iv(1,1))
    out(iv(1,2)) = (-2*q(iv(1,2))+q(iv(2,2)))/(dx^2)...
                + (-2*q(iv(1,2))+q(iv(1,3)))/(dy^2);
        % bottom-right corner, missing q(iv(nx+1,2)) and q(iv(nx,1))
    out(iv(nx,2)) = (q(iv(nx-1,2))-2*q(iv(nx,2)))/(dx^2)...
                + (-2*q(iv(nx,2))+q(iv(nx,3)))/(dy^2);
        % top-left corner, missing q(iv(1-1,ny)) and q(iv(1,ny+1))
    out(iv(1,ny)) = (-2*q(iv(1,ny))+q(iv(2,ny)))/(dx^2)...
                + (q(iv(1,ny-1))-2*q(iv(1,ny)))/(dy^2);
        % top-right corner, missing q(iv(nx+1,ny)) and q(iv(nx,ny+1))
    out(iv(nx,ny)) = (q(iv(nx-1,ny))-2*q(iv(nx,ny)))/(dx^2)...
                + (q(iv(nx,ny-1))-2*q(iv(nx,ny)))/(dy^2);
end
%%% Boundary Laplacian
function out = lapbc(q,nx,ny,iu,iv,nq,dx,dy,uR,vR,uL,vL,uT,vT,uB,vB)
    out = zeros(nq,1);
    % boundary for Lx
    for j = 2:ny-1
        % right, no extrapolation, q(iu(nx+1,j))=uR
        out(iu(nx,j)) = uR/(dx^2);
        % left, no extrapolation, q(iu(1,j))=uL
        out(iu(2,j)) = uL/(dx^2);
    end
    for i = 3:nx-1
        % top, with extrapolation of q(iu(i,ny+1))=2*uT-q(iu(i,ny))
        out(iu(i,ny)) = (2*uT-q(iu(i,ny)))/(dy^2);
        % bottom, with extrapolation of q(iu(i,1-1))=2*uB-q(iu(i,1))
        out(iu(i,1)) = (2*uB-q(iu(i,1)))/(dy^2);
    end
        % bottom-left corner, with extrapolation of uB and given uL
    out(iu(2,1)) = uL/(dx^2) + (2*uB-q(iu(2,1)))/(dy^2);
        % bottom-right corner, with extrapolation of uB and given uR
    out(iu(nx,1)) = uR/(dx^2) + (2*uB-q(iu(nx,1)))/(dy^2);
        % top-left corner, with extrapolation of uT and given uL;
    out(iu(2,ny)) = uL/(dx^2) + (2*uT-q(iu(2,ny)))/(dy^2);
        % top-right corner, with extrapolation of uT and given uR;
    out(iu(nx,ny)) = uR/(dx^2) + (2*uT-q(iu(nx,ny)))/(dy^2);

    % boundary for Ly
    for j = 3:ny-1
        % right, with extrapolation of q(iv(nx+1,j))=2*vR-q(iv(nx,j))
        out(iv(nx,j)) = (2*vR-q(iv(nx,j)))/(dx^2);
        % left, with extrapolation of q(iv(1-1,j))=2*vL-q(iv(1,j))
        out(iv(1,j)) = (2*vL-q(iv(1,j)))/(dx^2);
    end
    for i = 2:nx-1
        % top, no extrapolation, q(iv(i,ny+1))=vT
        out(iv(i,ny)) = vT/(dy^2);
        % bottom, no extrapolation, q(iv(i,1))=vB
        out(iv(i,2)) = vB/(dy^2);
    end
        % bottom-left corner, with extrapolation of vL and given vB
    out(iv(1,2)) = (2*vL-q(iv(1,2)))/(dx^2) + vB/(dy^2);
        % bottom-right corner, with extrapolation of vR and given vB
    out(iv(nx,2)) = (2*vR-q(iv(nx,2)))/(dx^2) + vB/(dy^2);
        % top-left corner, with extrapolation of vL and given vT
    out(iv(1,ny)) = (2*vL-q(iv(1,ny)))/(dx^2) + vT/(dy^2);
        % top-right corner, with extrapolation of vR and given vT
    out(iv(nx,ny)) = (2*vR-q(iv(nx,ny)))/(dx^2) + vT/(dy^2);
end

%%% Non-linear advection
function out = adv(q,nx,ny,iu,iv,nq,dx,dy,uR,vR,uL,vL,uT,vT,uB,vB)
    out = zeros(nq,1);
    %%%%%%%%%%%%%%%%%%%%%
    % Lx interior
    for j = 2:ny-1
        for i = 3:nx-1
            ubarxR = 0.5*( q(iu(i+1,j)) + q(iu(i,j)) );
            ubarxL = 0.5*( q(iu(i-1,j)) + q(iu(i,j)) );        
            
            ubaryT = 0.5*( q(iu(i,j+1)) + q(iu(i,j)) );
            vbarxT = 0.5*( q(iv(i-1,j+1)) + q(iv(i,j+1)) );
            ubaryB = 0.5*( q(iu(i,j-1)) + q(iu(i,j)) );
            vbarxB = 0.5*( q(iv(i-1,j)) + q(iv(i,j)) );
            out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                           (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
        end
    end
    % Lx left boundary, use uL directly
    i = 2;
    for j = 2:ny-1
        ubarxR = 0.5*( q(iu(i+1,j)) + q(iu(i,j)) );
        ubarxL = 0.5*( uL           + q(iu(i,j)) );        
        
        ubaryT = 0.5*( q(iu(i,j+1)) + q(iu(i,j)) );
        vbarxT = 0.5*( q(iv(i-1,j+1)) + q(iv(i,j+1)) );
        ubaryB = 0.5*( q(iu(i,j-1)) + q(iu(i,j)) );
        vbarxB = 0.5*( q(iv(i-1,j)) + q(iv(i,j)) );
        out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                       (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    end
    % Lx right boundary, use uR directly
    i = nx;
    for j = 2:ny-1
        ubarxR = 0.5*( uR           + q(iu(i,j)) );
        ubarxL = 0.5*( q(iu(i-1,j)) + q(iu(i,j)) );        
        
        ubaryT = 0.5*( q(iu(i,j+1)) + q(iu(i,j)) );
        vbarxT = 0.5*( q(iv(i-1,j+1)) + q(iv(i,j+1)) );
        ubaryB = 0.5*( q(iu(i,j-1)) + q(iu(i,j)) );
        vbarxB = 0.5*( q(iv(i-1,j)) + q(iv(i,j)) );
        out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                       (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    end
    % Lx top boundary, use vT directly and extrapolate with uT
    j = ny;
    for i = 3:nx-1
        ubarxR = 0.5*( q(iu(i+1,j)) + q(iu(i,j)) );
        ubarxL = 0.5*( q(iu(i-1,j)) + q(iu(i,j)) );        
        
        ubaryT = uT;
        vbarxT = vT;
        ubaryB = 0.5*( q(iu(i,j-1)) + q(iu(i,j)) );
        vbarxB = 0.5*( q(iv(i-1,j)) + q(iv(i,j)) );
        out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                       (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    end
    % Lx bottom boundary, use vB directly and extrapolate with uB
    j = 1;
    for i = 3:nx-1
        ubarxR = 0.5*( q(iu(i+1,j)) + q(iu(i,j)) );
        ubarxL = 0.5*( q(iu(i-1,j)) + q(iu(i,j)) );        
        
        ubaryT = 0.5*( q(iu(i,j+1)) + q(iu(i,j)) );
        vbarxT = 0.5*( q(iv(i-1,j+1)) + q(iv(i,j+1)) );
        ubaryB = uB;
        vbarxB = vB;
        out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                       (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    end
    % Lx bottom-left, use uL and vB directly and extrapolate with uB
    j = 1; i = 2;
    ubarxR = 0.5*( q(iu(i+1,j)) + q(iu(i,j)) );
    ubarxL = 0.5*( uL           + q(iu(i,j)) );        
    
    ubaryT = 0.5*( q(iu(i,j+1)) + q(iu(i,j)) );
    vbarxT = 0.5*( q(iv(i-1,j+1)) + q(iv(i,j+1)) );
    ubaryB = uB;
    vbarxB = vB;
    out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                   (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    % Lx bottom-right, use uR and vB directly and extrapolate with uB
    j = 1; i = nx;
    ubarxR = 0.5*( uR           + q(iu(i,j)) );
    ubarxL = 0.5*( q(iu(i-1,j)) + q(iu(i,j)) );        
    
    ubaryT = 0.5*( q(iu(i,j+1)) + q(iu(i,j)) );
    vbarxT = 0.5*( q(iv(i-1,j+1)) + q(iv(i,j+1)) );
    ubaryB = uB;
    vbarxB = vB;
    out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                   (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    % Lx top-left, use uL and vT directly and extrapolate wtih uT
    j = ny; i = 2;
    ubarxR = 0.5*( q(iu(i+1,j)) + q(iu(i,j)) );
    ubarxL = 0.5*( uL           + q(iu(i,j)) );        
    
    ubaryT = uT;
    vbarxT = vT;
    ubaryB = 0.5*( q(iu(i,j-1)) + q(iu(i,j)) );
    vbarxB = 0.5*( q(iv(i-1,j)) + q(iv(i,j)) );
    out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                   (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    % Lx top-right, use uR and vT directly and extrapolate with uT
    j = ny; i = nx;
    ubarxR = 0.5*( uR           + q(iu(i,j)) );
    ubarxL = 0.5*( q(iu(i-1,j)) + q(iu(i,j)) );        
    
    ubaryT = uT;
    vbarxT = vT;
    ubaryB = 0.5*( q(iu(i,j-1)) + q(iu(i,j)) );
    vbarxB = 0.5*( q(iv(i-1,j)) + q(iv(i,j)) );
    out(iu(i,j)) = (ubarxR^2 - ubarxL^2)/dx + ...
                   (ubaryT*vbarxT - ubaryB*vbarxB)/dy;
    %%%%%%%%%%%%%%%%%%%%%
    % Ly interior
    for j = 3:ny-1
        for i = 2:nx-1
            ubaryR = 0.5*( q(iu(i+1,j)) + q(iu(i+1,j-1)) );
            vbarxR = 0.5*( q(iv(i,j)) + q(iv(i+1,j)) );
            ubaryL = 0.5*( q(iu(i,j)) + q(iu(i,j-1)) );
            vbarxL = 0.5*( q(iv(i,j)) + q(iv(i-1,j)) );

            vbaryT = 0.5*( q(iv(i,j+1)) + q(iv(i,j)) );
            vbaryB = 0.5*( q(iv(i,j-1)) + q(iv(i,j)) );
            out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                           (vbaryT^2-vbaryB^2)/dy;
        end
    end        
    % Ly left boundary, use uL directly and extrapolate with vL
    i = 1;
    for j = 3:ny-1
        ubaryR = 0.5*( q(iu(i+1,j)) + q(iu(i+1,j-1)) );
        vbarxR = 0.5*( q(iv(i,j)) + q(iv(i+1,j)) );
        ubaryL = uL;
        vbarxL = vL;

        vbaryT = 0.5*( q(iv(i,j+1)) + q(iv(i,j)) );
        vbaryB = 0.5*( q(iv(i,j-1)) + q(iv(i,j)) );
        out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                       (vbaryT^2-vbaryB^2)/dy;
    end
    % Ly right boundary, use uR directly and extrapolate with vR
    i = nx;
    for j = 3:ny-1
        ubaryR = uR;
        vbarxR = vR;
        ubaryL = 0.5*( q(iu(i,j)) + q(iu(i,j-1)) );
        vbarxL = 0.5*( q(iv(i,j)) + q(iv(i-1,j)) );

        vbaryT = 0.5*( q(iv(i,j+1)) + q(iv(i,j)) );
        vbaryB = 0.5*( q(iv(i,j-1)) + q(iv(i,j)) );
        out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                       (vbaryT^2-vbaryB^2)/dy;
    end
    % Ly top boundary, use vT directly
    j = ny;
    for i = 2:nx-1
        ubaryR = 0.5*( q(iu(i+1,j)) + q(iu(i+1,j-1)) );
        vbarxR = 0.5*( q(iv(i,j)) + q(iv(i+1,j)) );
        ubaryL = 0.5*( q(iu(i,j)) + q(iu(i,j-1)) );
        vbarxL = 0.5*( q(iv(i,j)) + q(iv(i-1,j)) );

        vbaryT = 0.5*( vT           + q(iv(i,j)) );
        vbaryB = 0.5*( q(iv(i,j-1)) + q(iv(i,j)) );
        out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                       (vbaryT^2-vbaryB^2)/dy;
    end
    % Ly bottom boundary, use vB directly
    j = 2;
    for i = 2:nx-1
        ubaryR = 0.5*( q(iu(i+1,j)) + q(iu(i+1,j-1)) );
        vbarxR = 0.5*( q(iv(i,j)) + q(iv(i+1,j)) );
        ubaryL = 0.5*( q(iu(i,j)) + q(iu(i,j-1)) );
        vbarxL = 0.5*( q(iv(i,j)) + q(iv(i-1,j)) );

        vbaryT = 0.5*( q(iv(i,j+1)) + q(iv(i,j)) );
        vbaryB = 0.5*( vB           + q(iv(i,j)) );
        out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                       (vbaryT^2-vbaryB^2)/dy;
    end
    % Ly bottom-left, use uL and vB directly and extrapolate with vL
    i = 1; j = 2;
    ubaryR = 0.5*( q(iu(i+1,j)) + q(iu(i+1,j-1)) );
    vbarxR = 0.5*( q(iv(i,j)) + q(iv(i+1,j)) );
    ubaryL = uL;
    vbarxL = vL;

    vbaryT = 0.5*( q(iv(i,j+1)) + q(iv(i,j)) );
    vbaryB = 0.5*( vB           + q(iv(i,j)) );
    out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                   (vbaryT^2-vbaryB^2)/dy;
    % Ly bottom-right, use uR and vB directly and extrapolate with vR
    i = nx; j = 2;
    ubaryR = uR;
    vbarxR = vR;
    ubaryL = 0.5*( q(iu(i,j)) + q(iu(i,j-1)) );
    vbarxL = 0.5*( q(iv(i,j)) + q(iv(i-1,j)) );

    vbaryT = 0.5*( q(iv(i,j+1)) + q(iv(i,j)) );
    vbaryB = 0.5*( vB           + q(iv(i,j)) );
    out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                   (vbaryT^2-vbaryB^2)/dy;
    % Ly top-left, use uL and vT directly and extrapolate with vL
    i = 1; j = ny;
    ubaryR = 0.5*( q(iu(i+1,j)) + q(iu(i+1,j-1)) );
    vbarxR = 0.5*( q(iv(i,j)) + q(iv(i+1,j)) );
    ubaryL = uL;
    vbarxL = vL;

    vbaryT = 0.5*( vT           + q(iv(i,j)) );
    vbaryB = 0.5*( q(iv(i,j-1)) + q(iv(i,j)) );
    out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                   (vbaryT^2-vbaryB^2)/dy;
    % Ly top-right, use uR and vT directly and extrapolate with vR
    i = nx; j = ny;
    ubaryR = uR;
    vbarxR = vR;
    ubaryL = 0.5*( q(iu(i,j)) + q(iu(i,j-1)) );
    vbarxL = 0.5*( q(iv(i,j)) + q(iv(i-1,j)) );

    vbaryT = 0.5*( vT           + q(iv(i,j)) );
    vbaryB = 0.5*( q(iv(i,j-1)) + q(iv(i,j)) );
    out(iv(i,j)) = (ubaryR*vbarxR - ubaryL*vbarxL)/dx + ...
                   (vbaryT^2-vbaryB^2)/dy;
    % out = -out;
end
%% Visualizations
function visualizeq(q,q_pos,nx,ny,iu,iv)
    u = zeros(nx-1,ny,1); u_pos = zeros(nx-1,ny,2);
    v = zeros(nx,ny-1,1); v_pos = zeros(nx,ny-1,2);
    for j = 1:ny
        for i = 2:nx % assign u to 2D matrix
            u(i-1,j) = q(iu(i,j));
            u_pos(i-1,j,:) = q_pos(iu(i,j),:);
        end
    end
    for j = 2:ny
        for i = 1:nx % assign v to 2D matrix
            v(i,j-1) = q(iv(i,j));
            v_pos(i,j-1,:) = q_pos(iv(i,j),:);
        end
    end
    figure('Position', [400 400 500 500])
    contourf(u_pos(:,:,1),u_pos(:,:,2),u(:,:,1),'LineColor','none')
    colorbar
    title("Velocity $u$ (m/s)","Interpreter","latex")
    fontsize(gcf,16,"points")
    drawnow
    figure('Position', [400 400 500 500])
    contourf(v_pos(:,:,1),v_pos(:,:,2),v(:,:,1),'LineColor','none')
    colorbar
    title("Velocity $v$ (m/s)","Interpreter","latex")
    fontsize(gcf,16,"points")
end

function [u,u_pos] = get_u(q,q_pos,nx,ny,iu,iv)
    u = zeros(nx-1,ny,1); u_pos = zeros(nx-1,ny,2);
    % v = zeros(nx,ny-1,1); v_pos = zeros(nx,ny-1,2);
    for j = 1:ny
        for i = 2:nx % assign u to 2D matrix
            u(i-1,j) = q(iu(i,j));
            u_pos(i-1,j,:) = q_pos(iu(i,j),:);
        end
    end
    for j = 2:ny
        for i = 1:nx % assign v to 2D matrix
            v(i,j-1) = q(iv(i,j));
            v_pos(i,j-1,:) = q_pos(iv(i,j),:);
        end
    end
end

function out = get_vorticity(q,iu,iv,nx,ny,dx,dy)
    out = zeros(nx-1,ny-1);
    for j = 1:ny-1
        for i = 1:nx-1
            out(i,j) = (q(iv(i+1,j+1))-q(iv(i,j+1)))/dx - ...
                (q(iu(i+1,j+1))-q(iu(i+1,j)))/dy;
        end
    end
end

function visualizew(w,iw,w_pos,nx,ny)
    pos_new = zeros(nx-1,ny-1,2);
    for j = 1:ny-1
        for i = 1:nx-1
            pos_new(i,j,:) = w_pos(iw(i,j),:);
        end
    end
    figure('Position', [400 400 500 500])
    contour(pos_new(:,:,1),pos_new(:,:,2),w(:,:,1),'LineColor','none')
    colorbar
    title("Vorticity $\omega$ ($\mathrm(m)/\mathrm(s)^2$)", ...
        "Interpreter","latex")
    fontsize(gcf,16,"points")
    drawnow
end

function [u_c,v_c] = centerline_q(q,nx,ny,iu,iv)
    u_c = zeros(ny,1);
    v_c = zeros(ny-1,1);
    x = nx/2+1;
    % directly read u
    for j = 1:ny
        u_c(j) = q(iu(x,j));
    end
    % average v
    for j = 1:ny-1
        v_c(j) = 1/2*(q(iv(x-1,j+1)) + q(iv(x,j+1)));
    end
end
%% Initialization
function [q,g] = init(nx,ny,nq,np,nf,iu,iv,ip)
    q = zeros(nq,1);
    g = zeros(np+nf,1);
    for j = 1:ny
        for i = 2:nx
            q(iu(i,j)) = 1;
        end
    end
    for j = 2:ny
        for i = 1:nx
            q(iv(i,j)) = 0;
        end
    end
    % for i = 2:nx
    %     g(ip(i,1)) = 1;
    % end
    % for j = 2:ny
    %     for i = 1:nx
    %         g(ip(i,j)) = 1;
    %     end
    % end
end

function [q,p] = init2(nx,ny,nq,np,iu,iv,ip,iu_ini,iv_ini,ip_ini, ...
    q_ini,p_ini)
    N = 50;
    s_x = nx/N; s_y = ny/N;
    q = zeros(nq,1);
    p = zeros(np,1);
    % interpolating u
    for j = 1:ny-1
        for i = 2:nx-1
            x = floor(i/s_x)+1; y = floor(j/s_y)+1;
            if x == 1
                q(iu(i,j)) = q_ini(iu_ini(2,y));
            else
                q(iu(i,j)) = q_ini(iu_ini(x,y));
            end
        end
    end
    for j = 1:ny-1
        y = fix(j/s_y)+1;
        q(iu(nx,j)) = q_ini(iu_ini(N,y));
    end
    for i = 2:nx-1
        x = fix(i/s_x)+1;
        if x == 1
            q(iu(i,ny)) = q_ini(iu_ini(2,N));
        else
            q(iu(i,ny)) = q_ini(iu_ini(x,N));
        end
    end
    q(iu(nx,ny)) = q_ini(iu_ini(N,N));
    % interpolating v
    for j = 2:ny-1
        for i = 1:nx-1
            x = fix(i/s_x)+1; y = fix(j/s_y)+1;
            if y == 1
                q(iv(i,j)) = q_ini(iv_ini(x,2));
            else
                q(iv(i,j)) = q_ini(iv_ini(x,y));
            end
        end
    end
    for j = 2:ny-1
        y = fix(j/s_y)+1;
        if y == 1
            q(iv(nx,j)) = q_ini(iv_ini(N,2));
        else
            q(iv(nx,j)) = q_ini(iv_ini(N,y));
        end
    end
    for i = 2:nx-1
        x = fix(i/s_x)+1;
        q(iv(i,ny)) = q_ini(iv_ini(x,N));
    end
    q(iv(nx,ny)) = q_ini(iv_ini(N,N));
    % interpolating p
    for i = 2:nx-1
        x = fix(i/s_x)+1;
        if x == 1
            p(ip(i,1)) = p_ini(ip_ini(2,1));
        else
            p(ip(i,1)) = p_ini(ip_ini(x,1));
        end
    end
    for j = 2:ny-1
        for i = 2:nx-1
            x = fix(i/s_x)+1; y = fix(j/s_y)+1;
            if x == 1 && y == 1
                p(ip(i,j)) = p_ini(ip_ini(2,y));
            else
                p(ip(i,j)) = p_ini(ip_ini(x,y));
            end
        end
    end
    for i = 2:nx-1
        x = fix(i/s_x)+1;
        p(ip(i,ny)) = p_ini(ip_ini(x,N));
    end
    for j = 2:ny-1
        y = fix(j/s_y)+1;
        p(ip(nx,j)) = p_ini(ip_ini(N,y));
    end
    p(ip(nx,ny)) = p_ini(ip_ini(N,N));
end
%% Discrete delta function
function out = d3(x,ds)  
    if abs(x) < 0.5*ds
        out = 1/3*(1+sqrt(-3*(x/ds)^2+1));
    elseif abs(x) >= 0.5*ds && abs(x) <= 1.5*ds
        out = 1/6*(5-3*abs(x)/ds-sqrt(-3*(1-abs(x)/ds)^2+1));
    else
        out = 0;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IBPM-related
%% Define boundary cylinder
function [xi,ds] = IBPM_cylinder(R,x_c,y_c,dx,dy)
    ds = min(dx,dy);
    nb = 2*pi*R/ds;
    xi = zeros(int16(nb),2);
    for k = 1:nb+1
        theta = 2*pi/nb*(k-1);
        xi(k,1) = x_c + R*cos(theta);
        xi(k,2) = y_c + R*sin(theta);
    end
    % scatter(xi(:,1),xi(:,2));
end
function [xi,ds] = IBPM_triangle(L,x_c,y_c,dx,dy)
    ds = min(dx,dy);
    nb = L/ds*3;
    xi = zeros(int16(nb),2);
    nbs = length(xi(:,1))/3;
    %top
    xi(1:nbs,2) = y_c+L/2/sqrt(3);
    xi(1:nbs,1) = linspace(x_c-L/2,x_c+L/2-ds,nbs);
    %right
    xi(nbs+1:2*nbs,1) = linspace(x_c+ds/2,x_c+L/2,nbs);
    xi(nbs+1:2*nbs,2) = linspace(y_c-L/sqrt(3)+ds/2*sqrt(3), ...
        y_c+L/2/sqrt(3),nbs);
    %left
    xi(2*nbs+1:3*nbs,1) = linspace(x_c-L/2+ds/2,x_c,nbs);
    xi(2*nbs+1:3*nbs,2) = linspace(y_c+L/2/sqrt(3)-ds/sqrt(3) ...
        ,y_c-L/sqrt(3),nbs);
    % scatter(xi(:,1),xi(:,2));
end
%% Q()
function out = Qtimes(g,H,np,nf,nx,ny,nq,ip,iu,iv,dx,dy)
    gp = g(1:np,1);
    gf = g(np+1:np+nf,1);
    out = -grad(gp,nx,ny,nq,ip,iu,iv,dx,dy) + H*gf;
end
%% QT()
function out = QTtimes(q,E,nx,ny,iu,iv,ip,np,dx,dy)
    out1 = div(q,nx,ny,iu,iv,ip,np,dx,dy);
    out2 = E*q;
    out = [out1;out2];
end
%% H and E
function [H,E] = regintp(nq,nb,nf,nx,ny,iu,iv,dx,dy,ds,xi,q_pos)
    H = zeros(nq,nf);
    for j = 1:ny
        for i = 2:nx
            for k = 1:nb
                H(iu(i,j),k) = d3(q_pos(iu(i,j),1)-xi(k,1),ds)*...
                    d3(q_pos(iu(i,j),2)+dy/2-xi(k,2),ds);
            end
        end 
    end
    for j = 2:ny
        for i = 1:nx
            for k = 1:nb
                H(iv(i,j),k+nb) = d3(q_pos(iv(i,j),2)-xi(k,2),ds)*...
                    d3(q_pos(iv(i,j),1)+dx/2-xi(k,1),ds);
            end
        end
    end
    H = sparse(H);
    E = H.'; % directly save the sparse matrices
end
        