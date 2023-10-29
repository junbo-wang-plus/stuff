clear all; close all;

epsilon = 0.06; % Sonic point correction coeff
crit = 1e-5; % Convergence criteria
N = 20000; % Maximum steps
gamma = 1.4;
R = 287;
nu = 1;
M1 = [1.35 1.8 2.7 4.0 6.0];
interval = 1;
% M1 = 1.35;

% Preshock conditions
P1 = 101325; % 1atm static pressure
T1 = 300; % 300K
rho1 = 1.2; % 1.2kg/m^3

standoffDISTx = [-1.8 -1.2 -1 -0.9 -0.8]; % estimates from experiment
standoffDISTy = [6 2.5 2 1.5 1.2]; % estimates from experiment
% standoffDIST = -3.2;

for kk = 3 % looping different grids
    %%% Loading grid parameters and declaring variables
    load("grid_"+kk+".mat")
    load("gridparams_"+kk+".mat")
    U = zeros(IL+1,JL+1,4);
    U_int = zeros(N/interval,IL+1,JL+1,4);

    for ll = 1:5 % looping different Mach numbers

        %% Initialization
        [P2, T2, rho2, M2, u2] = NormalShock(gamma, M1(ll), P1, T1, rho1);
        c1 = sqrt(gamma*P1*rho1);
        u1 = M1(ll)*c1;
        e1 = rho1*u1^2/2+P1/(gamma-1);
        e2 = rho2*u2^2/2+P2/(gamma-1);
        for i = 1:IL+1
            for j = 1:JL+1
%                 U(i,j,1) = rho2;
%                 U(i,j,2) = rho2*u2;
%                 U(i,j,3) = 0;
%                 U(i,j,4) = rho2*u2^2/2+P2/(gamma-1);
                if y_FV(i,j) < -standoffDISTy(ll)/standoffDISTx(ll)*x_FV(i,j)+standoffDISTy(ll)
                    U(i,j,1) = rho2;
                    U(i,j,2) = rho2*u2;
                    U(i,j,3) = 0;
                    U(i,j,4) = e2;
                else
                    U(i,j,1) = rho1;
                    U(i,j,2) = rho1*u1;
                    U(i,j,3) = 0;
                    U(i,j,4) = e1;      
                end
            end
            U(i,1,1) = rho2;
            U(i,1,2) = rho2*u2*cos(2*atan(dy(i,2,4)/dx(i,2,4)));
            U(i,1,3) = rho2*u2*sin(2*atan(dy(i,2,4)/dx(i,2,4)));
            U(i,1,4) = (U(i,1,2)^2+U(i,1,3)^2)/2/rho2+P2/(gamma-1);
        end
        V = U2V(U,IL,JL,gamma);
        c = V2C(V,IL,JL,gamma);
        if isreal(c) == 0
                disp("imaginary c during initialization")
                break;
        end
        if isreal(V) == 0
                disp("imaginary V during initialization")
                break;
        end
        
        V_ini = V;
        %% Solving
        res = 1;
        T = 0;
        step = 0;
        stored = 0;    
%         figure(10)
        while res > crit
            if step >= N
                disp("Not converged");
                break
            end
%             delta_t = 0.000005;
            delta_t = getdt(V,c,DX,DY,IL,JL,nu);
           
            T = T+delta_t;
            U_temp = U;
            U = SW(U,c,S,nx,ny,Vol,delta_t,IL,JL,gamma,epsilon);
%             U = ROE(U,S,nx,ny,Vol,delta_t,IL,JL,gamma,epsilon);
          

            U = BC(U,U_temp,c,delta_t,dx,dy,x_FV,IL,JL,gamma,rho1,u1,e1);
          
            V = U2V(U,IL,JL,gamma);
            
            rho_temp = U(:,:,1);
            
            u_temp = V(:,:,2);
            v_temp = V(:,:,3);
            p_temp = V(:,:,4);
%             if isreal(p_temp) == 0
%                 disp("imaginary pressure")
%                 break
%             end
            c = V2C(V,IL,JL,gamma);
            if isreal(c) == 0
                disp("imaginary c")
                break
            end
            res = residual(U,U_temp,IL,JL,gamma);
            step = step + 1;
            res_his(step) = res;
%             quiver(x_FV,y_FV,V(:,:,2),V(:,:,3),1.5)
%             frame = getframe(10);
%             im = frame2im(frame);
%             [imind,cm] = rgb2ind(im,256);
%             
        end
%         contourf(x_FV,y_FV,V(:,:,4),10)
%         save("ROE_grid_"+kk+"_M_"+M1(ll)+"_nu_"+nu+"_eps_"+epsilon+".mat","U","V","res_his","step")
        save("SW_grid_"+kk+"_M_"+M1(ll)+"_nu_"+nu+"_eps_"+epsilon+".mat","U","V","res_his","step")   
        disp("Grid_"+kk+"_Mach_"+M1(ll)+",done after "+step+" steps")
    end
end

function [P2, T2, rho2, M2, u2] = NormalShock(gamma, M, P1, T1, rho1)
    M2 = sqrt(((gamma-1)*M^2+2)/(2*gamma*M^2-gamma+1));
    P2 = P1*((2*gamma*M^2/(gamma+1))-((gamma-1)/(gamma+1)));
    rho2 = rho1*(((gamma+1)*M^2)/((gamma-1)*M^2+2));
    T2 = T1*((1+(gamma-1)/2*M^2)*((2*gamma*M^2/(gamma-1)-1))/(M^2*(2*gamma/(gamma-1)+(gamma-1)/2)));
    c2 = sqrt(gamma*P2/rho2);
    u2 = M2*c2;
end

function delta_t = getdt(V,c,DX,DY,IL,JL,nu)
    DT = zeros(IL-1,JL-1);
    for i = 2:IL
        for j = 2:JL
            DT(i-1,j-1) = nu/(abs(V(i,j,2))/DX(i,j)+abs(V(i,j,3))/DY(i,j)+c(i,j)*sqrt(1/DX(i,j)^2+1/DY(i,j)^2));
        end
    end
    delta_t = min(min(DT));
end

function V = U2V(U,IL,JL,gamma)
    V = zeros(IL+1,JL+1,4);
    for i = 1:IL+1
        for j = 1:JL+1
            rho = U(i,j,1);
            u = U(i,j,2)/rho;
            v = U(i,j,3)/rho;
            e = U(i,j,4);
            V(i,j,1) = rho;
            V(i,j,2) = u;
            V(i,j,3) = v;
            V(i,j,4) = (e-rho/2*(u^2+v^2))*(gamma-1);
        end
    end
end

function c = V2C(V,IL,JL,gamma)
    c = zeros(IL+1,JL+1);
    for i = 1:IL+1
        for j = 1:JL+1
            p = V(i,j,4);
            rho = V(i,j,1);
            c(i,j) = sqrt(gamma*p/rho);
        end
    end
end

function U_new = SW(U,c,S,nx,ny,Vol,dt,IL,JL,gamma,eps)
    U_new = zeros(IL+1,JL+1,4);
    U_tempP = zeros(4,1); U_tempM = zeros(4,1);
    U_temp = zeros(4,1);
    term = zeros(4,4);
    term1 = zeros(4,1); term2 = zeros(4,1); term3 = zeros(4,1); term4 = zeros(4,1);
    beta = gamma-1; 
    %%% Solving interior
    for i = 2:IL
        for j = 2:JL
            U_temp(:) = U(i,j,:);
            for k = 1:4
                if k == 1
                    im = i+1; ip = i;           
                    jm = j; jp = j;                
                elseif k == 2
                    im = i; ip = i;
                    jm = j+1; jp = j;
                elseif k == 3
                    im = i; ip = i-1;
                    jm = j; jp = j;
                else
                    im = i; ip = i;
                    jm = j; jp = j-1;
                end
                U_tempP(:) = U(ip,jp,:); U_tempM(:) = U(im,jm,:);
                uP = U(ip,jp,2)/U(ip,jp,1); vP = U(ip,jp,3)/U(ip,jp,1);
                uM = U(im,jm,2)/U(im,jm,1); vM = U(im,jm,3)/U(im,jm,1);

                upP = uP*nx(i,j,k)+vP*ny(i,j,k); vpP = -uP*ny(i,j,k)+vP*nx(i,j,k);           
                upM = uM*nx(i,j,k)+vM*ny(i,j,k); vpM = -uM*ny(i,j,k)+vM*nx(i,j,k);    
                alphaP = 1/2*(uP^2+vP^2); alphaM = 1/2*(uM^2+vM^2);

                LambpP = SW_LambdaP(upP,c(ip,jp),eps); LambpM = SW_LambdaM(upM,c(im,jm),eps);  

                PiP = getPi(alphaP,beta,c(ip,jp),U(ip,jp,1),uP,upP,vP,vpP,nx(i,j,k),ny(i,j,k));
                PiM = getPi(alphaM,beta,c(im,jm),U(im,jm,1),uM,upM,vM,vpM,nx(i,j,k),ny(i,j,k));
                PP = getP(alphaP,beta,c(ip,jp),U(ip,jp,1),uP,upP,vP,vpP,nx(i,j,k),ny(i,j,k));
                PM = getP(alphaM,beta,c(im,jm),U(im,jm,1),uM,upM,vM,vpM,nx(i,j,k),ny(i,j,k));
                
                WP = PiP*U_tempP; WM = PiM*U_tempM; 
                EppP_temp = PP*(diag(LambpP).*WP);
                EppM_temp = PM*(diag(LambpM).*WM);
%                 ApP = PP*LambpP*PiP; ApM = PM*LambpM*PiM;
%                 EppP_temp = ApP*U_tempP; EppM_temp = ApM*U_tempM;
                term_temp = (EppP_temp+EppM_temp)*S(i,j,k);
                term(k,:) = term_temp(:);
            end
            term1(:) = term(1,:); term2(:) = term(2,:); 
            term3(:) = term(3,:); term4(:) = term(4,:); 
            U_new_temp = U_temp-dt/Vol(i,j)*(term1-term3+term2-term4);
            U_new(i,j,:) = U_new_temp(:);
        end
    end  
end

function U_new = ROE(U,S,nx,ny,Vol,dt,IL,JL,gamma,eps)
    U_new = zeros(IL+1,JL+1,4);
    U_tempP = zeros(4,1); U_tempM = zeros(4,1);
    U_temp = zeros(4,1);
    beta = gamma-1;
    term = zeros(4,4);
    term1 = zeros(4,1); term2 = zeros(4,1); term3 = zeros(4,1); term4 = zeros(4,1);
    for i = 2:IL
        for j = 2:JL
            U_temp(:) = U(i,j,:);
            for k = 1:4
                if k == 1
                    im = i; jm = j;
                    ip = i+1; jp = j;
                elseif k == 2
                    im = i; jm = j;
                    ip = i; jp = j+1;
                elseif k == 3
                    im = i-1; jm = j;
                    ip = i; jp = j;
                else
                    im = i; jm = j-1;
                    ip = i; jp = j;
                end
                U_tempP(:) = U(ip,jp,:); U_tempM(:) = U(im,jm,:);
                [rhoB,uB,vB,cB] = ROEavg(U,im,jm,ip,jp,gamma);
                alpha = 1/2*(uB^2+vB^2);
                upB = uB*nx(i,j,k)+vB*ny(i,j,k); vpB = -uB*ny(i,j,k)+vB*nx(i,j,k);

                LambB = ROE_Lambda(upB,cB,eps); 
                PiB = getPi(alpha,beta,cB,rhoB,uB,upB,vB,vpB,nx(i,j,k),ny(i,j,k));
                PB = getP(alpha,beta,cB,rhoB,uB,upB,vB,vpB,nx(i,j,k),ny(i,j,k));
                AB = PB*LambB*PiB;
                EM = ROE_U2E(U_tempM,nx(i,j,k),ny(i,j,k),gamma); 
                EP = ROE_U2E(U_tempP,nx(i,j,k),ny(i,j,k),gamma); 
                EB = 1/2*(EM+EP)-1/2*AB*(U_tempP-U_tempM);
                term_temp = EB*S(i,j,k);
                term(k,:) = term_temp(:);
            end
            term1(:) = term(1,:); term2(:) = term(2,:); 
            term3(:) = term(3,:); term4(:) = term(4,:); 
            U_new_temp = U_temp-dt/Vol(i,j)*(term1-term3+term2-term4);
            U_new(i,j,:) = U_new_temp(:);
        end
    end
end

function U_new  = BC(U,U_temp,c,dt,dx,dy,x_FV,IL,JL,gamma,rho_pre,u_pre,e_pre)
    % Copy interior
    U_new = zeros(IL+1,JL+1,4);
    for i = 2:IL
        for j = 2:JL
            U_new(i,j,:) = U(i,j,:);
        end
    end
    % Solving boundaries
    for i = 2:IL
        %wall
        U_new(i,1,1) = U_new(i,2,1);
        u1 = U_new(i,2,2)/U_new(i,1,1);
        v1 = U_new(i,2,3)/U_new(i,1,1);
        U_new(i,1,4) = U_new(i,2,4);
        u0 = cos(2*atan(dy(i,2,4)/dx(i,2,4)))*u1+sin(2*atan(dy(i,2,4)/dx(i,2,4)))*v1;
        v0 = sin(2*atan(dy(i,2,4)/dx(i,2,4)))*u1-cos(2*atan(dy(i,2,4)/dx(i,2,4)))*v1;
        U_new(i,1,2) = U_new(i,1,1)*u0;
        U_new(i,1,3) = U_new(i,1,1)*v0;
        %left/top,inlet
        U_new(i,JL+1,1) = rho_pre;
        U_new(i,JL+1,2) = rho_pre*u_pre;
        U_new(i,JL+1,3) = 0;
        U_new(i,JL+1,4) = e_pre;
    end
    for j = 2:JL %bottom
%         U_new(1,j,:) = U(2,j,:); %copy
        U_new(1,j,1) = U_new(2,j,1);
        U_new(1,j,2) = U_new(2,j,2);
        U_new(1,j,3) = -U_new(2,j,3);
        U_new(1,j,4) = U_new(2,j,4);
    end
    for j = 2:JL %subsonic exit
%         rho = U_temp(IL+1,j,1); rhoM = U_temp(IL,j,1); 
%         rhoM_new = U(IL,j,1);
% 
%         u = U_temp(IL+1,j,2)/rho; uM = U_temp(IL,j,2)/rhoM;
%         uM_new = U(IL,j,2)/rhoM_new;
% 
%         v = U_temp(IL+1,j,3)/rho; vM = U_temp(IL,j,3)/rhoM;
%         vM_new = U(IL,j,3)/rhoM_new;
% 
%         e = U_temp(IL+1,j,4); eM = U_temp(IL,j,4);
%         eM_new = U(IL,j,4);
%         
%         p = (e-rho/2*(u^2+v^2))*(gamma-1);
%         pM = (eM-rhoM/2*(uM^2+vM^2))*(gamma-1);  
%         pM_new = (eM_new-rhoM_new/2*(uM_new^2+vM_new^2))*(gamma-1);
%         
%         dx = x_FV(IL+1,j)-x_FV(IL,j);
%         
%         %copy interior
%         p_new = pM_new;
%         v_new = vM_new;
%         %characteristics relations
%         rho_new = rho+(p_new-p)/c(IL+1,j)^2-u*dt*((rho-rhoM)/dx-1/c(IL+1,j)^2/dx*(p-pM));
%         u_new = u-(p-p_new)/rho/c(IL+1,j)-(u+c(IL+1,j))*dt/rho/c(IL+1,j)*(p-pM)/dx- ...
%             (u+c(IL+1,j))*dt/dx*(u-uM);
% 
%         e_new = p_new/(gamma-1)+rho_new/2*(u_new^2+v_new^2);
%         U_new(IL+1,j,1) = rho_new;
%         U_new(IL+1,j,2) = rho_new*u_new;
%         U_new(IL+1,j,3) = rho_new*v_new;
%         U_new(IL+1,j,4) = e_new;

        %supersonic exit
        U_new(IL+1,j,:) = U_new(IL,j,:);
    end
end

function [rhoB,uB,vB,cB] = ROEavg(U,im,jm,ip,jp,gamma)
    rhoM = U(im,jm,1); uM = U(im,jm,2)/U(im,jm,1);
    vM = U(im,jm,3)/U(im,jm,1); eM = U(im,jm,4);
    rhoP = U(ip,jp,1); uP = U(ip,jp,2)/U(ip,jp,1);
    vP = U(ip,jp,3)/U(ip,jp,1); eP = U(ip,jp,4);

    pP = (eP-rhoP/2*(uP^2+vP^2))*(gamma-1);
    pM = (eM-rhoM/2*(uM^2+vM^2))*(gamma-1);
    hP = gamma*pP/(gamma-1)/rhoP+(uP^2+vP^2)/2;
    hM = gamma*pM/(gamma-1)/rhoM+(uM^2+vM^2)/2;
    
    rhoB = sqrt(rhoP*rhoM);
    uB = (sqrt(rhoM)*uM+sqrt(rhoP)*uP)/(sqrt(rhoM)+sqrt(rhoP));
    vB = (sqrt(rhoM)*vM+sqrt(rhoP)*vP)/(sqrt(rhoM)+sqrt(rhoP));
    hB = (sqrt(rhoM)*hM+sqrt(rhoP)*hP)/(sqrt(rhoM)+sqrt(rhoP));
    cB = sqrt((gamma-1)*(hB-(uB^2+vB^2)/2));
end

function E = ROE_U2E(U,nx,ny,gamma)
    E = zeros(4,1);
    rho = U(1);
    u = U(2)/rho;
    v = U(3)/rho;
    e = U(4);
    p = (e-rho/2*(u^2+v^2))*(gamma-1);
    up = u*nx+v*ny;
    E(1) = rho*up;
    E(2) = rho*u*up+p*nx;
    E(3) = rho*v*up+p*ny;
    E(4) = (e+p)*up;
end

function LambpP = SW_LambdaP(up,c,eps)
    EPS = eps*(abs(up)+c);
    LambpP = zeros(4,4);
    LambpP(1,1) = 1/2*(up+sqrt(up^2+EPS^2));
    LambpP(2,2) = 1/2*(up+c+sqrt((up+c)^2+EPS^2));
    LambpP(3,3) = 1/2*(up+sqrt(up^2+EPS^2));
    LambpP(4,4) = 1/2*(up-c+sqrt((up-c)^2+EPS^2));
end

function LambpM = SW_LambdaM(up,c,eps)
    EPS = eps*(abs(up)+c);
    LambpM = zeros(4,4);
    LambpM(1,1) = 1/2*(up-sqrt(up^2+EPS^2));
    LambpM(2,2) = 1/2*(up+c-sqrt((up+c)^2+EPS^2));
    LambpM(3,3) = 1/2*(up-sqrt(up^2+EPS^2));
    LambpM(4,4) = 1/2*(up-c-sqrt((up-c)^2+EPS^2));
end

function Lamb = ROE_Lambda(up,c,eps)
    EPS = eps*(abs(up)+c);
    Lamb = zeros(4,4);
    if EPS == 0
%         disp("No damping")
        Lamb(1,1) = abs(up);
        Lamb(2,2) = abs(up+c);
        Lamb(3,3) = abs(up);
        Lamb(4,4) = abs(up-c);
    else
        if abs(up) >= 2*EPS
            Lamb(1,1) = abs(up);
            Lamb(3,3) = abs(up);
        else
%             disp("Damped for 1,1 3,3")
            Lamb(1,1) = (up^2+4*EPS^2)/4/EPS;
            Lamb(3,3) = (up^2+4*EPS^2)/4/EPS;
        end
        if abs(up+c) >= 2*EPS
            Lamb(2,2) = abs(up+c);
        else
%             disp("Damped for 2,2")
            Lamb(2,2) = ((up+c)^2+4*EPS^2)/4/EPS;
        end
        if abs(up-c) >= 2*EPS
            Lamb(4,4) = abs(up-c);
        else
%             disp("Damped for 4,4")
            Lamb(4,4) = ((up-c)^2+4*EPS^2)/4/EPS;
        end
    end
end


function out = getPi(alpha,beta,c,rho,u,up,v,vp,kx,ky)
    out(1,1) = 1-alpha*beta/c^2;
    out(1,2) = u*beta/c^2;
    out(1,3) = v*beta/c^2;
    out(1,4) = -beta/c^2;
    out(2,1) = alpha*beta-up*c;
    out(2,2) = -u*beta+kx*c;
    out(2,3) = -v*beta+ky*c;
    out(2,4) = beta;
    out(3,1) = -vp/rho;
    out(3,2) = -ky/rho;
    out(3,3) = kx/rho;
    out(3,4) = 0;
    out(4,1) = alpha*beta+up*c;
    out(4,2) = -u*beta-kx*c;
    out(4,3) = -v*beta-ky*c;
    out(4,4) = beta;
end

function out = getP(alpha,beta,c,rho,u,up,v,vp,kx,ky)
    out(1,1) = 1;
    out(1,2) = 1/2/c^2;
    out(1,3) = 0;
    out(1,4) = 1/2/c^2;
    out(2,1) = u;
    out(2,2) = u/2/c^2+kx/2/c;
    out(2,3) = -ky*rho;
    out(2,4) = u/2/c^2-kx/2/c;
    out(3,1) = v;
    out(3,2) = v/2/c^2+ky/2/c;
    out(3,3) = kx*rho;
    out(3,4) = v/2/c^2-ky/2/c;
    out(4,1) = alpha;
    out(4,2) = alpha/2/c^2+up/2/c+1/2/beta;
    out(4,3) = rho*vp;
    out(4,4) = alpha/2/c^2-up/2/c+1/2/beta;
end

function out = residual(U,U_temp,IL,JL,gamma)
    res = zeros(IL,JL);
    for i = 2:IL
        for j = 2:JL
            rho = U(i,j,1); rho_old = U_temp(i,j,1);
            u = U(i,j,2)/rho; u_old = U_temp(i,j,2)/rho;
            v = U(i,j,3)/rho; v_old = U_temp(i,j,3)/rho;
            e = U(i,j,4); e_old = U_temp(i,j,4);
            p = (e-rho/2*(u^2+v^2))*(gamma-1);
            p_old = (e_old-rho_old/2*(u_old^2+v_old^2))*(gamma-1);
            res(i,j) = abs((p-p_old)/p_old);
        end
    end
    out = max(max(res));
end

