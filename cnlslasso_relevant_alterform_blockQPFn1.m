function [beta2,yhat1,d,c,theta1,theta2] = cnlslasso_relevant_alterform_blockQPFn1(X,y,lambda)
%%
%construct delta matrix
[n,p] = size(X);

for k=1:p 
    [XP(:,k),index(:,k)] = sort(X(:,k));
    delta{k}=max(repmat(X(:,k),1,n-1)-repmat(XP(1:n-1,k)',n,1),0);
    delta_hat{k}=delta{k}-(1/n)*ones(n,1)*ones(n,1)'*delta{k};
end

%%
%solve the block QP
xx=X'*X;
xy=X'*y;
beta_ols=(xx)^(-1)*xy;
d_initial(1,:)=beta_ols';
d_initial(2:n-1,:)=zeros(n-2,p);
d=d_initial;
iter = 0;
Max_Iter = 500;
Ind = (1>0);
theta1=zeros(n,p);
Fmat = ones(p,1);

while (Ind == (1>0)) 
    iter =iter +1;
    for k=1:p 
        delta_k=delta_hat{k};
        Fmat(k,1) = 0;
        R_k = y - theta1*Fmat;
        H = (1/n)*[eye(n) zeros(n,n);zeros(n,2*n)];
        f = zeros(2*n,1);
        f(1:n)=-(1/n)*R_k;
        f(2*n)=lambda;
        Aeq=[eye(n) -delta_k zeros(n,1)];
        beq=zeros(n,1);
        A=[eye(n) zeros(n,n-1) -ones(n,1); -eye(n) zeros(n,n-1) -ones(n,1)];
        b=zeros(2*n,1);
        lb = [repmat(-inf,n*2-1,1);0];
        ub = [repmat(inf,n+1,1);zeros(n-2,1);inf];
        [est,fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
        theta1(:,k) = est(1:n);
        d(:,k) = est(n+1:n*2-1);
        gamga=est(n*2);
        Fmat = ones(p,1);
        R2(k)= lambda*gamga;
    end
    obj(iter) = (0.5/n)*sum((y - theta1*ones(p,1)).^2) + sum(R2);
    if (iter > 3) 
        if (obj(iter-1) - obj(iter) < 0.0000000001) 
            Ind = (1<0);
        end
    end
    if (iter > Max_Iter)
        Ind = (1<0);
    end      
end
yhat1 = theta1*ones(p,1);
beta1(1,:)=d(1,:);
for i=2:n-1
beta1(i,:)=d(1,:)+sum(d(2:i,:),1);
end
beta1(n,:)=beta1(n-1,:);
for k=1:p 
    beta2(:,k)=beta1(index(:,k),k);
end

%%
%DC stage

Fmat = ones(p,1);
for k=1:p 
    delta_k=delta_hat{k};
    Fmat(k,1)=0;
    r_k=y-theta1*Fmat;
    H = (1/n)*[eye(n) zeros(n,n);zeros(n,2*n)];
    f = zeros(2*n,1);
    f(1:n)=-(1/n)*r_k;
    f(n*2)=lambda;
    Aeq=[eye(n) -delta_k zeros(n,1)];
    A=[eye(n) zeros(n,n-1) -ones(n,1); -eye(n) zeros(n,n-1) -ones(n,1)];
    b=zeros(2*n,1);    
    beq=zeros(n,1);
    lb = [repmat(-inf,n+1,1);zeros(n-1,1)];
    ub = [repmat(inf,2*n,1)];
    [est,fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    theta2(:,k) = est(1:n);
    c(:,k) = est(n+1:n*2-1);
    Fmat = ones(p,1);
    
end


