function [beta1,yhat1,iter] = pllasso_relevantscreen_blockquadprogFn(X,y,lambda)
[n,p] = size(X);
Fmat = ones(p,1);
l = zeros(n,1);

for k=1:p 
    [XP(:,k),index(:,k)] = sort(X(:,k));
end
xx=X'*X;
xy=X'*y;

beta_ols=(xx)^(-1)*xy;
Fmat = ones(p,1);
Ind = (1>0);
beta1 = repmat(beta_ols',n,1);
for k=1:p 
    theta1(:,k) = beta1(:,k).*X(:,k);
end

iter = 0;
Max_Iter = 500;

while (Ind == (1>0)) 
    iter =iter +1;
    for k=1:p 
        Fmat(k,1) = 0;
        R_k = y - theta1*Fmat;
        H = (1/n)*[eye(n) zeros(n,n+1);zeros(n+1,2*n+1)];
        f = zeros(2*n+1,1);
        f(1:n)=-(1/n)*R_k;
        f(2*n+1)=lambda;
        Aeq = zeros(n,2*n+1);
        A = zeros(n-1+2*n,2*n+1);
        for i=1:n-1 
            Aeq(i,index(i+1,k)) = 1;
            Aeq(i,index(i,k)) = -1;
            Aeq(i,n+index(i,k)) = (XP(i,k) - XP(i+1,k));
            A(i,n+index(i+1,k)) = 1;
            A(i,n+index(i,k)) = -1;
        end
        A(n-1+1:n-1+n,:)=[zeros(n,n) -eye(n) -ones(n,1)];
        A(n-1+n+1:n-1+2*n,:)=[zeros(n,n) eye(n) -ones(n,1)];
        Aeq(n,:)=[ones(1,n) zeros(1,n+1)];
        beq = zeros(n,1);
        b = zeros(3*n-1,1);
        lb = [repmat(-inf,n*2,1)];
        ub = repmat(inf,n*2,1);
        [est,fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
        beta1(:,k) = est(n+1:n*2);
        theta1(:,k) = est(1:n);
        Fmat = ones(p,1);
        R2(k) = lambda*beta1(index(1,k),k);
        
    end
    obj(iter) = (0.5/n)*sum((y - theta1*ones(p,1)).^2) + sum(R2);
    if (iter > 3) 
        if (obj(iter-1) - obj(iter) < 0.0000001) 
            Ind = (1<0);
        end
    end
    if (iter > Max_Iter)
        Ind = (1<0);
    end  
end
yhat1 = theta1*ones(p,1);