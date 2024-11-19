%%
%find the relevant variables for partially linear model 
function [beta1,delta1,yhat1] = lasso_part_linFn3(x,z,y,lambda)

X=[x z];
[n,p] = size(x);
r = size(z,2);

Fmat = ones(p,1);
l = zeros(n,1);

for k=1:p 
    [XP(:,k),index(:,k)] = sort(x(:,k));
end
xx=X'*X;
xy=X'*y;
beta_ols=(xx)^(-1)*xy;
Fmat = ones(p,1);
Ind = (1>0);
beta1 = repmat(beta_ols(1:p)',n,1);
delta1 = beta_ols(p+1:p+r);

for k=1:p 
    theta1(:,k) = beta1(:,k).*x(:,k);
end

iter = 0;
Max_Iter = 500;

while (Ind == (1>0)) 
    iter =iter + 1; 
    R_k1=y - z*delta1;
    [beta1,R_k1_hat] = cnlsFn2(x,R_k1);
    R_k2 = y - R_k1_hat;
    delta1 = lasso(z,R_k2,'Lambda',lambda);
    obj(iter) = (0.5/n)*sum((y - R_k1_hat - z*delta1).^2) + lambda*sum(abs(delta1));
    if (iter > 3) 
        if (obj(iter-1) - obj(iter) < 0.0000001) 
            Ind = (1<0);
        end
    end
    if (iter > Max_Iter)
        Ind = (1<0);
    end  
end
yhat1 = R_k1_hat + z*delta1;