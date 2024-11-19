
function [beta1,alpha1,yhat1] = cnlsFn(X,y)
[n,d]=size(X);
v1=[ones(1,n) zeros(1,n*d)];
H=sparse(1:n+n*d,1:n+n*d,v1');
f=[-y;zeros(n*d,1)];
A=zeros(n*(n-1),n+n*d);
row=1;
for i=1:n 
    for j=1:n 
        if i~=j 
            A(row,i)=-1;
            A(row,j)=1;
            for k = 1:d
                A(row,i + n*k) = X(i,k)-X(j,k);
            end
            row=row+1;
        end
        
    end
end

b=zeros(n*(n-1),1);
lb=[];
ub=[];
[est] = quadprog(H,f,A,b,[],[],[],[]);
est1=reshape(est(1:n+n*d),n,d+1);
yhat1=est1(:,1);
beta_est=est1(:,2:d+1);
beta1=beta_est;
for i=1:n 
    alpha1(i,1)=yhat1(i)-X(i,:)*beta1(i,:)';
end

