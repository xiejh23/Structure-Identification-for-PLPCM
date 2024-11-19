
function [beta1,yhat1] = cnlsFn2(x,y)
[n,d]=size(x);
v1=[ones(n,1);zeros(n*d,1)];
H=sparse(1:n+n*d,1:n+n*d,v1);
f=[-y;zeros(n*d,1)];
a1=eye(n-1);
A1(1:n-1,:)=[-ones(n-1,1),a1];
A1((n-1)*(n-1)+1:n*(n-1),:)=[a1 -ones(n-1,1)];
for i=2:n-1 
    A1((i-1)*(n-1)+1:i*(n-1),:)=[a1(:,1:i-1) -ones(n-1,1) a1(:,i:n-1)];
end
for i=1:n 
    AA2=zeros(n-1,n*d);
    for k=1:d 
        a2=x(:,k);
        a2(i)=[];
        AA2(:,(k-1)*n+i)=[x(i,k)*ones(n-1,1)-a2];
    end
    A2((n-1)*(i-1)+1:(n-1)*i,:)=sparse(AA2);

end
A=[A1 A2];
b=zeros(n*(n-1),1);
lb=[];
ub=[];
[est] = quadprog(H,f,A,b,[],[],[],[]);
est1=reshape(est(1:n+n*d),n,d+1);
yhat1=est1(:,1);
beta_est=est1(:,2:d+1);
beta1=beta_est;
