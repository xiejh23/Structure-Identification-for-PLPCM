%
%
clear
clc
addpath('C:\Users\Administrator\Desktop\simulation_final\glmnet_matlab-master');
n=800; % sample size
SNR=5;
M=100;
alpha=1;
Q=binornd(1,alpha,5,5)*0.5;
p1=16;
p=5+5+p1;
v=0.8;
P1_matrix1_ind=zeros(M,p);
P1_matrix2_ind=zeros(M,p);
P2_matrix1_ind=zeros(M,p);
P2_matrix2_ind=zeros(M,p);

cont1=zeros(M,1);
cont2=zeros(M,1);
cont3=zeros(M,1);
for i=1:5 
    Q(i,i)=1;
end
for m=1:M

     for i=1:p 
        for j=1:p
             delta(i,j)=v^(abs(i-j));
        end
     end
     
    X=mvnrnd(zeros(p,1),delta,n);

    
    x1 = X(:,1:5);
    x2 = X(:,6:10);
    x3 = X(:,11:p);
    b1 = randn(7,5);
%     b = b1./(sum(b1.*b1,2).^0.5);
    c=5;
    for i=1:n 
        f1(i,1) = c*log(sum(exp(x1(i,:)*b1')));
    end
    
    f = -f1 + mean(f1) + x2*[1 1 1 1 1]';
    s1=sqrt(var(f))/SNR;
    u = normrnd(0,s1,[n,1]);
    y=f+u;
    ym=y-mean(y);
    y_true=f;
    ym_true=y_true-mean(y_true);
    X = [x1 x2 x3];
    lambda=log(p*n)/(n^0.5)*0.5;
    %%
    % step 1 2 3 of procedure 1
    [P1_matrix1_ind(m,:),P1_matrix2_ind(m,:),P1_error1(:,m),P1_error2(:,m),P1_error3(:,m),P1_rope1(m),P1_rope2(m),P1_rope3(m),P1_cont1(m),P1_cont2(m),P1_cont3(m)] = procedure1(X,y,lambda,x1,x2,y_true);
    
    %%
    % step 1 2 3 of procedure 2
    [P2_matrix1_ind(m,:),P2_matrix2_ind(m,:),P2_error1(:,m),P2_error2(:,m),P2_error3(:,m),P2_rope1(m),P2_rope2(m),P2_rope3(m),P2_cont1(m),P2_cont2(m),P2_cont3(m)] = procedure2(X,y,lambda,x1,x2,y_true);
    
end
%%
