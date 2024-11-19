%
%
clear
clc
% addpath('C:\Users\Administrator\Desktop\simulation_final\glmnet_matlab-master');
n=800; % sample size
SNR=5;
M=10;
alpha=0.5;
Q=binornd(1,alpha,5,5)*0.5;
p1=128;

p=5+5+p1;
v=0.5;
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
    
    for i=1:n 
        f1(i,1)=x1(i,:)*Q*x1(i,:)';
    end
    
    f = -f1+ x2*[1 1 1 1 1]';
    s1=sqrt(var(f))/SNR;
    u = normrnd(0,s1,[n,1]);
    y=f+u;
    ym=y-mean(y);
    y_true=f;
    ym_true=y_true-mean(y_true);
    lambda=log(p*n)/(n^0.5)*0.5;
    %%
    % step 1 2 3 of procedure 1
    [matrix1_ind,matrix2_ind,cont1,cont2,cont3,t1(m),t2(m)]=procedure1_only_selection(X,y,lambda);
%     tic;
%     [matrix1_ind,matrix2_ind,error1,error2,error3,rope1,rope2,rope3,cont1,cont2,cont3]=procedure1(X,y,lambda,x1,x2,y_true);
%     t1(m)=toc;
    %%
    % step 1 2 3 of procedure 2
    [matrix1_ind,matrix2_ind,cont1,cont2,cont3,t3(m),t4(m)]=procedure2_only_selection(X,y,lambda);    
%      tic;
%      [matrix1_ind,matrix2_ind,error1,error2,error3,rope1,rope2,rope3,cont1,cont2,cont3]=procedure2(X,y,lambda,x1,x2,y_true);
%      t2(m)=toc;


%     tic;
%     [beta1,delta1,yhat1] = part_linFn2(x1,x2,y);
%     t1=toc;
%     
%     tic;
%     [beta2,yhat2] = cnlsFn2([x1 x2],y);
%     t2=toc;
%     
%     tic;
%     [beta3,yhat3] = cnlsFn2(X,y);
%     t3(m)=toc

     
end

time=[mean(t1) mean(t2) mean(t3) mean(t4)];