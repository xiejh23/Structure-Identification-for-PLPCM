%
%
clear
clc
n=500; % sample size
SNR=5;
M=1;
alpha=0.5;
Q=binornd(1,alpha,5,5)*0.5;
p1=16;
p=5+5+p1;
v=0.5;
matrix1_ind=zeros(M,p);
matrix2_ind=zeros(M,p);
cont1=zeros(M,1);
cont2=zeros(M,1);
cont3=zeros(M,1);
for i=1:5 
    Q(i,i)=1;
end
for m=1:M

    I1=0;
    I2=0;
     for i=1:p 
        for j=1:p
             delta(i,j)=v^(abs(i-j));
        end
     end
     
    X=mvnrnd(zeros(p,1),delta,n);
%     XX1=mvrandn(-1.8*ones(1,p),1.8*ones(1,p),delta,n);
%     XX2=unifrnd(-2,2,n,p);
%     X=0.95*XX1'+0.05*XX2;

    
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
    X = [x1 x2 x3];
    lambda=log(p*n)/(n^0.5)*0.5;
    
    %%
    % step 1: zero irrelative variables out
    ind_rel=[];
    ind_irrel=[];
    ind_lin=[];
    ind_con=[];
    [beta0,yhat0,d,c,theta1,theta2] = cnlslasso_relevant_alterform_blockQPFn1(X,y,lambda);
    R1(m,:)=max(abs(theta1));
    R2(m,:)=max(abs(theta2));
    ind=1:p;
    t=1;
    for k=1:p 
        if R1(m,k)<=0.0001 && R2(m,k)<=0.0001
            ind_irrel(t)=k;
            t=t+1;
        end
    end
    ind_rel=setdiff(ind,ind_irrel);
    
    %%
    % step 2: classify linear and concave variables
    
    X_rel=X(:,ind_rel);
    pp=size(ind_rel,2);
    ind=1:pp;
    [beta1,yhat1,dd,cc] = cnlslasso_alterform_blockQPFn1(X_rel,y,lambda);
    dd2=dd(2:n-1,:);
    cc2=cc(2:n-1,:);    
    RR1(1,:)=max(abs(dd2));
    RR2(1,:)=max(abs(cc2));
    t=1;
    for k=1:pp 
        if RR1(1,k)<=0.0001 && RR2(1,k)<=0.0001
            ind_lin(t)=ind_rel(k);
            t=t+1;
        end
    end
    ind_con=setdiff(ind_rel,ind_lin);
    RR1=[];
    RR2=[];
    %%
    % step 3: post lasso
    X_lin=X(:,ind_lin);
    X_con=X(:,ind_con);
    num_con=size(ind_con,2);
    beta2=zeros(n,p);
    if num_con==0 
        beta2_lin=(X_lin'*X_lin)^(-1)*X_lin'*y;
        yhat2=X_lin*beta2_lin;
        beta2(:,ind_lin)=repmat(beta2_lin',n,1);
    else
        [beta2_con,beta2_lin,yhat2] = part_linFn2(X_con,X_lin,y);
        beta2(:,ind_con)=beta2_con;
        beta2(:,ind_lin)=repmat(beta2_lin',n,1);
    end
    
    %%
    % step 4 oracle
    X_con=x1;
    X_lin=x2;
    [beta3_con,beta3_lin,yhat3] = part_linFn2(X_con,X_lin,y);
    beta3=[beta3_con repmat(beta3_lin',n,1)];
    
    %%
    % performance measures
    % recovery rate
    matrix1_ind(m,ind_rel)=1;
    matrix2_ind(m,ind_con)=1;
    if sum(matrix1_ind(m,:))==10 && sum(matrix1_ind(m,1:10))==10
        cont1(m)=1;
        I1=1;
    end
    if sum(matrix2_ind(m,:))==5 && sum(matrix2_ind(m,1:5))==5
        cont2(m)=1;
        I2=1;
    end    
    if I1==1 && I2==1
        cont3(m)=1;
    end  
    
    %%
    % RMSE, MAD and ACC
    error1(:,m)=yhat1-y_true;
    error2(:,m)=yhat2-y_true;
    error3(:,m)=yhat3-y_true;
    
    rope1(m)=corr(yhat1,y_true);
    rope2(m)=corr(yhat2,y_true);
    rope3(m)=corr(yhat3,y_true);
    
    beta1_cell{m}=beta1;
    beta2_cell{m}=beta2;
    beta3_cell{m}=beta3;
    
end
%%
