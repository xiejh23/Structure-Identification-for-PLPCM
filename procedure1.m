function [matrix1_ind,matrix2_ind,error1,error2,error3,rope1,rope2,rope3,cont1,cont2,cont3]=procedure1(X,y,lambda,x1,x2,y_true)
    [n,p]=size(X);
    matrix1_ind=zeros(1,p);
    matrix2_ind=zeros(1,p);
    cont1=0;
    cont2=0;
    cont3=0;

    %%
    % step 1: zero irrelative variables out
    ind_rel=[];
    ind_irrel=[];
    ind_lin=[];
    ind_con=[];
    [beta0,yhat0,d,c,theta1,theta2] = cnlslasso_relevant_alterform_blockQPFn1(X,y,lambda);
    R1(1,:)=max(abs(theta1));
    R2(1,:)=max(abs(theta2));
    ind=1:p;
    t=1;
    for k=1:p 
        if R1(1,k)<=0.0001 && R2(1,k)<=0.0001
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
    I1=0;
    I2=0;
    matrix1_ind(1,ind_rel)=1;
    matrix2_ind(1,ind_con)=1;
    if sum(matrix1_ind(1,:))==10 && sum(matrix1_ind(1,1:10))==10
        cont1=1;
        I1=1;
    end
    if sum(matrix2_ind(1,:))==5 && sum(matrix2_ind(1,1:5))==5
        cont2=1;
        I2=1;
    end    
    if I1==1 && I2==1
        cont3=1;
    end  
    
    %%
    % RMSE, MAD and ACC
    error1(:,1)=yhat1-y_true;
    error2(:,1)=yhat2-y_true;
    error3(:,1)=yhat3-y_true;
    
    rope1=corr(yhat1,y_true);
    rope2=corr(yhat2,y_true);
    rope3=corr(yhat3,y_true);
    



