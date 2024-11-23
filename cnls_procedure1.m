function [MSE_cnls1,vec_lin,vec_con] = cnls_procedure1(X,y,y_true)
    


    [n,p]=size(X);
    vec_lin=zeros(1,p);
    vec_con=zeros(1,p);
    %%
    % step 1: zero irrelative variables out

    lambda=log(p*n)/(n^0.5)*0.5;
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
    
    lambda=log(p*n)/(n^0.5)*0.5;
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
    
    vec_lin(ind_lin)=1;
    vec_con(ind_con)=1;
    
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
    
    MSE_cnls1=sum((y_true-yhat2).^2)/n;
