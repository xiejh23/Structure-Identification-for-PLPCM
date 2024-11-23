function [MSE_cnls2,vec_lin,vec_con] = cnls_procedure2(X,y,y_true)
    

    [n,p]=size(X);
    vec_lin=zeros(1,p);
    vec_con=zeros(1,p);
        %%
    % step 1: classify linear and concave variables

    lambda=log(p*n)/(n^0.5)*0.5;
    ind=1:p;
    [beta1,yhat1,dd,cc] = cnlslasso_alterform_blockQPFn1(X,y,lambda);
    dd2=dd(2:n-1,:);
    cc2=cc(2:n-1,:);    
    RR1(1,:)=max(abs(dd2));
    RR2(1,:)=max(abs(cc2));
    t=1;
    for k=1:p 
        if RR1(1,k)<=0.0001 && RR2(1,k)<=0.0001
            ind_lin(t)=k;
            t=t+1;
        end
    end
    ind_con=setdiff(ind,ind_lin);
    RR1=[];
    RR2=[];
    
        %%
    % step 2: zero irrelative variables out
    lambda=log(p*n)/(n^0.5)*0.5;
    X_nonlin=X(:,ind_con);
    X_lin=X(:,ind_lin);

    [beta1_nonlin,beta1_lin,yhat1] = lasso_part_linFn3(X_nonlin,X_lin,y,lambda);
    ind_zero=find(beta1_lin==0);
    ind_irrel=ind_lin(ind_zero);
    ind_rel=setdiff(ind,ind_irrel);
    ind_lin=setdiff(ind_lin,ind_irrel);
    
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
    
    MSE_cnls2=sum((y_true-yhat2).^2)/n

