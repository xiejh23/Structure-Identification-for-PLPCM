function [matrix1_ind,matrix2_ind,cont1,cont2,cont3,t1,t2]=procedure1_only_selection(X,y,lambda)
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
    tic;
    [beta0,yhat0,d,c,theta1,theta2] = cnlslasso_relevant_alterform_blockQPFn1(X,y,lambda);
    t1=toc
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
    tic;
    [beta1,yhat1,dd,cc] = cnlslasso_alterform_blockQPFn1(X_rel,y,lambda);
    t2=toc
    dd2=dd(2:n-1,:);
    cc2=cc(2:n-1,:);    
    RR1(1,:)=max(abs(dd2));
    RR2(1,:)=max(abs(cc2));
    t=1;
    for k=1:pp 
        if RR1(1,k)<=0.0001 && RR2(1,k)<=0.0001
            ind_lin(t)=k;
            t=t+1;
        end
    end
    ind_con=setdiff(ind,ind_lin);
    RR1=[];
    RR2=[];
   
    
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
    
  