clear;
clc;

data1=[ please copy the boston housing data here from https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data];

for i=1:13 
    data(:,i)=(data1(:,i)-mean(data1(:,i)))/(std(data1(:,i)));
%     data(:,i)=(data1(:,i)-min(data1(:,i)))/(max(data1(:,i))-min(data1(:,i)));
end
    
title1=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"];

MEDV=data1(:,14);

X=data(:,1:13);
y=MEDV;

[n,p]=size(X);
a=log(p*n)/(n^0.5)*0.5;
lambda_vec1=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 14.3 15 16]*a;
b=log(p*n)/(n^0.5)*0.5;
lambda_vec2=[0.3 0.4 0.5 0.6 0.8 1 1.2 1.4 2]*b;

[beta0_pre,alpha0_pre,yhat0_pre] = cnlsFn2(X,y);
sigma1 = std(yhat0_pre-y);

%%
% step 1: zero irrelevant covariates out
for m=1:size(lambda_vec1,2) 
    lambda1=lambda_vec1(1,m);
    ind_rel=[];
    ind_irrel=[];
    
    [beta0,yhat0,d,c,theta1,theta2] = cnlslasso_relevant_alterform_blockQPFn1(X,y,lambda1);
    R1(m,:)=max(abs(theta1));
    R2(m,:)=max(abs(theta2));
    ind=1:p;
    t=1;
    for k=1:p 
        if R1(m,k)<=0.00001 && R2(m,k)<=0.00001
            ind_irrel(t)=k;
            t=t+1;
        end
    end
    ind_rel=setdiff(ind,ind_irrel);
    X_rel=X(:,ind_rel);
    [beta0_refit,alpha0_refit,yhat0_refit] = cnlsFn2(X_rel,y);
    ind_rel_save{m}=ind_rel;
    pp=size(ind_rel,2);
    error=y-yhat0_refit;
    RSS=sum(error.^2);
    Cp(m)=RSS/(sigma1^2)+pp*n*2-n;

end
    

[Cp_min,J_C]=min(Cp);

ind_rel=ind_rel_save{J_C}; 
X_rel=X(:,ind_rel);
title_rel=title1(ind_rel);
pp=size(ind_rel,2);
%%
% step 2: classify into lienar and nonlinear sets
% 
[beta1_pre,alpha1_pre,yhat1_pre] = cnlsFn2(X_rel,y);
sigma2 = std(yhat1_pre-y);


for m=1:size(lambda_vec2,2)  
    ind_lin=[];
    ind_con=[];
    lambda2=lambda_vec2(1,m);
    [beta1,yhat1,dd,cc] = cnlslasso_alterform_blockQPFn1(X_rel,y,lambda2);
    dd2=dd(2:n-1,:);
    cc2=cc(2:n-1,:);    
    RR1(m,:)=max(abs(dd2));
    RR2(m,:)=max(abs(cc2));
    t=1;
    for k=1:pp 
        if RR1(m,k)<=0.00001 && RR2(m,k)<=0.00001
            ind_lin(t)=ind_rel(k);
            t=t+1;
        end
    end
    ind_con=setdiff(ind_rel,ind_lin);
    ind_con_save{m}=ind_con;
    ind_lin_save{m}=ind_lin;
    
    p_con=size(ind_con,2);
    p_lin=size(ind_lin,2);
    X_con=X(:,ind_con);
    X_lin=X(:,ind_lin);

    [alpha1_refit,beta1_refit,delta1_refit,yhat1_refit] = part_linFn2(X_con,X_lin,y);
    

    error2=y-yhat1_refit;
    RSS2=sum(error2.^2);
    Cp2(m)=RSS2/(sigma1^2)+(p_con*n+p_lin)*2-n;
    RSS2_save(m)=RSS2;


end
% 

[beta3,yhat3,dd3,cc3,theta1,theta2] = cnlslasso_alterform_blockQPFn1(X_con,y,0);

 ddd3=dd3(2:n-1,:);
 ccd3=cc3(2:n-1,:);    

[Cp2_min,J_C2]=min(Cp2);

ind_con=ind_con_save{J_C2}; 
ind_lin=ind_lin_save{J_C2};

X_con=X(:,ind_con);
X_lin=X(:,ind_lin);
t_cx=0;
t_cv=0;
for k=1:pp 
    if RR1(J_C2,k)>=0.00001 && RR2(J_C2,k)<=0.00001
        t_cv=t_cv+1;
        ind_concave(t_cv)=ind_rel(k);
    elseif RR1(J_C2,k)<=0.00001 && RR2(J_C2,k)>=0.00001
        t_cx=t_cx+1;
        ind_convex(t_cx)=ind_rel(k);
    end
end

subplot(2,2,1)
f_max=max(R1,R2);
lam_1=sum(f_max,2)/sum(f_max(1,:));
hLine = line(lam_1, f_max, 'LineWidth', 1.5);
hold on
plot([0.1612 0.1612], [0 30],'-k');
legend('CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','FontSize',6);
xlabel('normalized |f_k|_∞');
ylabel('|f_k|_∞');
title('Step 1 of Pro 1');

subplot(2,2,2)
d_max=max(RR1,RR2);
lam_2=sum(d_max,2)/sum(d_max(1,:));
hLine = line(lam_2, d_max, 'LineWidth', 1.5);
hold on
plot([0.6531 0.6531], [0 20],'-k');
legend(title1(ind_rel),'FontSize',6);
xlabel('normalized |d_k|_1');
ylabel('|d_k|_1');
title('Step 2 of Pro 1');



subplot(2,2,3)
% d3_max=max(RR1,RR2);
lam_3=sum(d3_max,2)/sum(d3_max(1,:));
hLine = line(lam_3, d3_max, 'LineWidth', 1.5);
hold on
legend(title1,'FontSize',6);
xlabel('normalized |d_k|_1');
ylabel('|d_k|_1');
title('Step 1 of Pro 2');

subplot(2,2,4)
% d4_max=abs(delta2_mat');
lam_4=sum(d4_max,2)/sum(d4_max(1,:));
hLine = line(lam_4, d4_max, 'LineWidth', 1.5);
hold on
legend(title1([1,2,3,4,5,7,8,9,10,11,12]),'FontSize',6);
xlabel('normalized |β|_1');
ylabel('|β|');
title('Step 2 of Pro 2');


subplot(2,3,1)
LSTAT=X(:,13);
plot(LSTAT,theta2(:,5),'.')
xlabel('LSTAT');
ylabel('MEDV');

% 
subplot(2,3,2)
RM=X(:,6);
plot(RM,theta2(:,3),'.')
xlabel('RM');
ylabel('MEDV');
% 
subplot(2,3,3)
TAX=X(:,10);
plot(TAX,theta2(:,4),'.')
xlabel('TAX');
ylabel('MEDV');
% 
% 
subplot(2,3,4)
CRIM=X(:,1);
plot(CRIM,theta1(:,1),'.')
xlabel('CRIM');
ylabel('MEDV');
% 
subplot(2,3,5)
NOX=X(:,5);
plot(NOX,theta1(:,2),'.')
xlabel('NOX');
ylabel('MEDV');




