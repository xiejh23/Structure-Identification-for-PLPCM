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
% for the final table
matrix1_ind=P1_matrix1_ind;
matrix2_ind=P1_matrix2_ind;
error1=P1_error1;
error2=P1_error2;
error3=P1_error3;
rope1=P1_rope1;
rope2=P1_rope2;
rope3=P1_rope3;
cont1=P1_cont1;
cont2=P1_cont2;
cont3=P1_cont3;

num_rel=sum(matrix1_ind,2);      % number of relevant variables
num_con=sum(matrix2_ind,2);      % number of concave variables
num_lin=num_rel-num_con;         % number of linear variables
num_rel2=sum(matrix1_ind(:,1:10),2);      % number of relevant variables
num_con2=sum(matrix2_ind(:,1:5),2);      % number of concave variables
num_lin2=sum(matrix1_ind(:,6:10),2)-sum(matrix2_ind(:,6:10),2);         % number of linear variables
num_irr=p1-sum(matrix1_ind(:,11:p1+10),2);
num_LCto0=sum(matrix1_ind(:,11:p1+10),2);
num_LtoC=sum(matrix2_ind(:,6:10),2);
num_CtoL=5-num_con2;
num_lin3=num_rel-sum(matrix2_ind(:,6:p1+10),2)-5;

rr1=(num_rel2+num_irr)./(10+p1);
recov1=mean(rr1);
std_recov1=std(rr1);

rr2=(num_con2+num_lin3)./(num_rel);
recov2=mean(rr2);
std_recov2=std(rr2);

rr3=(num_con2+num_lin2+num_irr)./(10+p1);
recov3=mean(rr3);
std_recov3=std(rr3);

corrC=mean(num_con2);
std_corrC=std(num_con2);
corrL=mean(num_lin2);
std_corrL=std(num_lin2);
corrCL=mean(num_rel2);
std_corrCL=std(num_rel2);
corr0=mean(num_irr);
std_corr0=std(num_irr);
LCto0=mean(num_LCto0);
std_LCto0=std(num_LCto0);
LtoC=mean(num_LtoC);
std_LtoC=std(num_LtoC);
CtoL=mean(num_CtoL);
std_CtoL=std(num_CtoL);

rmse1=sum(sqrt(sum(error1.^2,1)/n))/(M);
std_rmse1=std(sqrt(sum(error1.^2,1)/n));
rmse2=sum(sqrt(sum(error2.^2,1)/n))/(M);
std_rmse2=std(sqrt(sum(error2.^2,1)/n));
rmse3=sum(sqrt(sum(error3.^2,1)/n))/(M);
std_rmse3=std(sqrt(sum(error3.^2,1)/n));
MAD1=mean(median(abs(error1),1),2);
std_mad1=std(median(abs(error1),1));
MAD2=mean(median(abs(error2),1),2);
std_mad2=std(median(abs(error2),1));
MAD3=mean(median(abs(error3),1),2);
std_mad3=std(median(abs(error3),1));

ACC1=mean(rope1);
std_acc1=std(rope1);
ACC2=mean(rope2);
std_acc2=std(rope2);
ACC3=mean(rope3);
std_acc3=std(rope3);

Matrix1=[ rmse1 std_rmse1 MAD1 std_mad1 ACC1 std_acc1;
     rmse2 std_rmse2 MAD2 std_mad2 ACC2 std_acc2;
     rmse3 std_rmse3 MAD3 std_mad3 ACC3 std_acc3];
Matrix2=[corrC std_corrC corrL std_corrL corrCL std_corrCL corr0 std_corr0 LCto0 std_LCto0 LtoC std_LtoC CtoL std_CtoL];

rate=[recov1 std_recov1 recov2 std_recov2 recov3 std_recov3];
rate2=sum([cont1' cont2' cont3'])./M;


%%
% tables for procedure 2
matrix1_ind=P2_matrix1_ind;
matrix2_ind=P2_matrix2_ind;
error1=P2_error1;
error2=P2_error2;
error3=P2_error3;
rope1=P2_rope1;
rope2=P2_rope2;
rope3=P2_rope3;
cont1=P2_cont1;
cont2=P2_cont2;
cont3=P2_cont3;

num_rel=sum(matrix1_ind,2);      % number of relevant variables
num_con=sum(matrix2_ind,2);      % number of concave variables
num_lin=num_rel-num_con;         % number of linear variables
num_rel2=sum(matrix1_ind(:,1:10),2);      % number of relevant variables
num_con2=sum(matrix2_ind(:,1:5),2);      % number of concave variables
num_lin2=sum(matrix1_ind(:,6:10),2)-sum(matrix2_ind(:,6:10),2);         % number of linear variables
num_irr=p1-sum(matrix1_ind(:,11:p1+10),2);
num_LCto0=sum(matrix1_ind(:,11:p1+10),2);
num_LtoC=sum(matrix2_ind(:,6:10),2);
num_CtoL=5-num_con2;
num_lin3=num_rel-sum(matrix2_ind(:,6:p1+10),2)-5;

rr1=(num_rel2+num_irr)./(10+p1);
recov1=mean(rr1);
std_recov1=std(rr1);

rr2=(num_con2+num_lin3)./(num_rel);
recov2=mean(rr2);
std_recov2=std(rr2);

rr3=(num_con2+num_lin2+num_irr)./(10+p1);
recov3=mean(rr3);
std_recov3=std(rr3);

corrC=mean(num_con2);
std_corrC=std(num_con2);
corrL=mean(num_lin2);
std_corrL=std(num_lin2);
corrCL=mean(num_rel2);
std_corrCL=std(num_rel2);
corr0=mean(num_irr);
std_corr0=std(num_irr);
LCto0=mean(num_LCto0);
std_LCto0=std(num_LCto0);
LtoC=mean(num_LtoC);
std_LtoC=std(num_LtoC);
CtoL=mean(num_CtoL);
std_CtoL=std(num_CtoL);

rmse1=sum(sqrt(sum(error1.^2,1)/n))/(M);
std_rmse1=std(sqrt(sum(error1.^2,1)/n));
rmse2=sum(sqrt(sum(error2.^2,1)/n))/(M);
std_rmse2=std(sqrt(sum(error2.^2,1)/n));
rmse3=sum(sqrt(sum(error3.^2,1)/n))/(M);
std_rmse3=std(sqrt(sum(error3.^2,1)/n));
MAD1=mean(median(abs(error1),1),2);
std_mad1=std(median(abs(error1),1));
MAD2=mean(median(abs(error2),1),2);
std_mad2=std(median(abs(error2),1));
MAD3=mean(median(abs(error3),1),2);
std_mad3=std(median(abs(error3),1));

ACC1=mean(rope1);
std_acc1=std(rope1);
ACC2=mean(rope2);
std_acc2=std(rope2);
ACC3=mean(rope3);
std_acc3=std(rope3);

p2Matrix1=[ rmse1 std_rmse1 MAD1 std_mad1 ACC1 std_acc1;
     rmse2 std_rmse2 MAD2 std_mad2 ACC2 std_acc2;
     rmse3 std_rmse3 MAD3 std_mad3 ACC3 std_acc3];
p2Matrix2=[corrC std_corrC corrL std_corrL corrCL std_corrCL corr0 std_corr0 LCto0 std_LCto0 LtoC std_LtoC CtoL std_CtoL];

p2rate=[recov1 std_recov1 recov2 std_recov2 recov3 std_recov3];
p2rate2=sum([cont1' cont2' cont3'])./M;

