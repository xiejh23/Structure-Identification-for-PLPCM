clear
clc
n=800;
SNR=5;
M=100;
alpha=0.5;
p1=16;
p=5+5+p1;
v=0;
matrix1_ind=zeros(M,p);
matrix2_ind=zeros(M,p);
matrix3_ind=zeros(M,p);
matrix4_ind=zeros(M,p);
matrix5_ind=zeros(M,p);
matrix6_ind=zeros(M,p);

for m=1:M
%%
% data generating process

    [X,y,y_true] = DGP(n,p,SNR,alpha,v);
    data.X{m}=X;
    data.y{m}=y;
    data.y_true{m}=y_true;
    
%       X=data.X{m};
%       y=data.y{m};
%       y_true=data.y_true{m};

    %%
    % two step land
    
%     [MSE_ssanova(m),MSE_land(m),MSE_2land(m),vec_lin,vec_nonlin] = twostepland(X,y,y_true);
%     matrix1_ind(m,:)=vec_lin;
%     matrix2_ind(m,:)=vec_nonlin;

    
    %%
    % procedure 1
     [MSE_cnls1(m),vec_lin1,vec_con1] = cnls_procedure1(X,y,y_true);
     
    matrix3_ind(m,:)=vec_lin1;
    matrix4_ind(m,:)=vec_con1;

    
    %%
    % procedure 2
    
    [MSE_cnls2(m),vec_lin2,vec_con2] = cnls_procedure2(X,y,y_true);
    matrix5_ind(m,:)=vec_lin2;
    matrix6_ind(m,:)=vec_con2;

end

matrix={matrix1_ind matrix2_ind;matrix3_ind matrix4_ind;matrix5_ind matrix6_ind};
for i=1:3 
    matrix1=matrix{i,1};
    matrix2=matrix{i,2};
    [corrC(i) std_corrC(i) corrL(i) std_corrL(i) corrCL(i) std_corrCL(i) corr0(i) std_corr0(i) LCto0(i) std_LCto0(i) LtoC(i) std_LtoC(i) CtoL(i) std_CtoL(i)]...
    =performancerate(matrix1, matrix2);

end
performance1=[ mean(MSE_cnls1) mean(MSE_cnls2);
     std(MSE_cnls1) std(MSE_cnls2)];
performance2=[corrC;std_corrC;corrL;std_corrL;corrCL;std_corrCL;corr0;std_corr0;LCto0;std_LCto0;LtoC;std_LtoC;CtoL;std_CtoL];
