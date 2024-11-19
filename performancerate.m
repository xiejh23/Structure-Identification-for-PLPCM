function [corrC std_corrC corrL std_corrL corrCL std_corrCL corr0 std_corr0 LCto0 std_LCto0 LtoC std_LtoC CtoL std_CtoL]...
    =performancerate(matrix1, matrix2)

p=size(matrix1,2);
num_lin=sum(matrix1,2);
num_con=sum(matrix2,2);
num_correctlin=sum(matrix1(:,6:10),2);
num_correctcon=sum(matrix2(:,1:5),2);
num_CL=num_lin+num_con; % number of relevant variables
num_correctCL=sum(matrix1(:,1:10),2)+sum(matrix2(:,1:10),2);% number of correctly selected relevant variables
num_irr=p-num_CL; % number of irrelevant variables
num_correctirr=p-10-sum(matrix1(:,11:p),2)-sum(matrix2(:,11:p),2);
num_LCto0=10-num_correctCL;
num_LtoC=sum(matrix1(:,1:5),2);
num_CtoL=sum(matrix2(:,6:10),2);



corrC=mean(num_correctcon);
std_corrC=std(num_correctcon);
corrL=mean(num_correctlin);
std_corrL=std(num_correctlin);
corrCL=mean(num_correctCL);
std_corrCL=std(num_correctCL);
corr0=mean(num_correctirr);
std_corr0=std(num_correctirr);
LCto0=mean(num_LCto0);
std_LCto0=std(num_LCto0);
LtoC=mean(num_LtoC);
std_LtoC=std(num_LtoC);
CtoL=mean(num_CtoL);
std_CtoL=std(num_CtoL);


