function [X,y,y_true] = DGP(n,p,SNR,alpha,v)

% data generating process

    Q=binornd(1,alpha,5,5)*0.5;
    for i=1:5 
        Q(i,i)=1;
    end


     for i=1:p 
        for j=1:p
             delta(i,j)=v^(abs(i-j));
        end
     end
    X=mvnrnd(zeros(p,1),delta,n);
    for j=1:p 
        X(:,j)=(X(:,j)-min(X(:,j)))/(max(X(:,j))-min(X(:,j)));
    end
    x1 = X(:,1:5);
    x2 = X(:,6:10);
    x3 = X(:,11:p);
    b = [1 1 1 1 1]';
    for i=1:n 
%         f1(i,1)=4*(x1(i,:)-1/2)*Q*(x1(i,:)-1/2)';
        f1 = 4./(1+exp(-x1*b));
    end
    f = -f1+ x2*b;
    s1=sqrt(var(f))/SNR;
    u = normrnd(0,s1,[n,1]);
    y=f+u;
    y_true=f;
    X = [x1 x2 x3];
