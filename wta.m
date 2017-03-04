function [w,x] = wta(alpha,x,wzero,iter)

%Winner take all algorithm

%alpha is the learning rate

%wzero is the initial weight vector

%x is the matrix of column (training) vectors

%iter is the number of presentations

%the function returns the final weights (wf)

%and x, the training patterns

 

[a,b] = size(x);

[c,d] = size(wzero);

w = wzero;

%x loop on the outside, w loop on the inside

for k = 1:iter

    for i = 1:b

        max=-1;

        ind = 1;

        for j = 1:d

            product = (w(1:c,j)')*x(1:a,i);

            if (product>max)

                ind = j;

                max = product;

            end

        end

        %update the weight vector

        w(1:c,ind) = w(1:c,ind) + alpha.*(x(1:a,i) - w(1:c,ind));

        w(1:c,ind) = w(1:c,ind)./(norm(w(1:c,ind)));

    end

end

