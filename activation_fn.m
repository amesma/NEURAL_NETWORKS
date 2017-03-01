function output = activation_fn(incomingMatrix)
    outgoingVector = zeros(size(incomingMatrix,1),size(incomingMatrix,2));
    
    for j = 1:size(incomingMatrix,2)
        for i = 1:size(incomingMatrix,1)
            outgoingVector(i,j) = 1/(1+exp(-(incomingMatrix(i,j))));
        end
    end
    output = outgoingVector;
end