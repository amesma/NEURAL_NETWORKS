function output = fPrime2(inputVector)
    size = length(inputVector);
    outputVector = zeros(size,1);
    for i = 1:size
        outputVector(i,1) = (4*exp(2*(inputVector(i,1))))/(( (exp(2*(inputVector(i,1)))) + 1 )^2); 
    end
    output = outputVector;
end