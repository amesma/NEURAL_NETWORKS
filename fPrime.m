function output = fPrime(inputVector)
    size = length(inputVector);
    outputVector = zeros(size,1);
    for i = 1:size
        outputVector(i,1) = (exp(inputVector(i,1)))/(((exp(inputVector(i,1)))+1)^2);
    end
    output = outputVector;
end