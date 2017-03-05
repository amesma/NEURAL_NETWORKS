function output = shiftElementsRight(inputVector)    
    dimensions = size(inputVector,1);
    outputVector = zeros(dimensions,1);

    for i = 1:dimensions-1
       outputVector(i+1,1) = inputVector(i,1);
   
    end
    outputVector(1,1) = inputVector(dimensions,1);
    output = outputVector;

end