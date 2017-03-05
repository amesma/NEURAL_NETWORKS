function output = makeInhibitoryWeights(dimensions,halfDimensions,maxStrength,lengthConstant)
    inhibitoryWeights = zeros(dimensions,dimensions);
    inhibitoryWeights(:,1) = makeFirstVector(dimensions,halfDimensions,maxStrength,lengthConstant);
    for i = 1:(dimensions-1)
       inhibitoryWeights(:,(i+1))=shiftElementsRight(inhibitoryWeights(:,i)); 
    end
    output = inhibitoryWeights;
end