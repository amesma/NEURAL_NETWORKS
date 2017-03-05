function output = makeFirstVector(dimensions,halfDimensions,maxStrength,lengthConstant)
    firstInhibitoryWeight = zeros(dimensions, 1);
    for i = 1:halfDimensions
        firstInhibitoryWeight(i,1) = (-maxStrength*(exp(-(i-1)/lengthConstant)));
    end
    for i = (halfDimensions+1):dimensions
        firstInhibitoryWeight(i,1) = (-maxStrength*(exp(-((dimensions+1)-i)/lengthConstant)));
    end
    output = firstInhibitoryWeight;
end