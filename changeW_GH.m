function output = changeW_GH(learningConstant,w_gh,hidden_activation,output_error)
    
    output = learningConstant*(diag(fPrime(w_gh*hidden_activation))*(output_error))*hidden_activation';
    
end