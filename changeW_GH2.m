function output = changeW_GH2(learningConstant,w_gh,hidden_activation,output_error)
    
    output = learningConstant*(diag(fPrime2(w_gh*hidden_activation))*(output_error))*hidden_activation';
    
end