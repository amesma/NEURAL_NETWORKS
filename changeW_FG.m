function output = changeW_FG(learningConstant, w_fg, inputPattern, w_gh, output_error, hidden_activation)
    output = learningConstant*diag(fPrime(w_fg*inputPattern))*(w_gh')*diag(output_error)*fPrime(w_gh*hidden_activation)*(inputPattern');
end

