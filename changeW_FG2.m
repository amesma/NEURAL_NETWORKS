function output = changeW_FG2(learningConstant, w_fg, inputPattern, w_gh, output_error, hidden_activation)
    output = learningConstant*diag(fPrime2(w_fg*inputPattern))*(w_gh')*diag(output_error)*fPrime(w_gh*hidden_activation)*(inputPattern');
end