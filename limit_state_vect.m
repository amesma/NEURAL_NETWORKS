function vect = limit_state_vect(vect, dimensionality, upper_limit, lower_limit)
  for i=1:dimensionality
    if(vect(i) > upper_limit)
      vect(i) = upper_limit;
    end
    if (vect(i) < lower_limit)
      vect(i) = lower_limit;
    end
end
end