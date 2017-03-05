function output = compute_inhibited_vect(inhibit_weights, initial_state_vect, state_vect, dimensionality, num_itr, epsilon)
  
new_state_vect = zeros(dimensionality,1);

for i=1:num_itr
  for k=1:(dimensionality)
    inhibit_weights(k,k) = 0;
  end
  
  for j=1:dimensionality
      new_state_vect(j,1) = state_vect(j,1) +  (epsilon * (initial_state_vect(j,1) + dot(inhibit_weights(:,j), state_vect) - state_vect(j,1)));
  end
    
    new_state_vect = limit_state_vect(new_state_vect, dimensionality, 1, 0);
    
    state_vect = new_state_vect;
end
output = state_vect;
end