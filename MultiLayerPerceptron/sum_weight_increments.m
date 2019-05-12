function weight_increments = sum_weight_increments(weight_increments,other_weight_increments)
  global NN;  
  for j=1:NN.hidden_layers+1
    if isempty(weight_increments{j})
      weight_increments(j)=other_weight_increments{j};
    else
      weight_increments(j)= weight_increments{j}+other_weight_increments{j};
    endif
  endfor  
endfunction
