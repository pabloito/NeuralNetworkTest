function updateETA(current_error)
  global NN;
  persistent previous_errors = [];
  persistent previous_errors_diff = [0];
  persistent k = 0;
  if(k<NN.learning_rate_k)
    previous_errors = [previous_errors current_error];
    if(k <NN.learning_rate_k -1)
      previous_errors_diff = [previous_errors_diff current_error];
    endif
    k++;
  else
    k = 0;
    sumvector = previous_errors-previous_errors_diff;
    sumvector = sign(sumvector);
    sum = sum(sumvector);
    if(sum>0)
      NN.learning_rate = NN.learning_rate * (1-NN.learning_rate_b);          
     elseif(sum<0)
      NN.learning_rate = NN.learning_rate + NN.learning_rate_a;          
    endif
    previous_errors = [];
    previous_errors_diff = [0];
  endif       
endfunction