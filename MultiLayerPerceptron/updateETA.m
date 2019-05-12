function eta_updates = updateETA(current_error, eta_plot, eta_updates)
  global NN;
  persistent previous_errors = [];
  persistent previous_errors_diff = [0];
  persistent k = 0;

  if ( k < NN.learning_rate_k)
    previous_errors = [previous_errors current_error];
    if ( k < NN.learning_rate_k -1)
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
    
    figure(eta_plot);
    plot(eta_updates, NN.learning_rate, '.', 'markersize', 15, 'color', 'b');
    hold on
    eta_updates = eta_updates + 1;
    
    previous_errors = [];
    previous_errors_diff = [0];
  endif       
endfunction