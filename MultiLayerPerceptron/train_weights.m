function output = train_weights(inputs, expected_outputs)
  global NN;
  tic
    inputUnits 		 = rows(inputs);
    
    outputs = zeros(size(expected_outputs));
    error=NN.max_error;
    analized_rows = 0;
    error_updates = 0;
    eta_updates = 0;

    error_plot = figure(1, 'name', 'Error evolution', 'numbertitle', 'off', 'position', [500 500 580 380]);
    eta_plot = figure(2, 'name', 'Eta evolution', 'numbertitle', 'off', 'position', [1100 500 580 380]);
    
    while error>=NN.max_error
      
      #------Randomize Entry Data------
      perm = randperm(size(inputs, 1));
      inputs = shuffle(inputs,perm);
      expected_outputs = shuffle(expected_outputs,perm);
      #--------------------------------
      
      for index = 1 : inputUnits
        input  = inputs(index, :);
        calculate_layers(input);

        expected_output = expected_outputs(index);
        output = NN.layers{NN.hidden_layers+2};
        
        if(output != expected_output)
          #calculate Deltas
          deltaCalculation(expected_output, output);
          
          #update weights
          incrementalWeightUpdate();

          if(NN.adaptive_learning==1)
            eta_updates = updateETA(expected_output-output, eta_plot, eta_updates);
          endif
        endif
        analized_rows = analized_rows + 1;
        outputs(index,1)=output; 
      endfor

      error = mean((outputs-expected_outputs).^2);
      figure(error_plot);
      plot(error_updates, error, '.', 'markersize', 15, 'color', 'r');
      pause(0.0000001)
      hold on
      error_updates = error_updates + 1;
             
    endwhile
  t=toc
  output = training_output(error,NN.weights,analized_rows,outputs,t,inputs); 
endfunction
