function output = train_weights(inputs, expected_outputs)
  global NN;
  tic
    inputUnits 		 = rows(inputs);
    
    outputs = zeros(size(expected_outputs));
    error=NN.max_error;
    analized_rows = 0;

    figure(1);
    title("Error evolution");
    
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
            updateETA(expected_output-output);
          endif
        endif
        analized_rows = analized_rows + 1;
      outputs(index,1)=output; 
    endfor

    error = mean((outputs-expected_outputs).^2);
    plot(analized_rows, error, '.', "markersize", 15, "color", "r");
    pause(0.0000001)
    hold on
             
    endwhile
  t=toc
  output = training_output(error,NN.weights,analized_rows,outputs,t,inputs); 
endfunction
