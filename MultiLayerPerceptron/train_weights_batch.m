function output = train_weights_batch(inputs, expected_outputs)
  global NN;
  tic
  inputUnits 		 = rows(inputs);
  
  if(NN.batch_quantity > inputUnits)
    fprintf("ERROR: batch_quantity ('%d'), cannot be larger than inputUnits ('%d')",NN.batch_quantity,inputUnits);
    return
  endif

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
      
    batches = get_batches(inputs, NN.batch_quantity); #(1) Falta codear esta funcion.
    batchUnits =   #number of batches
    
    for batch_index = 1 : batchUnits
      
      current_batch = batches{i} #assuming batches is Cell.
      current_batch_size = rows(current_batch);
      
      acum_error = 0
      
      for row_index =1:current_batch_size          
        input  = inputs(index, :);
        calculate_layers(input);

        expected_output = expected_outputs(index);
        output = NN.layers{NN.hidden_layers+2};
        
        # (2) Falta definir como se acumula el error.
        acum_error = acum_error + #algo
          
      endfor      
      #calculate Deltas
      deltaCalculation(acum_error); #(3) cambiar deltaCalculation para que reciba solo un argumento.
      
      #update weights
      incrementalWeightUpdate();

      if(NN.adaptive_learning==1)
        updateETA(expected_output-output);
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
