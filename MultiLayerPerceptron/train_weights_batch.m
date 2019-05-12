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
  analyzed_rows = 0;
  analyzed_epochs = 0;

  figure(1);
  title("Error evolution");
  
  analyzed_epochs =0;
  analyzed_rows = 0;
  while error>=NN.max_error
    
    #------Randomize Entry Data------
    perm = randperm(size(inputs, 1));
    inputs = shuffle(inputs,perm);
    expected_outputs = shuffle(expected_outputs,perm);
    #--------------------------------
      
    batches = get_batches(inputs, NN.batch_quantity);
    expected_batches = get_batches(expected_outputs,NN.batch_quantity);
    batchUnits =  numel(batches);
    
    epoch_outputs = zeros(size(expected_outputs));
    
    for batch_index = 1 : batchUnits
      
      current_batch = batches{batch_index};
      currented_expected_batch = expected_batches{batch_index};
      current_batch_size = rows(current_batch);
      
      outputs = zeros(size(currented_expected_batch));
      for row_index =1:current_batch_size          
        input  = current_batch(row_index, :);
        calculate_layers(input);
        
        output = NN.layers{NN.hidden_layers+2};
        
        outputs(row_index)=output;
        analyzed_rows = analyzed_rows + 1;
      endfor
      acum_error = mean(currented_expected_batch-outputs);
      #calculate Deltas
      deltaCalculation(acum_error);
      
      #update weights
      incrementalWeightUpdate();

      if(NN.adaptive_learning==1)
        updateETA(acum_error);
      endif
      output_size = rows(outputs);
      first_output_index = 1+ (batch_index-1)*NN.batch_quantity;
      last_output_index = first_output_index + output_size-1;
      epoch_outputs(first_output_index:last_output_index,:)=outputs;
    endfor

  error = mean((epoch_outputs-expected_outputs).^2)
  analyzed_epochs = analyzed_epochs+1
  plot(analyzed_epochs, error, '.', "markersize", 15, "color", "r");
  pause(0.0000001);
  hold on;
           
  endwhile
  t=toc
  output = training_output(error,NN.weights,analyzed_rows,outputs,t,inputs); 
endfunction

function batches = get_batches(inputs, batch_quantity)
  inputSize =rows(inputs);
  
  remainder = mod(inputSize, batch_quantity);
  amount_of_batches = idivide(inputSize,batch_quantity);
  
  if(remainder > 0)
    amount_of_batches = amount_of_batches+1;
  endif
  
  batches = cell(amount_of_batches,1);
  for i=1:amount_of_batches
    first_index = 1+(i-1)* batch_quantity;
    last_index=0;
    if i==amount_of_batches && remainder!=0
      last_index = first_index+remainder-1;
    else
      last_index = i*batch_quantity;
    endif
    batches(i)=inputs(first_index:last_index,:);
  endfor 
endfunction
