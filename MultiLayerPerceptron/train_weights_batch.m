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
  eta_updates = 0;
  
  error_plot = figure(1, 'name', 'Error evolution', 'numbertitle', 'off', 'position', [400 600 580 380]);
  eta_plot = figure(2, 'name', 'Eta evolution', 'numbertitle', 'off', 'position', [1000 600 580 380]);
  
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
      
      final_weight_increments = cell(NN.hidden_layers+1,1);
      
      for row_index =1:current_batch_size          
        input  = current_batch(row_index, :);
        calculate_layers(input);
        
        output = NN.layers{NN.hidden_layers+2};
        expected_output = currented_expected_batch(row_index);
        
        error = expected_output-output;
        #calculate Deltas
        deltaCalculation(error);
        
        current_weight_increments=batch_weight_incrementals_calc();
        
        final_weight_increments = sum_weight_increments(final_weight_increments,current_weight_increments);
        
        outputs(row_index)=output;
        analyzed_rows = analyzed_rows + 1;
      endfor
      #update weights
      for j=1:NN.hidden_layers+1
        NN.weights{j}= NN.weights{j}+final_weight_increments{j}';
      endfor    

      if(NN.adaptive_learning==1)
        eta_updates = updateETA(expected_output-output, eta_plot, eta_updates);
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
  output = training_output(error,NN.weights,analyzed_rows,epoch_outputs,t,inputs); 
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

