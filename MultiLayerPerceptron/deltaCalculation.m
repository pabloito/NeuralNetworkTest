function deltaCalculation(expected_output, output)
  global NN;
  #Error for last (output) layer 
  current_error = expected_output-output;
  #Decreasing loop (from last layer to first layer)
  for layer_index = NN.hidden_layers + 2 : -1 : 2
          
    current_layer = NN.layers{layer_index};
    
    current_delta = NN.activation.apply_der(current_layer).*current_error;
    
    if layer_index != NN.hidden_layers + 2 
      #Remove bias
      current_delta(1) = [];
      
    endif 

    NN.deltas(layer_index) = current_delta; 
    #Dont calculate for layer_index=2 as input layer doesnt have deltas
    if layer_index != 2    
      current_weight = NN.weights{layer_index-1};
      current_error = current_weight*current_delta;
    endif        
  endfor
endfunction