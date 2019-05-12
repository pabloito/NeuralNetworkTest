function weight_incrementals = batch_weight_incrementals_calc()
 global NN;
 weight_incrementals =cell(NN.hidden_layers+1,1);
 for weight_index = 1 : NN.hidden_layers + 1
    
    current_layer = NN.layers{weight_index};
    current_delta = NN.deltas{weight_index+1};
    current_weight = NN.weights{weight_index};
    
    weight_incremental = (NN.learning_rate *  current_delta * current_layer');
    weight_incrementals(weight_index) = weight_incremental; 
  endfor
  NN.previous_weight_incremental(:) = weight_incremental;
endfunction
