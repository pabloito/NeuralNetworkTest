function incrementalWeightUpdate()
  global NN;
 for weight_index = 1 : NN.hidden_layers + 1
    
    current_layer = NN.layers{weight_index};
    current_delta = NN.deltas{weight_index+1};
    current_weight = NN.weights{weight_index};
    
    if(!isempty(NN.previous_weight_incremental{weight_index}))
      weight_incremental = (NN.learning_rate *  current_delta * current_layer') + NN.momentum_alpha*NN.previous_weight_incremental{weight_index};
    else
      weight_incremental = (NN.learning_rate *  current_delta * current_layer');
    endif
    NN.previous_weight_incremental(weight_index) = weight_incremental;        
    NN.weights(weight_index)= current_weight + weight_incremental';
  endfor
endfunction