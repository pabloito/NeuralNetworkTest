function calculate_layers(row)
  global NN;
  row = [NN.bias; row'];
  NN.layers(1)=row;
  for current_layer = 1 : NN.hidden_layers + 1
   
     current_input = NN.layers{current_layer};
     current_weight = NN.weights{current_layer};
     current_output = 0;
     
     current_output = current_weight'*current_input;

     #Don't add bias if layer is last layer 
     if(current_layer!=NN.hidden_layers +1)
      current_output = [NN.bias; current_output];
     endif
     #Add output to structure
     NN.layers(current_layer+1)= NN.activation.apply(current_output);
  endfor
endfunction