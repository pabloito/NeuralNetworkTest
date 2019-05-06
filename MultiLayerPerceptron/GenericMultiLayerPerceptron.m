classdef GenericMultiLayerPerceptron
  properties
    weights;
    deltas;
    layers;
    
    bias;
    hidden_layers;
    units_per_layer;
    weight_init_method;
    learning_rate;
    max_error;
    activation;
    delta_learning_rate;
    adaptive_learning;
  endproperties
  methods 
    function NN = GenericMultiLayerPerceptron()
     source("config.conf");
     NN.adaptive_learning = adaptive_learning();
     NN.delta_learning_rate = delta_learning_rate();
     NN.hidden_layers = hidden_layers;
     NN.units_per_layer = units_per_layer;
     NN.weight_init_method = weight_init_method;
     NN.learning_rate = learning_rate;
     NN.max_error = max_error;
     disp(function_type);
     NN.activation = activation_function(function_type);
     NN.bias=-1;
    
     NN.layers = cell(NN.hidden_layers+2,1);
     NN = initialize_multi_layer_perceptron(NN);
    endfunction

    function NN = initialize_weights(NN)      
      if(size(NN.units_per_layer)(2) != NN.hidden_layers + 2)
        fprintf("Error expected hidden_layers = units_per_layer column size. Found hidden_layers='%d', units_per_layer column size ='%d'",
          hidden_layers + 2,size(units_per_layer)(2));
        exit(1);
      endif
      switch NN.weight_init_method
        case 0 # standard
          NN.weights = cell(NN.hidden_layers+1,1);
          
          for layer = 1 : NN.hidden_layers + 1
            NN.weights(layer) = rand(NN.units_per_layer(layer) + 1, NN.units_per_layer(layer + 1));        
          endfor
      endswitch
    endfunction

    function NN = initialize_layer_structure(NN)
      NN.deltas = cell(NN.hidden_layers + 2, 1);
    endfunction

    function NN = initialize_multi_layer_perceptron(NN)
      NN = initialize_weights(NN);
      NN = initialize_layer_structure(NN);
    endfunction
    
    function output = train_weights(NN, inputs, expected_outputs)
      #------Randomize Entry Data------
      perm = randperm(size(inputs, 1));
      inputs = shuffle(inputs,perm);
      expected_outputs = shuffle(expected_outputs,perm);
      #--------------------------------
      inputUnits 		 = rows(inputs);
      
      outputs = zeros(size(expected_outputs));
      
      error=NN.max_error;
      analized_rows = 0;
      
      while error>=NN.max_error
        for index = 1 : inputUnits
          input  = inputs(index, :);
          NN = NN.calculate_layers(input);

          expected_output = expected_outputs(index);
          output = NN.activation.apply(NN.layers{NN.hidden_layers+2});
          
          if(output != expected_output)
            #calculate Deltas
            NN = NN.deltaCalculation(expected_output, output);
            
            #update weights
            NN = NN.incrementalWeightUpdate();
          endif
          analized_rows = analized_rows + 1;
         outputs(index,1)=output; 
        endfor
        error=immse(expected_outputs,outputs)
        if (isnan(error))
          dbstop 71
        endif
 
      endwhile
    
      output = training_output(error,NN.weights,analized_rows); 
    endfunction
    
    function NN = deltaCalculation(NN,expected_output, output)

      current_error = expected_output-output;
      for elem = current_error
        for elem2 = elem
          if(NN.adaptive_learning==1)
            [NN.delta_learning_rate, delta_n] = NN.delta_learning_rate.calculate_learning_rate(elem);
            NN.learning_rate = NN.learning_rate + delta_n;
          endif
        endfor
      endfor
      
      for layer_index = NN.hidden_layers + 2 : -1 : 2
        
        current_layer = NN.layers{layer_index};
        
        current_weight = NN.weights{layer_index-1};
        
        current_delta = NN.activation.apply_der(current_layer).*current_error;
        
        if layer_index != NN.hidden_layers + 2 
          
          current_delta(1) = [];
          
        endif 
  
        NN.deltas(layer_index) = current_delta; 
  
        current_error = current_weight.*current_delta';

        for elem = current_error
          for elem2 = elem
            if(NN.adaptive_learning==1)
              [NN.delta_learning_rate, delta_n] = NN.delta_learning_rate.calculate_learning_rate(elem);
              NN.learning_rate = NN.learning_rate + delta_n;
            endif
          endfor
        endfor
      
      endfor
    endfunction
    
    function NN = calculate_layers(NN, row)
      row = [NN.bias; row'];
      NN.layers(1)=row;
      for current_layer = 1 : NN.hidden_layers + 1
         current_input = NN.layers{current_layer};
         
         current_weight = NN.weights{current_layer};
         
         current_output = current_weight'*NN.activation.apply(current_input);
         
         if(current_layer!=NN.hidden_layers +1)
          current_output = [NN.bias; current_output];
         endif
         
         NN.layers(current_layer+1)=current_output;
      endfor
    endfunction
    
    function NN = incrementalWeightUpdate(NN)
      for weight_index = 1 : NN.hidden_layers + 1
        current_layer = NN.layers{weight_index};
        current_delta = NN.deltas{weight_index+1};
        current_weight = NN.weights{weight_index};
        
        weight_incremental = NN.learning_rate * NN.activation.apply(current_layer) * current_delta';
        
        NN.weights(weight_index)= current_weight + weight_incremental;# + momentum(NN, weight_index);
      endfor
    endfunction
    
    function print_neural_network(NN)
      disp("-----Object of type GenericMultiLayerPerceptron:------");
      disp("Weights: ");
      disp(NN.weights);
      disp("Deltas: ");
      disp(NN.deltas);
      disp("Hidden Layers");
      disp(NN.hidden_layers);
      disp("Units per Layer");
      disp(NN.units_per_layer);
      disp("weight_init_method");
      disp(NN.weight_init_method);
      disp("learning_rate");
      disp(NN.learning_rate);
      disp("activation function");
      NN.activation.print_function(NN.activation);
      disp("max_error");
      disp(NN.max_error);
      disp("-----Object of type GenericMultiLayerPerceptron:------");
    endfunction
  endmethods  
endclassdef