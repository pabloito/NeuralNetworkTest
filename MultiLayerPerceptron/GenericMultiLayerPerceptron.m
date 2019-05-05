classdef GenericMultiLayerPerceptron
  properties
    weights;
    deltas;
    layers;
    
    bias;
    hidden_layers;
    units_per_layer;
    weight_init_method;
    learning_factor;
    max_error;
    activation;
  endproperties
  methods 
    function NN = GenericMultiLayerPerceptron()
     source("config.conf");
     NN.hidden_layers = hidden_layers;
     NN.units_per_layer = units_per_layer;
     NN.weight_init_method = weight_init_method;
     NN.learning_factor = learning_factor;
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
            NN.weights(layer) = rand(NN.units_per_layer(layer + 1), NN.units_per_layer(layer) + 1);        
          endfor
      endswitch
    endfunction

    function NN = initialize_layer_structure(NN)
      NN.deltas = cell(NN.hidden_layers + 1, 1);
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
      inputUnits 		 = NN.units_per_layer(1);
      
      outputs = zeros(size(expected_outputs));
      
      error=NN.max_error;
      analized_rows = 0;
      
      while error>=NN.max_error
        for index = 1 : inputUnits
          row  = inputs(index, :);
          NN = NN.calculate_layers(row);

          expected_output = expected_outputs(index);
          output = cell2mat(NN.layers(NN.hidden_layers+2));
          
          if(output != expected_output)
            #calculate Deltas
            NN = NN.deltaCalculation(expected_output, output);
            
            
            #update weights
            NN = NN.incrementalWeightUpdate();
          endif
          analized_rows = analized_rows + 1;
         outputs(index,1)=output; 
        endfor
        error=immse(expected_outputs,outputs);
      endwhile
      output = training_output(error,NN.weights,analized_rows); 
    endfunction
    
    function NN = deltaCalculation(NN,expected_output, output)

      current_error = expected_output-output;
      
      for i=1:size(NN.layers)(1) -1
        layer_index = size(NN.layers)(1)-i+1;
        
        current_layer = cell2mat(NN.layers(layer_index-1));
        
        current_weight = cell2mat(NN.weights(layer_index-1,1));
        
        current_delta = NN.activation.apply_der(current_layer*current_weight');
        
        current_delta = current_error.*current_delta';
            
        NN.deltas(layer_index)=current_delta; 
        
        if layer_index>2
          current_weight(1)=[]; #saco el umbral
          
          current_error=current_delta*(current_weight');
        endif
      endfor
    endfunction
    
    function NN = calculate_layers(NN, row)
      row = [NN.bias, row];
      NN.layers(1)=row;
      for current_layer = 1 : NN.hidden_layers + 1
         current_input = cell2mat(NN.layers(current_layer));
         
         current_weight = cell2mat(NN.weights(current_layer,:));
         
         current_output =  zeros(1,size(current_weight)(1));
         
         for i=1:size(current_weight)(1)
           val = current_input* current_weight(i,:)';
          current_output(1,i) = NN.activation.apply(val);
         endfor
         
         if(current_layer!=NN.hidden_layers +1)
          current_output = [NN.bias, current_output];
         endif
         
         NN.layers(current_layer+1)=current_output;
      endfor
    endfunction
    
    function NN = incrementalWeightUpdate(NN)
      for weight_index = 1 : NN.hidden_layers + 1
        current_layer = cell2mat(NN.layers(weight_index));
        current_delta = cell2mat(NN.deltas(weight_index+1));
        product = (current_layer'*current_delta');
        
        current_weight= (cell2mat(NN.weights(weight_index,1)))';
        
        NN.weights(weight_index,1)= current_weight' + (NN.learning_factor *product)' + momentum(NN, weight_index);
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
      disp("learning_factor");
      disp(NN.learning_factor);
      disp("activation function");
      NN.activation.print_function(NN.activation);
      disp("max_error");
      disp(NN.max_error);
      disp("-----Object of type GenericMultiLayerPerceptron:------");
    endfunction
  endmethods  
endclassdef