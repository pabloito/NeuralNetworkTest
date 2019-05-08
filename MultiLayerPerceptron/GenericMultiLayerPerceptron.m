classdef GenericMultiLayerPerceptron
  properties
    weights;
    deltas;
    layers;
    previous_weight_incremental;
    bias;
    hidden_layers;
    units_per_layer;
    weight_init_method;
    learning_rate;
    learning_rate_k;
    learning_rate_a;
    learning_rate_b;
    max_error;
    activation;    
    adaptive_learning;
    momentum_alpha;
  endproperties
  methods
    function NN = updateETA(NN, current_error)
      persistent previous_errors = [];
      persistent previous_errors_diff = [0];
      persistent k = 0;
      if(k<NN.learning_rate_k)
        previous_errors = [previous_errors current_error];
        if(k <NN.learning_rate_k -1)
          previous_errors_diff = [previous_errors_diff current_error];
        endif
        k++;
      else
        k = 0;
        sumvector = previous_errors-previous_errors_diff;
        sumvector = sign(sumvector);
        sum = sum(sumvector);
        if(sum>0)
          NN.learning_rate = NN.learning_rate * (1-NN.learning_rate_b);          
         elseif(sum<0)
          NN.learning_rate = NN.learning_rate + NN.learning_rate_a;          
        endif
        previous_errors = [];
        previous_errors_diff = [0];
      endif       
    endfunction

    function NN = GenericMultiLayerPerceptron()
     source("config.conf");
     NN.adaptive_learning = adaptive_learning();
     NN.hidden_layers = hidden_layers;
     NN.units_per_layer = units_per_layer;
     NN.weight_init_method = weight_init_method;
     NN.learning_rate = learning_rate;
     NN.learning_rate_k = learning_rate_k;
     NN.learning_rate_a = learning_rate_a;
     NN.learning_rate_b = learning_rate_b;
     NN.momentum_alpha = momentum_alpha;
     NN.max_error = max_error;
     NN.activation = activation_function(function_type);
     NN.bias=-1;
    
     NN.layers = cell(NN.hidden_layers+2,1);
     NN = initialize_multi_layer_perceptron(NN);
    endfunction

    function NN = initialize_weights(NN)      
      if(size(NN.units_per_layer)(2) != NN.hidden_layers + 2)
        fprintf("Error expected hidden_layers = units_per_layer column size. Found hidden_layers='%d', units_per_layer column size ='%d'",
          NN.hidden_layers + 2,size(NN.units_per_layer)(2));
          return
      endif

      NN.weights = cell(NN.hidden_layers+1,1);
      NN.previous_weight_incremental = cell(NN.hidden_layers+1,1);
      
      for layer = 1 : NN.hidden_layers + 1
        switch NN.weight_init_method
          case 0 # random    
            NN.weights(layer) = rand(NN.units_per_layer(layer) + 1, NN.units_per_layer(layer + 1));
          case 1 # he-et-al
            high = 2/sqrt(NN.units_per_layer(layer));
            low = high/4;
            NN.weights(layer) = rand(NN.units_per_layer(layer) + 1, NN.units_per_layer(layer+1)) * (high - low) + low;
        endswitch  
      endfor    
    endfunction

    function NN = initialize_layer_structure(NN)
      NN.deltas = cell(NN.hidden_layers + 2, 1);
    endfunction

    function NN = initialize_multi_layer_perceptron(NN)
      NN = initialize_weights(NN);
      NN = initialize_layer_structure(NN);
    endfunction
    
    function output = train_weights(NN, inputs, expected_outputs)
      tic
        #------Randomize Entry Data------
        perm = randperm(size(inputs, 1));
        inputs = shuffle(inputs,perm);
        expected_outputs = shuffle(expected_outputs,perm);
        #--------------------------------
        inputUnits 		 = rows(inputs);
        
        outputs = zeros(size(expected_outputs));
        error=NN.max_error;
        analized_rows = 0;

        figure
        hold on
        
        while error>=NN.max_error
          for index = 1 : inputUnits
            input  = inputs(index, :);
            NN = NN.calculate_layers(input);

            expected_output = expected_outputs(index);
            output = NN.layers{NN.hidden_layers+2};
            
            if(output != expected_output)
              #calculate Deltas
              NN = NN.deltaCalculation(expected_output, output);
              
              #update weights
              NN = NN.incrementalWeightUpdate();

              if(NN.adaptive_learning==1)
                NN = NN.updateETA(expected_output-output);
              endif
            endif
            analized_rows = analized_rows + 1;
          outputs(index,1)=output; 
        endfor

        error = mean((outputs-expected_outputs).^2);
        plot(analized_rows, error, '.')
          
        endwhile
      t=toc
      output = training_output(error,NN.weights,analized_rows,outputs,t,inputs); 
    endfunction
    
    function NN = deltaCalculation(NN,expected_output, output)
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
    
    function NN = calculate_layers(NN, row)
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
    
    function NN = incrementalWeightUpdate(NN)
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