function NN = GenericMultiLayerPerceptron()
 source("config.conf");
 global NN
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
 initialize_multi_layer_perceptron();
endfunction

function initialize_weights()      
  global NN;
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

function initialize_layer_structure()
  global NN;
  NN.deltas = cell(NN.hidden_layers + 2, 1);
endfunction

function NN = initialize_multi_layer_perceptron(NN)
  initialize_weights();
  initialize_layer_structure();
endfunction
