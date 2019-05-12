function output = train(path, sample_percentage)
  #graphics_toolkit("gnuplot")
  terrain = dlmread(path);

  E = terrain(2:end, 1:2);
  X = E(:,1);
  Y = E(:,2);
  
  S = terrain(2:end, 3);
  Z = S;

  GenericMultiLayerPerceptron();
  global NN
  
  if(NN.activation.fun_type == 2)
    E = min_max_normalize(E,-1.7,1.7);
    S = min_max_normalize(S,tanh(-1.7),tanh(1.7));
  endif

  #E = gaussian_normalize(E);
  #S = gaussian_normalize(S);

  [E_rand,S_rand] = remove_random(1-sample_percentage, E, S);
    
  output=[];
  switch NN.method
    case 0
      output = train_weights(E_rand,S_rand);
    case 1
      output = train_weights_batch(E_rand,S_rand);
    otherwise
      fprintf("ERROR: Unrecognized Method: '%d'",NN.method);
  endswitch
    

  # Terrain graphic
  colormap('default');
  figure(3, 'name', 'Terrain', 'numbertitle', 'off', 'position', [400 100 580 380]);
  tri = delaunay(X, Y);
  trisurf(tri, X, Y, Z, 'facecolor', 'interp');
  hold on

  plot_nn(NN, output.weights, E(:,1), E(:,2), Z, X, Y);

  function data_set = gaussian_normalize(m)
    mean = mean(m);  
    standard_dev = std(m);
    data_set = (m - mean) ./ standard_dev;
  endfunction

  function norm = min_max_normalize(m, x, y)
    norm = m - min(m(:));
    norm = norm ./ max(norm(:));
    norm = norm*(y-x) + x;
  endfunction

  function [E,S] = remove_random(percentage, E, S)
    to_remove = floor(percentage * length(E));
    for i=1:to_remove
      r = randi([1,length(E)]);
      E(r,:) = [];
      S(r) = [];
    endfor
  endfunction
endfunction

function plot_nn(NN, weights, X, Y, Z, X_orig, Y_orig)
   z_n = [];

   for index = 1:numel(X)
        z_n(index) = run_pattern(NN, weights, [X(index); Y(index)]);
    endfor

    if(NN.activation.fun_type == 2)
      z_n = min_max_normalize(z_n, min(Z), max(Z));
    endif
    
    # Interpretation graphic
    colormap('default');
    figure(4, 'name', "Neural network's interpretation", 'numbertitle', 'off', 'position', [1000 100 580 380]);
    tri = delaunay(X_orig, Y_orig);
    trisurf(tri, X_orig, Y_orig, z_n, 'facecolor', 'interp');
    hold on
endfunction

function output = run_pattern(NN, weights, row)
      row = [NN.bias; row];
      NN.layers(1)=row;

      for current_layer = 1 : NN.hidden_layers + 1
       
         current_input = NN.layers{current_layer};
         current_weight = weights{current_layer};
         
         current_output = current_weight'*current_input;
 
         #Don't add bias if layer is last layer 
         if (current_layer!=NN.hidden_layers +1)
           current_output = [NN.bias; current_output];
         endif

         NN.layers(current_layer+1)= NN.activation.apply(current_output);

         if (current_layer == NN.hidden_layers + 1)
            output = NN.activation.apply(current_output);
         endif
      endfor
endfunction