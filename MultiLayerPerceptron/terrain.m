function output = terrain(N)
  source("config.conf");

  if(N<=0 || N>14)
    display("flasheaste");
    return;
  endif

  if(N<10)
    terrain = dlmread(strcat("terrain/terrain0",num2str(N),".data"));
  else
    terrain = dlmread(strcat("terrain/terrain",num2str(N),".data"));
  endif

  E = terrain(2:end, 1:2);
  X = E(:,1);
  Y = E(:,2);
  
  S = terrain(2:end, 3);
  Z = S;

  E = min_max_normalize(E,-1.7,1.7);
  S = min_max_normalize(S,tanh(-1.7),tanh(1.7));

  #E = gaussian_normalize(E);
  #S = gaussian_normalize(S);

  [E_rand,S_rand] = remove_random(1-sample_percentage, E, S);
    
  NN = GenericMultiLayerPerceptron();
  output = NN.train_weights(E_rand,S_rand);

  # Terrain graphic
  colormap('default');
  #[X Y] = meshgrid(X, Y);
  #tri = delaunay(X, Y;
  #trisurf(tri,X, Y, Z,'facecolor','interp');
  figure(2)
  plot3(X, Y, Z, '.');
  title ("Terrain");
  hold on

  plot_nn(NN, E(:,1), E(:,2), Z, X, Y);

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

function plot_nn(NN, X, Y, Z, X_orig, Y_orig)
   z_n = [];

   for index = 1:numel(X)
        z_n(index) = run_pattern(NN, [X(index); Y(index)]);
    endfor

    min(Z)
    max(Z)
    z_n = min_max_normalize(z_n, min(Z), max(Z));
    
    # Interpretation graphic
    colormap('default');
    #tri = delaunay(X, Y);
    #trisurf(tri,X_orid, Y_orig, z_n,'facecolor','interp');
    figure(3)
    plot3(X_orig, Y_orig, z_n, '.');
    title ("Neural network's interpretation");
endfunction

function output = run_pattern(NN, row)
      row = [NN.bias; row];
      NN.layers(1)=row;

      for current_layer = 1 : NN.hidden_layers + 1
       
         current_input = NN.layers{current_layer};
         current_weight = NN.weights{current_layer};
         
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