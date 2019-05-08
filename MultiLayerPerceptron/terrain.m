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

  E = terrain(1:end, 1:2);
  S = terrain(1:end, 3);

  E = min_max_normalize(E,-1.7,1.7);
  S = min_max_normalize(S,tanh(-1.7),tanh(1.7));

  [E,S] = remove_random(1-sample_percentage, E, S);
    
  NN = GenericMultiLayerPerceptron();
  output = NN.train_weights(E,S);

  # Grafico terreno correcto
  plot3(E(:,1), E(:,2), S, ".");
  title ("Terrain");

  # calculated_output = calcular S con outputs.weights para plotear
  # plot3(E(:,1), E(:,2), calcu lated_output);  
  title ("Neural network's interpretation");

  function norm = normalize(m, x, y)
    norm = m - min(m(:));
    norm = norm ./ max(norm(:));
    norm = norm*(y-x) + x;
  end

  function [E,S] = remove_random(percentage, E, S)
    to_remove = floor(percentage * length(E));
    for i=1:to_remove
      r = randi([1,length(E)]);
      E(r,:) = [];
      S(r) = [];
    endfor
  endfunction
endfunction
