function M = momentum(NN, layer_index)
  source("config.conf");

  #W = NN.weights(layer_index,:);
  W = [1,2;3,4]
  [m,n] = size(W);
  M = zeros(m,n);
  
  for i = 1:m
    for j = 1:n
      if(i==1 && j==1)
        continue;
      else
        display((i-1) * n + j);
        M(i,j) = momentum_alpha * W((i-1) * n + j);
      endif
    endfor
  endfor
endfunction