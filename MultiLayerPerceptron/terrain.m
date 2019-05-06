function output = terrain(N)
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
  
NN = GenericMultiLayerPerceptron();
output = NN.train_weights(E,S);

# Grafico terreno correcto
plot3(E(:,1), E(:,2), S, ".");
title ("Terrain");

# calculated_output = calcular S con outputs.weights para plotear
# plot3(E(:,1), E(:,2), calculated_output);  
title ("Neural network's interpretation");
endfunction
