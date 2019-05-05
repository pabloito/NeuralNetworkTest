function output = xor(N)
  switch N
    case 2
      input=[
            0,0;
            0,1;
            1,0;
            1,1
            ];
      expected_output=[
            0;
            1;
            1;
            0;
            ];      
    otherwise
    fprintf("Expected N=2, found N='%d'\n",N);
    return;
  endswitch
  
  NN = GenericMultiLayerPerceptron();
  output = NN.train_weights(input,expected_output);
  
endfunction
