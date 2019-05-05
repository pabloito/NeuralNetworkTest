function weight_1,weight_2 = twoLayerPerceptron(input,expected_output,LR,echoes)
  
  #------Randomize Entry Data------
  perm = randperm(size(input, 1));
  input = shuffle(input,perm);
  expected_output = shuffle(expected_output,perm);
  #--------------------------------
  
  #------Randomize Initial Weights--
  weight_1 = rand(size(input);
  #---------------------------------
  
  for i = 1:echoes
    #----Find layer values---
    l1=A(dot(l0,weight_matrix(:,1)));
    l2=A(dot(l1,weight_matrix(:,2)));
    #-----------------------
    
    #---Calculate Errors and Deltas--
    l2_error = l2-expected_output;
    l2_delta = l2_error.*A_der(l2);
    
    l1_error = dot(l2_delta,weight_matrix(:,2));
    l1_delta = l1_error.*A_der(l1);
    #-------------------------------
    
    #---Update Weight Values-----
    weight_matrix(:,2) += LR * (dot(l1,l2_delta));
    weight_matrix(:,1) += LR * (dot(l0,l1_delta));
    #----------------------------
  endfor
endfunction


function matrix = shuffle(matrix, permutation)
  matrix = matrix(permutation, :);
endfunction

function val = A(x)
  val = sigmoid(x);
endfunction

function der = A_der(x)
  der = sigmoid_der(x);
endfunction
