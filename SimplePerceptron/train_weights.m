function final_weights = train_weights(dataset, desired_values,learning_rate,number_of_epochs)
  perm = randperm(size(dataset, 1));
  datacopy = shuffle(dataset,perm);
  desired_values_copy = shuffle(desired_values,perm);
  
  final_weights = zeros(size(datacopy)(2));
  for epoch = 1:number_of_epochs
    for row_number = 1:size(datacopy)(1)
      row=datacopy(row_number,:);
      final_weights=next_weights(row,final_weights,learning_rate,desired_values_copy(row_number,1));
    end
  end
endfunction


function matrix = shuffle(matrix, permutation)
  matrix = matrix(permutation, :);
endfunction