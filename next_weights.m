function new_weights = next_weights(row,old_weights,learning_rate, desired_value)
  prediction = predict(row, old_weights);
  error = desired_value - prediction;
  #fprintf("error: '%d',desired_value: '%d',prediction: '%d',",error,desired_value,prediction)
  for i = 1:size(row)(2)
				new_weights(i) = old_weights(i) + learning_rate * error * row(i);
  end
end