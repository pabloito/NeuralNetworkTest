function ret = predict(row, weights)
  value = dot_product(row,weights);
  if value>0
    ret=1;
  else
    ret=0;
  end
end