function value = dot_product(vector1,vector2)
  value=0;
  for i = 1:size(vector1)(2)
    value+= vector1(i)*vector2(i);
  end
endfunction