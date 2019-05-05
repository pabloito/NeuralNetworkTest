function isCorrect = checkSolution(finalWeights, dataset, desiredValues)
  for i=1:size(dataset)(1)
    desiredValue=desiredValues(i,1);
    datarow = dataset(i,:);
    
    product = dot_product(datarow,finalWeights);
    if desiredValue==1
      if product<=0
        fprintf("Error: Expected product greater than 0, found product ='%d'.\n",product);
        return;
      endif
    else
      if product>0
        fprintf("Error: Expected product less than or equal to 0, found product ='%d'.\n",product);
        return;
      endif
    endif
  endfor
  disp("OK: Results checked!");
  isCorrect=true;
endfunction
  