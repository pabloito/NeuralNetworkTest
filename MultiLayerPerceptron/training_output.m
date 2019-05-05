classdef training_output
  properties
    error;
    weights;
    analyzed_rows;
  endproperties
  methods
    function obj= training_output(error_val, weights_val,analyzed_rows_val)
      obj.error=error_val;
      obj.weights=weights_val;
      obj.analyzed_rows=analyzed_rows_val;
    endfunction
  endmethods
endclassdef

