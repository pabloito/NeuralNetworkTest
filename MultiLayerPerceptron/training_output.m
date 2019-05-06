classdef training_output
  properties
    error;
    weights;
    analyzed_rows;
    output;
    elapsed_time;
  endproperties
  methods
    function obj= training_output(error_val, weights_val,analyzed_rows_val, output_val, elapsed_time)
      obj.error=error_val;
      obj.weights=weights_val;
      obj.analyzed_rows=analyzed_rows_val;
      obj.output = output_val;
      obj.elapsed_time = elapsed_time;
    endfunction
  endmethods
endclassdef

