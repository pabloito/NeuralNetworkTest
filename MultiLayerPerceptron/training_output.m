classdef training_output
  properties
    error;
    weights;
    analyzed_rows;
    output;
    input;
    elapsed_time;
    analyzed_epochs
  endproperties
  methods
    function obj= training_output(error_val, weights_val,analyzed_rows_val, output_val, elapsed_time, input_val, epochs_val)
      obj.error=error_val;
      obj.weights=weights_val;
      obj.analyzed_rows=analyzed_rows_val;
      obj.output = output_val;
      obj.elapsed_time = elapsed_time;
      obj.input = input_val;
      obj.analyzed_epochs = epochs_val;
    endfunction
    
    function mat = output_matrix(TO)
      fprintf("\t[%d,%d]= %d\n",[TO.input,TO.output].');
    endfunction
  endmethods
endclassdef

