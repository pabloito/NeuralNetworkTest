classdef activation_function
  properties
    fun_type;
  endproperties
  methods
    function A = activation_function(param)
      A.fun_type=param;
    endfunction

    function ret = apply(AF, x)
      ret=AF.sigmoid(x);
    endfunction
    
    function ret = apply_der(AF, x)
      ret=AF.sigmoid_der(x);
    endfunction 
 
    function print_function(A)
      disp("----Object of type Activation_Function-----");
      disp("fun_type:");
      disp(A.fun_type);
      disp("----Object of type Activation_Function-----");
    endfunction

    function ret = sigmoid(AF, x)
		  ret = tanh(x);
    endfunction

    function ret = sigmoid_der(AF, x);
  		ret = 1 - x.** 2; #tanh derivative
    endfunction
  endmethods      
endclassdef

