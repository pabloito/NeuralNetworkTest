classdef activation_function
  properties
    fun_type;
  endproperties
  methods
    function A = activation_function(param)
      A.fun_type=param;
    endfunction

    function ret = apply(AF, x)
      switch(AF.fun_type)
        case 0
          ret = AF.linear(x);
        case 1
          ret = AF.sigmoid_exp(x);
        case 2  
          ret = AF.hiperbolic_tangent(x);
      endswitch
    endfunction
    
    function ret = apply_der(AF, x)
      switch(AF.fun_type)
        case 0
          ret = AF.linear_der(x);
        case 1
          ret = AF.sigmoid_exp_der(x);
        case 2  
          ret = AF.hiperbolic_tangent_der(x);
      endswitch
    endfunction 
 
    function print_function(A)
      disp("----Object of type Activation_Function-----");
      disp("fun_type:");
      disp(A.fun_type);
      disp("----Object of type Activation_Function-----");
    endfunction

    function ret = linear(AF, x)
		  ret = x;
    endfunction

    function ret = sigmoid_exp(AF, x)
		  ret = 1.0 ./ (1.0 + exp(-x));
    endfunction

    function ret = hiperbolic_tangent(AF, x)
		  ret = tanh(x);
    endfunction

    function ret = linear_der(AF, x);
  		ret = sign(x); # constante
    endfunction

    function ret = sigmoid_exp_der(AF, x);
  		ret = x .* (1-x);
    endfunction

    function ret = hiperbolic_tangent_der(AF, x);
  		ret = (1 - x.**2); #tanh derivative
    endfunction
  endmethods      
endclassdef

