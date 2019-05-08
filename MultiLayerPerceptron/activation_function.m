classdef activation_function
  properties
    fun_type;
  endproperties
  methods
    function A = activation_function(param)
      A.fun_type=param;
    endfunction

    function ret = apply(AF, x)
      switch(fun_type)
        case 0
          ret = AF.step(x);
        case 1
          ret = AF.expo(x);
        case 2  
          ret = AF.sigmoid(x);
      endswitch
    endfunction
    
    function ret = apply_der(AF, x)
      switch(fun_type)
        case 0
          ret = AF.step_der(x);
        case 1
          ret = AF.expo_der(x);
        case 2  
          ret = AF.sigmoid_der(x);
      endswitch
    endfunction 
 
    function print_function(A)
      disp("----Object of type Activation_Function-----");
      disp("fun_type:");
      disp(A.fun_type);
      disp("----Object of type Activation_Function-----");
    endfunction

    function ret = step(AF, x)
		  ret = sign(x);
    endfunction

    function ret = expo(AF, x)
		  ret = 0; #TODO
    endfunction

    function ret = sigmoid(AF, x)
		  ret = tanh(x);
    endfunction

    function ret = step_der(AF, x);
  		ret = 0;
    endfunction

    function ret = expo_der(AF, x);
  		ret = 0; #TODO
    endfunction

    function ret = sigmoid_der(AF, x);
  		ret = (1 - x.**2); #tanh derivative
    endfunction
  endmethods      
endclassdef

