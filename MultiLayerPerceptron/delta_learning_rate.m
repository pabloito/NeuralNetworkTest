classdef delta_learning_rate
  properties
    learning_rate_k;
    learning_rate_a;
    learning_rate_b;
    learning_factor;

    errors_size=0;
    errors;
  endproperties
  methods
    function DLR = delta_learning_rate()
      source("config.conf");
      DLR.learning_rate_k = learning_rate_k;
      DLR.learning_rate_a = learning_rate_a;
      DLR.learning_rate_b = learning_rate_b;
      DLR.learning_factor = learning_factor;
      DLR.errors=zeros(1,learning_rate_k);
    endfunction

    function [DLR,delta_n] = calculate_learning_rate(DLR, delta_E)

      if(DLR.errors_size <= DLR.learning_rate_k)
        DLR.errors_size++;
        DLR.errors(DLR.errors_size) = delta_E;
      else
        DLR.errors(end+1) = delta_E
        DLR.errors(1)=[];
      endif

      if(delta_E == 0)
        delta_n = 0;
      elseif(DLR.last_k_are_negative())
        delta_n = DLR.learning_rate_a;
      else
        delta_n = -1*DLR.learning_factor*DLR.learning_rate_b;
      endif   
    endfunction

    function all_negative_flag = last_k_are_negative(DLR)
      if(DLR.errors_size <= DLR.learning_rate_k)
        all_negative_flag = false;
        return;
      endif

      all_negative_flag = true;

      for elem = DLR.errors
        if(elem > 0)
          all_negative_flag=false;
          return
        endif
      end
    endfunction

    function print(DLR)
      display(DLR.learning_rate_k);
      display(DLR.learning_rate_a);
      display(DLR.learning_rate_b);
      display(DLR.errors_size);
      display(DLR.errors);
    endfunction
  endmethods
endclassdef