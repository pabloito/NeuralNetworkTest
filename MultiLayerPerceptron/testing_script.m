function testing_script(filename_out, filename_in)
    source("config.conf");
    file = fopen(filename_out, 'w');

    fprintf(file, "error: "); 
    fprintf(file, num2str(max_error)); 

    epochs = [];
    for i=1:10
        epochs = [epochs, train(filename_in, 0.9).analyzed_epochs]
    endfor

    avg = mean(epochs);

    fprintf(file, ", average epochs: "); 
    fprintf(file, num2str(avg));
endfunction