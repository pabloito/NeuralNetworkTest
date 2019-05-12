function testing_script(filename_out, filename_in, sample, iterations)
    source("config.conf");
    file = fopen(filename_out, 'w');

    fprintf(file, "error: "); 
    fprintf(file, num2str(max_error)); 

    epochs = [];
    for i=1:iterations
        epochs = [epochs, train(filename_in, sample).analyzed_epochs]
    endfor

    avg = mean(epochs);

    fprintf(file, ", average epochs: "); 
    fprintf(file, num2str(avg));
endfunction