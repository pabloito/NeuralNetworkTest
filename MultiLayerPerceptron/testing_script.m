function testing_script(filename, sample, iterations)
    source("config.conf");
    file = fopen(filename, 'w');

    fprintf(file, "error: "); 
    fprintf(file, max_error); 

    epochs = [];
    for i=1:iterations
        epochs = [epochs, terrain(filename, sample).analyzed_epochs]
    endfor

    avg = mean(epochs);

    fprintf(file, num2str(avg));
endfunction