function testing_script(filename, iterations)
    file = fopen(filename, 'w');

    time = [];
    for i=1:iterations
        time = [time, terrain(9).elapsed_time]
    endfor

    avg = mean(time);

    fprintf(file, avg);
endfunction