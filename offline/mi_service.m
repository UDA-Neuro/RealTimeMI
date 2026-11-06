% Keep MATLAB and the GPU ready while a recording is acquired.
serviceDir = getenv('MI_SERVICE_DIR');
requestPath = fullfile(serviceDir, 'request.marker');
donePath = fullfile(serviceDir, 'done.marker');
errorPath = fullfile(serviceDir, 'error.txt');
stopPath = fullfile(serviceDir, 'stop.marker');
readyPath = fullfile(serviceDir, 'ready.marker');
gpuDevice();
fid = fopen(readyPath, 'w'); fclose(fid);
while ~isfile(stopPath)
    if isfile(requestPath)
        delete(requestPath);
        try
            run(fullfile(fileparts(mfilename('fullpath')), 'run_besa_mi.m'));
            fid = fopen(donePath, 'w'); fclose(fid);
        catch err
            fid = fopen(errorPath, 'w');
            fprintf(fid, '%s', getReport(err));
            fclose(fid);
        end
    else
        pause(0.05);
    end
end
