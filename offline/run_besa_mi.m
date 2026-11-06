% Called by benchmark_besa_pipeline.py. Input contains float32 sample rows.
channels = str2double(getenv('MI_NCHANNELS'));
inputPath = getenv('MI_INPUT_PATH');
outputPath = getenv('MI_OUTPUT_PATH');
fid = fopen(inputPath, 'rb');
if fid < 0
    error('Could not open resampled input');
end
closer = onCleanup(@() fclose(fid));
loadTimer = tic;
data = fread(fid, [channels, Inf], 'single=>single')';
clear closer;
loadSeconds = toc(loadTimer);
addpath(fileparts(mfilename('fullpath')));
computeTimer = tic;
[mi, windows] = recording_mi_gpu(data, 1000);
computeSeconds = toc(computeTimer);
saveTimer = tic;
writematrix([windows, mi], outputPath);
saveSeconds = toc(saveTimer);
fprintf('MI windows=%d channels=%d\n', size(mi, 1), size(mi, 2));
fprintf('MI stages: load=%.3f compute=%.3f save=%.3f seconds\n', loadSeconds, computeSeconds, saveSeconds);
