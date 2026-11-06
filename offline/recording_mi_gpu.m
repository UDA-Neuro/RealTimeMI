function [mi, windowSeconds] = recording_mi_gpu(data, sampleRate)
% RECORDING_MI_GPU Compute 30-second PAC windows at 15-second intervals.
% data is samples by channels. mi is windows by channels. windowSeconds
% contains start and end times, in seconds from the start of the recording.
% Requires MATLAB Parallel Computing Toolbox and a supported NVIDIA GPU.

    validateattributes(data, {'single','double'}, {'2d','nonempty','real','finite'});
    validateattributes(sampleRate, {'numeric'}, {'scalar','real','finite','>',500});

    [samples, channels] = size(data);
    hop = round(15 * sampleRate);
    width = 2 * hop;
    if samples < width
        error('recording_mi_gpu:ShortRecording', 'At least 30 seconds of input are required.');
    end
    starts = 0:hop:samples-width;
    windowSeconds = [starts(:), starts(:)+width] / sampleRate;
    mi = nan(numel(starts), channels);

    % Reflect the recording edges so the centered FIR covers every window.
    delay = round(1.5 * sampleRate);
    paddedIndices = [delay+1:-1:2, 1:samples, samples-1:-1:samples-delay];
    filteredIndices = 2*delay+(1:samples);
    nfft = 2^nextpow2(samples + 4*delay);
    lag = (-delay:delay)';
    taper = 0.5 + 0.5*cos(pi*lag/delay);
    phaseFilter = fft(gpuArray(analyticKernel(3, 4, sampleRate, lag, taper)), nfft);
    amplitudeFilter = fft(gpuArray(analyticKernel(80, 250, sampleRate, lag, taper)), nfft);

    bins = 18;
    batchSize = 4;
    for first = 1:batchSize:channels
        cols = first:min(first+batchSize-1, channels);
        spectrum = fft(gpuArray(double(data(paddedIndices,cols))), nfft, 1);
        phase = ifft(spectrum.*phaseFilter, [], 1);
        labels = gather(min(floor((angle(phase(filteredIndices,:))+pi)*(bins/(2*pi))),bins-1)+1);
        amplitude = ifft(spectrum.*amplitudeFilter, [], 1);
        amps = gather(abs(amplitude(filteredIndices,:)));
        for col = 1:numel(cols)
            for k = 1:numel(starts)
                indices = starts(k)+(1:width);
                ids = labels(indices,col);
                counts = accumarray(ids,1,[bins 1]);
                sums = accumarray(ids,amps(indices,col),[bins 1]);
                if any(counts == 0) || ~any(sums > 0)
                    continue
                end
                means = sums./counts;
                p = means/sum(means);
                p = p(p > 0);
                mi(k,cols(col)) = min(1,max(0,sum(p.*log(bins*p))/log(bins)));
            end
        end
    end
end

function kernel = analyticKernel(low, high, rate, lag, taper)
    bandwidth = (high-low)/rate;
    kernel = 2*bandwidth*sinc(bandwidth*lag).*taper.*exp(1i*pi*(low+high)/rate*lag);
end
