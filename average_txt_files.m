%% average_txt_files.m
% Import experiment text files (1.txt, 2.txt, ..., N.txt) in a loop,
% take the time average of every column, and store the result as one
% 1 x nCols row per file (condition) in a matrix.
%
% Each file: 10 s of data at 1 kHz (~10,000 rows). Column 1 is time,
% columns 2..end are measured channels.
%
% Output:
%   avgData   - nFiles x nCols matrix. Row k = time-average of file k.
%               (Column 1 is the mean of the time column, ~5 s, kept so
%               each row stays 1 x nCols as in the original file.)
%   fileIdx   - nFiles x 1 vector of file numbers (row k <-> fileIdx(k).txt)
%   nSamples  - nFiles x 1 vector of rows read from each file
% Saved to averaged_data.mat and averaged_data.csv.

clear; clc;

%% ---------------- User settings ----------------
dataFolder = pwd;        % folder containing 1.txt, 2.txt, ... (change as needed)
firstFile  = 1;          % first file number
lastFile   = 100;        % last file number (e.g. 100 or 200)
fs         = 1000;       % sampling frequency [Hz] (used for sanity check only)
duration   = 10;         % record length [s]   (used for sanity check only)

% Optional: average only over a time window [tStart tEnd] in seconds,
% e.g. [1 9] to drop transients. Leave empty [] to use the whole record.
timeWindow = [];

outMat = fullfile(dataFolder, 'averaged_data.mat');
outCsv = fullfile(dataFolder, 'averaged_data.csv');
%% -----------------------------------------------

fileIdx  = (firstFile:lastFile).';
nFiles   = numel(fileIdx);
avgData  = [];                 % allocated after the first file is read
nSamples = zeros(nFiles, 1);
missing  = false(nFiles, 1);

for k = 1:nFiles
    fname = fullfile(dataFolder, sprintf('%d.txt', fileIdx(k)));

    if ~isfile(fname)
        warning('File not found, skipped: %s', fname);
        missing(k) = true;
        continue
    end

    % readmatrix auto-detects delimiter (tab, space, comma) and skips
    % text header lines. Non-numeric entries become NaN.
    data = readmatrix(fname, 'FileType', 'text');

    % Drop rows that are entirely NaN (e.g. blank lines / header remnants)
    data = data(~all(isnan(data), 2), :);

    % Allocate the output once we know the number of columns
    if isempty(avgData)
        nCols   = size(data, 2);
        avgData = NaN(nFiles, nCols);
        fprintf('Detected %d columns (1 time + %d channels).\n', nCols, nCols - 1);
    end

    if size(data, 2) ~= nCols
        warning('%s has %d columns (expected %d), skipped.', ...
                fname, size(data, 2), nCols);
        missing(k) = true;
        continue
    end

    % Optional time window based on column 1 (time)
    if ~isempty(timeWindow)
        t    = data(:, 1);
        data = data(t >= timeWindow(1) & t <= timeWindow(2), :);
    end

    nSamples(k) = size(data, 1);
    if isempty(timeWindow) && abs(nSamples(k) - fs*duration) > 0.01*fs*duration
        warning('%s has %d samples (expected ~%d).', ...
                fname, nSamples(k), fs*duration);
    end

    % Time average of every column -> 1 x nCols
    avgData(k, :) = mean(data, 1, 'omitnan');

    fprintf('Processed %s  (%d samples)\n', fname, nSamples(k));
end

if isempty(avgData)
    error('No files were read. Check dataFolder and file names.');
end

fprintf('\nDone: %d of %d files processed, %d missing/skipped.\n', ...
        nnz(~missing), nFiles, nnz(missing));

%% Save results
% .mat : full variables for later use in MATLAB
save(outMat, 'avgData', 'fileIdx', 'nSamples', 'missing', 'timeWindow');

% .csv : first column = file number, then the averaged columns
colNames = [{'File', 'Time_mean'}, ...
            arrayfun(@(c) sprintf('Ch%d', c), 1:size(avgData,2)-1, ...
                     'UniformOutput', false)];
T = array2table([fileIdx, avgData], 'VariableNames', colNames);
writetable(T, outCsv);

fprintf('Saved: %s\n       %s\n', outMat, outCsv);

%% Example: channel averages only (drop the time column)
% chanAvg = avgData(:, 2:end);   % nFiles x (nCols-1)
% plot(fileIdx, chanAvg, '-o'); xlabel('Condition (file #)'); ylabel('Mean value');
