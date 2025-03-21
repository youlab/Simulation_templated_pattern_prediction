
% Directory containing the experimental pattern files
folderName = 'C:\Users\kinsh\Documents\MATLAB\OptimalPatterns_NanLuo-master\OptimalPatterns_NanLuo-master\Pattern_prediction\Experimental_Patterns';

% Assuming folderName is defined as your directory path
folderName = 'Experimental_Patterns\Fixed_seeding';

% Pattern to match files
% pattern = fullfile(folderName, 'Batch_*.dl.txt');

% Get a list of files that match the pattern
% fileList = dir(pattern);

fileList=dir(folderName);
fileList = fileList(~[fileList.isdir]); % Filter out directories% here no patterns just processing all files 


% Initialize parameters
numFiles = length(fileList); % Total number of files to process
simRows = 32; % Rows in simulated grid
simColumns = 37; % Columns in simulated grid

% Initialize the 3D matrix for storing simulated patterns
simPattern_Fixeds = zeros(simRows, simColumns, numFiles);


% 
% % Extract batch numbers and sort files by them
% batchNumbers = zeros(length(fileList), 1);
% for i = 1:length(fileList)
%     % Extract batch number from each filename
%     % Assumption: filenames are in the form 'Batch_X_*.dl.txt'
%     tokens = regexp(fileList(i).name, 'Batch_(\d+)_.*\.dl\.txt', 'tokens');
%     if ~isempty(tokens)
%         batchNumbers(i) = str2double(tokens{1}{1});
%     end
% end

% Sort files by batch number
% [~, sortIdx] = sort(batchNumbers);


%Note that this is not intuititve sorting as it appears in windows ah
%before a
[~, sortIdx] = sort({fileList.name});  % sort alphabetically 
sortedFileList = fileList(sortIdx);


% Loop through each file sequentially
for idx = 1:numFiles
   disp(idx)
    filename = fullfile(folderName, sortedFileList(idx).name);
    % Load the experimental pattern from the CSV file
    
    disp("Processing file")
    disp(filename)
    expPattern = readmatrix(filename);
    
    disp(sum(sum(expPattern)))

    % Initialize a matrix for the current simulated pattern
    simPattern_Fixed = zeros(simRows, simColumns);

    % Loop through each cell in the experimental pattern
    for row = 1:size(expPattern, 1)
        for col = 1:size(expPattern, 2)
            if expPattern(row, col) == 0.1
                % Apply the shift to match simulated grid's center
                simRow = row - 2; % Adjust row
                simCol = col - 6; % Adjust column
                
                % Check if the shifted location is within the bounds of the simulated grid
                if simRow >= 1 && simRow <= simRows && simCol >= 1 && simCol <= simColumns
                    % Set the value to 0.5 for a seed at this location in the simulated pattern
                    simPattern_Fixed(simRow, simCol) = 0.5;
%                 elseif simRow<1 
                    
                end
            end
        end
    end

    % Store the simulated pattern in the 3D matrix
    simPattern_Fixeds(:, :, idx) = simPattern_Fixed;
end

% Save the 3D matrix to a .mat file
save('simulatedPatterns_Fixed.mat', 'simPattern_Fixeds');
