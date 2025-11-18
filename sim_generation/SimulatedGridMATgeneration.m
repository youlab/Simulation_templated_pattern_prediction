% Initialize parameters
numFiles = 1000; % Total number of files to process
simRows = 32; % Rows in simulated grid
simColumns = 37; % Columns in simulated grid

% Initialize the 3D matrix for storing simulated patterns
simPatterns = zeros(simRows, simColumns, numFiles);

% Directory containing the experimental pattern files
folderName = 'Experimental_Patterns';

% Pattern to match files starting with "Batch_" and ending with ".dl.txt"
pattern = fullfile(folderName, 'Batch_*.dl.txt');

% Get a list of files that match the pattern
fileList = dir(pattern);

names   = {fileList.name};
batchID = cellfun(@(s) sscanf(s, 'Batch_%d'), names);  % reads the number after "Batch_"
[~, order] = sort(batchID);
fileList = fileList(order);

% Loop through each file sequentially
for idx = 1:length(fileList)
    % Construct the filename assuming a straightforward enumeration
     filename = fullfile(folderName, fileList(idx).name);
    
    % Load the experimental pattern from the CSV file
    expPattern = readmatrix(filename);

    % Initialize a matrix for the current simulated pattern
    simPattern = zeros(simRows, simColumns);

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
                    simPattern(simRow, simCol) = 0.5;
                end
            end
        end
    end

    % Store the simulated pattern in the 3D matrix
    simPatterns(:, :, idx) = simPattern;
end

% Save the 3D matrix to a .mat file
save('simulatedPatterns_new1118.mat', 'simPatterns');
