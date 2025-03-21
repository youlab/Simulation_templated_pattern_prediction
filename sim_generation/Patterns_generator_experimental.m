% here we will generate random seeds for experimental patterns

% code to generate random seed 

row=32;
columns=48;


[center_x,center_y]=deal(18,25) ;% center coordinates of circle, having dimensions of plate


% this is the total size of the grids

well_size=2.5; % in mm

plate_radius=44; 


grid_radius=round(0.8*(plate_radius/well_size));  % 80 percent of the plate will be used. 
% make square grid first based on the plate radius 

grid_length_half=plate_radius/well_size;

% from that grid length go into grid_length half on two sides
% from one side it is already less than a circle 


start_index_col= round(center_y-(grid_length_half-1));
end_index_col=round(center_y+(grid_length_half+1)); 


%%
% Create the directory if it doesn't exist
folderName = 'Experimental_Patterns';
if ~exist(folderName, 'dir')
    mkdir(folderName);
end


% Initialize an array to keep track of the number of dots per plate
numDotsPerPlate = zeros(1, 1000);

% Initialize a matrix to accumulate the positions of the dots
cumulativeDotsMatrix = zeros(row, columns);


total_random_mat=zeros(1,1000);


for batch = 1:1000



MAT=zeros(32,48);
% make a list of usable grids that are within 80% of the colony dimensions

% points are within radius if (x-X)^2 + (y-Y)^2=r^2

     
%               s1 = RandStream('mt19937ar', 'seed', 2023); 
%     
%     
%                 RandStream.setGlobalStream(s1)
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
            % pick a random number between 1 and 10
            r=randi([1,10],1);
            total_random_mat(batch)=r;
            % picks r random numbers for both row and column
            % for row picks between 1 and 32 
            % for row picks between 1 and 37
%             r_index=randsample(32,r);
%             c_index=randsample(37,r);

              pop_r=1:32;
              pop_c=start_index_col:end_index_col;
              
         
              r_index=randsample(pop_r,r);
              c_index=randsample(pop_c,r);
              
              
              
             % code for excluding points
             % on the vertical direction, number of rows are lower than the
             % plate size so no truncating 
             
  % Initialize arrays to store the valid indices
valid_r_indices = [];
valid_c_indices = [];
           
             
             
for i = 1:length(c_index)
    % Calculate the distance from the center
            dist_from_center = sqrt((r_index(i) - center_x)^2 + (c_index(i) - center_y)^2);

            % Check if the point is within the radius of the circle
            if dist_from_center <= grid_radius
                valid_r_indices = [valid_r_indices, r_index(i)];
                valid_c_indices = [valid_c_indices, c_index(i)];

            end
end



while length(valid_r_indices) < r
    % Calculate how many more points are needed
    points_needed = r - length(valid_r_indices);
    
    % Generate new random indices
    new_r_index = randsample(pop_r, points_needed);
    new_c_index = randsample(pop_c, points_needed);
    
    for i = 1:length(new_c_index)
        % Calculate grid coordinates relative to center
       % Calculate the distance from the center
            dist_from_center = sqrt((new_r_index(i) - center_x)^2 + (new_c_index(i) - center_y)^2);

            % Check if the point is within the radius of the circle
            if dist_from_center <= grid_radius
                valid_r_indices = [valid_r_indices, new_r_index(i)];
                valid_c_indices = [valid_c_indices, new_c_index(i)];
            end
            
    end
end


% Fill the matrix with 0.1 at the chosen indices
for i = 1:length(valid_r_indices)
    MAT(valid_r_indices(i), valid_c_indices(i)) = 0.1;
end



 % Update numDotsPerPlate
    numDotsPerPlate(batch) = r;

 % Update cumulativeDotsMatrix
    cumulativeDotsMatrix = cumulativeDotsMatrix + (MAT == 0.1);

 % File name for each batch, saved in the Experimental_Patterns folder
    filename = fullfile(folderName, sprintf('Batch_%d_%d.dl.txt', batch,r));

    % Open file for writing
    fileID = fopen(filename, 'w');
 % Use \r\n for CRLF line endings in the header
    fprintf(fileID, '[ Version: 6 ]\r\n');
    fprintf(fileID, 'agar plate ks.pd.txt\r\n');
    fprintf(fileID, '0\r\n');
    fprintf(fileID, '1\r\n');
    fprintf(fileID, '1\t0\t\r\n');
    fprintf(fileID, '0\r\n');
    fprintf(fileID, 'ES_HD\t\t1 cP\r\n');
    fprintf(fileID, 'Well\t1\r\n');

    % Adjust fprintf within your matrix writing loop to use CRLF
    for i = 1:size(MAT, 1)
        for j = 1:size(MAT, 2)
            if MAT(i, j) == 0
                fprintf(fileID, '0\t');
            else
                fprintf(fileID, '%.1f\t', MAT(i, j));
            end
        end
        fprintf(fileID, '\r\n'); % Ensure to use CRLF here
    end

    % Close the file
    fclose(fileID);
end


   
figure;
histogram(total_random_mat, 10); % There are 10 possible values (1 to 10)
title('Histogram of Number of Points per Matrix');
xlabel('Number of Points');
ylabel('Frequency');

% Create heat map
figure;
imagesc(cumulativeDotsMatrix);
colormap('hot'); % or choose another colormap as per your preference
colorbar;
title('Heat Map of Cumulative Dots on a 2D Grid');
xlabel('Column Index');
ylabel('Row Index');
            
        


