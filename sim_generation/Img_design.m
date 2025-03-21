% script to test 8 bit scale images for patterns

% files=dir('C:\Users\kinsh\Downloads\Designs\selected_designs\*.png');
% 
% script to test 8 bit scale images for patterns

files=dir('C:\Users\kinsh\Downloads\Designs\new_design\*.png');
MAT=zeros(32,37,numel(files));


for i = 1:numel(files)  % loop for each file 
    file = files(i);
   filename = fullfile(file.folder,file.name);

   disp(filename);

   img = imread(filename);
   img=im2gray(img);
   img_double=im2double(img);
   
   MAT(:,:,i)=img_double;
   
end
 
save('Designs.mat')