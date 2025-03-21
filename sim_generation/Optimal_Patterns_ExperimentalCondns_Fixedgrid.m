% Predicting colony patterns with various seeding configurations
% Generates results in Figure 6 and Figure S9

clear
% Load parameter set

%%%%%%%%%%%%%%%%
%%Random number shuffling for parallel computing 





  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    %%% 3 different conditions
    
 
    
    param_vector_1=[1,2,5.7486,195.4832,1.1050,0.6635,0.0789,4];
    param_vector_2=[1,3,5.7486,195.4832,1.1050,0.6635,0.0789,4];
    param_vector_3=[0,1,9.326,152.9,1.878,0.8466,0.06050,7.385];
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
      
%     taskID = str2num(getenv('SLURM_ARRAY_TASK_ID'));  % vary bw 1 to 100
taskID=1; 
    
    
for t=1:29 % since this varies between 1 to 10 total variation is bw 1 to 1000
      
    
  
    
    
    number= num2str(strcat(num2str(mod(floor(t/100),10)),num2str(mod(floor(t/10),10)),num2str(mod(t,10))));
    newnumber= str2double(strcat(num2str(taskID),num2str(number)));
    s1 = RandStream('mt19937ar', 'seed', sum(fix(clock))+newnumber); 
    
    
    RandStream.setGlobalStream(s1)
    
    
%     s1 = RandStream('mt19937ar','Seed',taskID);


    
    
    %%%%%%%%%%%%%%%%%%% Randomizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            size_x=32;
            size_y=37;
            Pattern=zeros(size_x,size_y);
            
%              pick a random number between 1 and 20
                        no_of_seeds=randi(s1,[1,20],1);
                        
                        exc_size=10;

        % [new_size_x,new_size_y]=deal(size_x-6,size_y-6);

        trunc_size_x=size_x-exc_size;
        trunc_size_y=size_y-exc_size;

        rand_mat_size=trunc_size_x*trunc_size_y;

        mat_trunc_flat=zeros(1,rand_mat_size);


        r_index=randsample(s1,1:rand_mat_size,no_of_seeds);

        mat_trunc_flat(r_index)=0.5;

        mat_trunc=reshape(mat_trunc_flat,trunc_size_x,trunc_size_y);

        Pattern(exc_size/2+1:size_x-exc_size/2,exc_size/2+1:size_y-exc_size/2)=mat_trunc;
       

            tag1='simple';
            tag2='intermediate';
            tag3='complex';
         
  
    
    Pattern_Generator(param_vector_2,Pattern,newnumber,tag2,t,taskID)
%     Pattern_Generator(param_vector_2,Pattern,newnumber,tag2)
%     Pattern_Generator(param_vector_3,Pattern,newnumber,tag3)
%     
 
    
end
 


function Pattern_Generator(param_vector,Pattern1,newnumber,tag,t,taskID)



load('Parameters_multiseeding.mat'); % select parameter file
config  = 17; % select seeding configuration (see below)
% NutrientLevel = 2;  % select nutrient level: 1-low, 2-medium, 3-high


load('simulatedPatterns_Fixed.mat')
% 
% % parameters change for normal patterns based on Nan's figures from MSB
% % paper
% DN=5.7486;
% bN=195.4832;
% aC=1.1050;
% KN=0.6635;
% Cm=0.0789;
% gama=4;

        param_vector=num2cell(param_vector);



       [Density_select,NutrientLevel,DN,bN,aC,KN,Cm,gama]=param_vector{:};




    % Obtain optimal W & D from the mapping
    N0      = N0s(NutrientLevel);
%     N0=4;
    Width   = interp1(mapping_N, mapping_optimW, N0, 'linear', 'extrap');
%     Width=0.4;


    if Density_select==1
        Density = interp1(mapping_N, mapping_optimD, N0, 'linear', 'extrap'); 
    else 
        Density=0.47;  % for complex patterns manually increasing density 
    end
    
    




% jobID= str2num(getenv('SLURM_ARRAY_JOB_ID'));


    
    
   
    
%     
%     NutrientLevel = 2;  % select nutrient level: 1-low, 2-medium, 3-high
% 
%     % parameters change for normal patterns based on Nan's figures from MSB
%     % paper
%     DN=5.7486;
%     bN=195.4832;
%     aC=1.1050;
%     KN=0.6635;
%     Cm=0.0789;
%     gama=4;



% taskID=1;


    
    % randomize 
    
    
   
%     Density=0.47;
    
    

    % ------------------------ Seeding configurations -------------------------
    switch config
        case 1; x0 = 0; y0 = 0; % one dot
        case 2; x0 = 17/2 * [-1, 1]; y0 = [0, 0]; % two dots side by side
        case 3; x0 = 38/2 * [-1, 1]; y0 = [0, 0]; % two dots side by side
        case 4; x0 = 19 * [-1, 0, 1]; y0 = [0, 0, 0]; % three dots side by side
        case 5; x0 = 10 * [0, sqrt(3)/2, -sqrt(3)/2]; y0 = 10 * [1, -0.5, -0.5]; % triangular
        case 6; x0 = 20 * [0, sqrt(3)/2, -sqrt(3)/2]; y0 = 20 * [1, -0.5, -0.5]; % triangular
        case 7; x0 = 15 * [-1, 1, 1, -1]; y0 = 15 * [1, 1, -1, -1]; % square
        case 8; x0 = 19 * [0, 0.5, 1, 0.5, -0.5, -1, -0.5]; % core-ring
                y0 = 19 * [0, sqrt(3)/2, 0, -sqrt(3)/2, -sqrt(3)/2, 0, sqrt(3)/2];
        case 9; x0 = 19 * [0, sqrt(2)/2, 1, sqrt(2)/2, 0, -sqrt(2)/2, -1, -sqrt(2)/2]; % ring
                y0 = 19 * [1, sqrt(2)/2, 0, -sqrt(2)/2, -1, -sqrt(2)/2, 0, sqrt(2)/2];
        case 10;x0 = 19 * [0, 0.3827, sqrt(2)/2, 0.9239, 1, 0.9239, sqrt(2)/2, 0.3827, 0, -0.3827, -sqrt(2)/2, -0.9239, -1, -0.9239, -sqrt(2)/2, -0.3827]; % ring
                y0 = 19 * [1, 0.9239, sqrt(2)/2, 0.3827, 0, -0.3827, -sqrt(2)/2, -0.9239, -1, -0.9239, -sqrt(2)/2, -0.3827, 0, 0.3827, sqrt(2)/2, 0.9239];
        case 11;x0 = [0, 0, 0, 0, 0, 0, 0, 0]; y0 = 6 * [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]; % line
        case 12;ld = load('DUKE.mat'); Pattern = ld.D;
        case 13;ld = load('DUKE.mat'); Pattern = ld.U;
        case 14;ld = load('DUKE.mat'); Pattern = ld.K;
        case 15;ld = load('DUKE.mat'); Pattern = ld.E;
        case 16; Pattern=Pattern1;
        case 17; Pattern= simPattern_Fixeds(:,:,t);
    




    end
    
    
    
   
   
   
   


    L      = 90;
    totalt = 24;
%     totalt=5;

    if config >= 12 && config <= 17
        Pattern = flipud(Pattern);  % this flips array upside down 
        [row,col] = find(Pattern > 0);  % returns the row and column values of non zero elements
        row = row - (size(Pattern, 1) + 1) / 2;  % restruct values from -size/2 to size/2
        col = col - (size(Pattern, 2) + 1) / 2;
        domainsize = 42;
        x0 = col' * L / domainsize;  % thus rescaling by a factor of L/domainsize
        y0 = row' * L / domainsize;  % this results in everything being less than domain size 
    end
    % -------------------------------------------------------------------------

    % Parameters
    % L      = 90;
    % totalt = 24;

    dt = 0.02;
    nt = totalt / dt;  % this is the number of steps 
    nx = 1001; ny = nx;
    dx = L / (nx - 1); dy = dx;   % 90/1000 0.09
    x  = linspace(-L/2, L/2, nx);  % -45 to 45,  1001 elements
    y  = linspace(-L/2, L/2, ny);
    [xx, yy] = meshgrid(x, y);% meshgrid 

    noiseamp = 0 * pi;

    % Initialization
    P = zeros(nx, ny);      % Pattern
    C = zeros(nx, ny);      % Cell density
    N = zeros(nx, ny) + N0; 

%     r0 = 5;    % initial radius 

    r0=5;
    C0 = 1.6;

    nseeding = length(x0);
    rr = zeros(nx, ny, nseeding);   % nd array of zeros
    for isd = 1 : nseeding
        rr(:,:,isd) = sqrt((xx - x0(isd)).^ 2 + (yy - y0(isd)) .^ 2);
    end
    rr = min(rr, [], 3);
    P(rr <= r0) = 1;
    C(P == 1) = C0 / (sum(P(:)) * dx * dy); C_pre = C;

    % calculate the actual length of boundary of each inoculum
    nseeding = length(x0);
    nseg = 50; seglength = 2 * pi * (r0) / nseg;   % modification made on 031424 added +3 REVERTED BACK
    theta = linspace(0, 2 * pi, nseg + 1)'; theta = theta(1 : nseg);
    colonyarray = polyshape(); % boundary of each colony
    for iseed = 1 : nseeding
        colony = polyshape(r0 * sin(theta) + x0(iseed), r0 * cos(theta) + y0(iseed));
        colonyarray(iseed) = colony;
    end
    colonyunion = union(colonyarray); % joined boundary of all colonies
    boundarylengths = zeros(nseeding, 1);
    for iseed = 1 : nseeding
        colonyboundary = intersect(colonyunion.Vertices, colonyarray(iseed).Vertices, 'rows');
        boundarylengths(iseed) = seglength * size(colonyboundary, 1);
    end
    % ------------------------------------------------------------------------

    ntips0 = ceil(boundarylengths * Density); % initial branch number
    theta = []; Tipx = []; Tipy = [];
    for iseed = 1 : nseeding
    Tipxi = ones(ntips0(iseed), 1) * x0(iseed);  Tipx = [Tipx; Tipxi]; % x coordinates of every tip
    Tipyi = ones(ntips0(iseed), 1) * y0(iseed);  Tipy = [Tipy; Tipyi]; % y coordinates of every tip
    thetai = linspace(pi/2, 2 * pi+pi/2, ntips0(iseed) + 1)'; 
    thetai = thetai(1 : ntips0(iseed)) + iseed /10 * pi; % growth directions of every branch
    theta = [theta; thetai];
    end
    ntips0 = sum(ntips0);

    dE = zeros(ntips0, 1);
    BranchDomain = cell(ntips0, 1); % the domain covered by each branch
    for k = 1 : ntips0; BranchDomain{k} = C > 0; end

    Biomass = sum(C(:)) * (dx * dy);
    delta = linspace(-1, 1, 201) * pi;
    [MatV1N,MatV2N,MatU1N,MatU2N] = Branching_diffusion(dx,dy,nx,ny,dt,DN);
%     
%      indices_array=zeros(3,nx,ny);
%      
%      initial_point= round(floor(nt/3),-1);
%      interm_point=round(floor(2*nt/3),-1);
%      final_point=round(nt,-1);
%      j=1; % index of the indices_Array list 
     

    for i = 0 : nt

        % -------------------------------------
        % Nutrient distribution and cell growth

        fN = N ./ (N + KN) .* Cm ./ (C + Cm) .* C;
        dN = - bN * fN;
        N  = N + dN * dt; 
        NV = MatV1N \ (N * MatU1N); N = (MatV2N * NV) / MatU2N;

        dC = aC * fN;
        C  = C + dC * dt; 

        % -------------------------------------
        % Branch extension and bifurcation
        ntips = length(Tipx);

        if mod(i, 0.2/dt) == 0

            dBiomass = (C - C_pre) * dx * dy; 
            % compute the amount of biomass accumulation in each branch
            BranchDomainSum = cat(3, BranchDomain{:});
            BranchDomainSum = sum(BranchDomainSum, 3);
            ntips = length(Tipx);
            for k = 1 : ntips
                branchfract = 1 ./ (BranchDomainSum .* BranchDomain{k}); 
                branchfract(isinf(branchfract)) = 0;
                dE(k) = sum(sum(dBiomass .* sparse(branchfract)));
            end

            % extension rate of each branch
            dl = gama * dE / Width;
            if i == 0; dl = 0.5; end

            % Bifurcation
            R = 1.5 / Density;  % a branch will bifurcate if there is no other branch tips within the radius of R
            TipxNew = Tipx; TipyNew = Tipy; thetaNew = theta; dlNew = dl;
            BranchDomainNew = BranchDomain;
            for k = 1 : ntips
                dist2othertips = sqrt((TipxNew - Tipx(k)) .^ 2 + (TipyNew - Tipy(k)) .^ 2);
                dist2othertips = sort(dist2othertips);
                if dist2othertips(2) > R
                    TipxNew = [TipxNew; Tipx(k) + dl(k) * sin(theta(k) + 0.5 * pi)]; % splitting the old tip to two new tips
                    TipyNew = [TipyNew; Tipy(k) + dl(k) * cos(theta(k) + 0.5 * pi)]; 
                    TipxNew(k) = TipxNew(k) + dl(k) * sin(theta(k) - 0.5 * pi);
                    TipyNew(k) = TipyNew(k) + dl(k) * cos(theta(k) - 0.5 * pi);
                    dlNew = [dlNew; dl(k) / 2];
                    dlNew(k) = dl(k) / 2;
                    thetaNew = [thetaNew; theta(k)];
                    BranchDomainNew{end+1} = BranchDomain{k};
                end
            end
            Tipx = TipxNew; Tipy = TipyNew; theta = thetaNew; dl = dlNew;
            BranchDomain = BranchDomainNew;

            ntips = length(Tipx);
            % Determine branch extension directions
            Tipx_pre = Tipx; Tipy_pre = Tipy;
            if i == 0
                Tipx = Tipx + dl .* sin(theta);
                Tipy = Tipy + dl .* cos(theta);
            else
                thetaO = ones(ntips, 1) * delta;
                TipxO = Tipx + dl .* sin(thetaO);
                TipyO = Tipy + dl .* cos(thetaO);
                NO = interp2(xx, yy, N, TipxO, TipyO);
                [~, ind] = max(NO, [], 2); % find the direction with maximum nutrient
                for k = 1 : ntips
                    Tipx(k) = TipxO(k, ind(k));
                    Tipy(k) = TipyO(k, ind(k));
                    theta(k) = thetaO(k, ind(k));
                end
            end

            % Growth stops when approaching edges
            ind = sqrt(Tipx.^2 + Tipy.^2) > 0.8 * L/2;
            Tipx(ind) = Tipx_pre(ind);
            Tipy(ind) = Tipy_pre(ind);

            % Fill the width of the branches
            for k = 1 : ntips
                d = sqrt((Tipx(k) - xx) .^ 2 + (Tipy(k) - yy) .^ 2);
                P(d <= Width/2) = 1;
                BranchDomain{k} = BranchDomain{k} | (d <= Width/2); 
            end
            C(P == 1) = sum(C(:)) / sum(P(:)); % Make cell density uniform
            C_pre = C;
            
            % 
            
           

            
            
            %clf; 
            ind = 1 : 2 : nx;
            
            % this might be ineffecient 
            
%             if (i==initial_point || i==interm_point || i==final_point)
%                 indices_array(j,:,:)=C;
%                 j=j+1;
%             end
%             
            
            
%             
%             %%%%%%%%%%comment this section if looping 
%             subplot 121
%                 pcolor(xx(ind, ind), yy(ind, ind), C(ind, ind)); shading interp; axis equal;
%                 axis([-L/2 L/2 -L/2 L/2]); colormap('gray'); hold on
%                 set(gca,'YTick',[], 'XTick',[])
%                 plot(Tipx, Tipy, '.', 'markersize', 5)
%             subplot 122
%                 pcolor(xx(ind, ind), yy(ind, ind), N(ind, ind)); shading interp; axis equal;
%                 axis([-L/2 L/2 -L/2 L/2]); set(gca,'YTick',[], 'XTick',[]);  
%                 colormap('parula'); caxis([0 N0])
%             drawnow
% % 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end

    end

    %
% 

    %%%%%%%
    % Storing the output

%     for i=1:size(indices_array,1)
    % 
    % t=1;
%         C=reshape(indices_array(i,:,:),nx,ny);
    % 
        f1=figure();
        hold on
        pcolor(xx(ind, ind), yy(ind, ind), C(ind, ind)); shading interp; axis equal;
        axis([-L/2 L/2 -L/2 L/2]); colormap('gray');
                    set(gca,'YTick',[], 'XTick',[])


        H = getframe(gca);
        close(f1)

        % note we create both the Sim_input and Sim_output folder ourselves 
       base_name_output='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Sim_output';
       
%         base_name_output='C:\Users\kinsh\Documents\MATLAB\OptimalPatterns_NanLuo-master\OptimalPatterns_NanLuo-master\Pattern_prediction\Sim_031424';
       
       
%             base_name_output='C:\Users\kinsh\Documents\MATLAB\OptimalPatterns_NanLuo-master\OptimalPatterns_NanLuo-master\Pattern_prediction\Sim_input_simple';
%         base_name_timepoint=strcat('Tp',num2str(i));

          base_name_complexity=tag;
          base_name_iteration=t;
        
%         base_name_output=fullfile(base_name_output,'Sim_input',base_name_iteration);

        % Create Sim_output folder if it doesn't exist
        if ~isfolder(base_name_output)
        mkdir(base_name_output);
        end


        time_time=clock;
        time_day=time_time(3);
        time_month=time_time(2);


    %     base_name_output='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_040423_100pointgrid/Sim_input'; 
        filename_output=strcat('Output','_','Fixed','_',num2str(base_name_iteration),'.png')  ; % here t is the value of iteration 
        fullfilename_output=fullfile(base_name_output,filename_output);

        imwrite(H.cdata, fullfilename_output)
    % imwrite(H.cdata, filename_output)

        % A=imread('test.png');
        % r
        % size(A)

        %%%%%%%%%
        % store the input



        input=flipud(Pattern);
        input=uint8(input);

        % figure()
        % imshow(double(input))
%         base_name_input='C:\Users\kinsh\Documents\MATLAB\OptimalPatterns_NanLuo-master\OptimalPatterns_NanLuo-master\Pattern_prediction\Sim_output_simple';
       
        
%         base_name_output=fullfile(base_name_input,base_name_timepoint);
  
    
    
        base_name_input='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Sim_output';
%           base_name_input='C:\Users\kinsh\Documents\MATLAB\OptimalPatterns_NanLuo-master\OptimalPatterns_NanLuo-master\Pattern_prediction\Sim_031424\Sim_output';

        % Create Sim_output folder if it doesn't exist
        if ~isfolder(base_name_input)
        mkdir(base_name_input);
        end



        filename_input=strcat('Input','_','Fixed','_',num2str(base_name_iteration),'.png'); %'_', ,num2str(time_month),'_',num2str(time_day)
        fullfilename_input=fullfile(base_name_input,filename_input);

        imwrite(double(input),fullfilename_input)
    %       imwrite(double(input),filename_input)
    
end








% end
