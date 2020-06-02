%% set parameters
% folderpath = '.\train_icvl\Depth\';
% filepath = '.\train_icvl\labels.txt';
% frameNum = 331006;

% folderpath = '/media/windows/data/data/ICVL/Testing/depth/';
% filepath = '.\test_icvl\icvl_test_list.txt';
% frameNum = 702+894;

folderpath = '/media/windows/data/data/ICVL/Testing/Depth/';
filepath = '/media/windows/data/data/ICVL/Testing/test_seq_1.txt';
frameNum = 702;

% due to test images are separated into two folders, we should process them
% and put them into one folder
save_dir = '/media/windows/data/data/ICVL/Testing/processed_seq/';

fp = fopen(filepath);
fid = 1;

tline = fgetl(fp);
while fid <= frameNum
    
    splitted = strsplit(tline);
    img_name = splitted{1};
    
    if exist(strcat(folderpath,img_name), 'file')
        img = imread(strcat(folderpath,img_name));
       
        fp_save = fopen(strcat(folderpath,img_name(1:size(img_name,2)-3),'bin'),'w');
        fwrite(fp_save,permute(img,[2,1,3]),'float');
        fclose(fp_save);
        
        save('-v6', fullfile(save_dir, strcat(num2str(fid), '.mat')),'img');
        
        %delete(strcat(folderpath,img_name));
    end

    tline = fgetl(fp);
    if isempty(tline)
        tline = fgetl(fp);
    end
    fid = fid + 1;
end

fclose(fp);







