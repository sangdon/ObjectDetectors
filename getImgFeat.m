function [ o_feat ] = getImgFeat( i_img, i_sqCellSz )
%GETIMGFEAT Summary of this function goes here
%   Detailed explanation goes here

nrows = floor(size(i_img, 1)/i_sqCellSz);
ncols = floor(size(i_img, 2)/i_sqCellSz);

img_gray = rgb2gray(i_img);
img_gray = img_gray(1:nrows*i_sqCellSz, 1:ncols*i_sqCellSz);
img_cell = mat2cell(img_gray, ones(1, nrows)*i_sqCellSz, ones(1, ncols)*i_sqCellSz);
for rInd=1:nrows
    for cInd=1:ncols
        img_cur = img_cell{rInd, cInd};
        img_cell{rInd, cInd} = mean(img_cur(:));
    end
end
o_feat = cell2mat(img_cell);

end

