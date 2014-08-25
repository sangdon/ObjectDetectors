function [ o_feat ] = getHOGCFeat( i_img, i_sqCellSz )
%GETIMGFEAT Summary of this function goes here
%   Detailed explanation goes here

HOGFeat = getFeature_HoG_DPM5(i_img, i_sqCellSz);
ColFeat = getColorFeat(i_img, i_sqCellSz, size(HOGFeat, 1), size(HOGFeat, 2));
o_feat = cat(3, HOGFeat, ColFeat);
end

function [o_feat] = getFeature_HoG_DPM5( i_im, i_sqCellSize )
im = im2double(i_im);
imhog = features_wN(im, i_sqCellSize);
o_feat = double(imhog);
end

function [o_feat] = getColorFeat(i_im, i_sqCellSize, nrows, ncols)

nColBin = 8;

im = im2double(i_im);
colorTransForm = makecform('srgb2lab');
img_lab = applycform(im, colorTransForm);
img_lab(:, :, 1) = img_lab(:, :, 1)/100;
img_lab(:, :, 2) = (img_lab(:, :, 2)+128)/255;
img_lab(:, :, 3) = (img_lab(:, :, 2)+128)/255;

% nrows = floor(size(i_img, 1)/i_sqCellSz);
% ncols = floor(size(i_img, 2)/i_sqCellSz);
feat = mat2cell(img_lab, ...
    [i_sqCellSize*ones(1, nrows-1) size(im, 1)-i_sqCellSize*(nrows-1)], ...
    [i_sqCellSize*ones(1, ncols-1) size(im, 2)-i_sqCellSize*(ncols-1)], ...
    [1 1 1]);
for k=1:size(feat, 3)
    for i=1:size(feat, 1)
        for j=1:size(feat, 2)
            feat{i, j, k} = histc(feat{i, j, k}(:), 0:1/nColBin:1); 
            feat{i, j, k} = reshape(feat{i, j, k}, [1 1 numel(feat{i, j, k})]);
            % color normalization????????????????????????????
        end
    end
end
o_feat = cell2mat(feat);
end