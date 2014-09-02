function [ o_feat ] = getHOGCFeat( i_img, i_sqCellSz )
%GETIMGFEAT Summary of this function goes here
%   Detailed explanation goes here

HOGFeat = getFeature_HoG_DPM5(i_img, i_sqCellSz);
HCFeat = getHCFeat_wrapper(i_img, i_sqCellSz);

nrows = min(size(HOGFeat, 1), size(HCFeat, 1));
ncols = min(size(HOGFeat, 2), size(HCFeat, 2));

HOGFeat = HOGFeat(1:nrows, 1:ncols, :);
HCFeat = HCFeat(1:nrows, 1:ncols, :);
o_feat = cat(3, HOGFeat, HCFeat);
end

function [o_feat] = getFeature_HoG_DPM5( i_im, i_sqCellSz )
im = im2double(i_im);
imhog = features_wN(im, i_sqCellSz);
o_feat = double(imhog);
end

function [o_feat] = getHCFeat_wrapper(i_im, i_sqCellSz)

nColBin = 8;

im = im2double(i_im);
colorTransForm = makecform('srgb2lab');
img_lab = applycform(im, colorTransForm);
img_lab(:, :, 1) = img_lab(:, :, 1)/100;
img_lab(:, :, 2) = (img_lab(:, :, 2)+128)/255;
img_lab(:, :, 3) = (img_lab(:, :, 3)+128)/255;

o_feat = getHCFeat(img_lab, i_sqCellSz, nColBin);
end