function [ o_feat ] = getHOGFeat( i_img, i_sqCellSize, i_type )
%GETHOGFEAT Summary of this function goes here
%   Detailed explanation goes here

% HOG
switch i_type
    case 1
%         o_feat = getFeature_HoG_DPM5(i_img, i_sqCellSize);
        o_feat = getFeature_HoG_vlfeat(i_img, i_sqCellSize);
    case 2
        o_feat = getFeature_HoG_DPM5_woCN(i_img, i_sqCellSize);
    case 3
        o_feat = getFeature_HoG_DPM5_woAN(i_img, i_sqCellSize);
%         o_feat = getFeature_HoG_vlfeat(i_img, i_sqCellSize);
end
% o_feat = getFeature_HoG_vlfeat(i_img, i_sqCellSize);
% o_feat = getFeature_HoG_vlfeat_INRIA(i_img, i_sqCellSize);
% o_feat = getFeature_HoG_DPM5(i_img, i_sqCellSize);
% o_feat = getFeature_HoG_DPM5_woCN(i_img, i_sqCellSize);

end


function [o_feat] = getFeature_HoG_vlfeat( i_im, i_sqCellSize)
im = im2single(i_im);
imhog = vl_hog(im, i_sqCellSize);
o_feat = double(imhog);
end

function [o_feat] = getFeature_HoG_vlfeat_INRIA( i_im, i_sqCellSize )
im = im2single(i_im);
imhog = vl_hog(im, i_sqCellSize, 'variant', 'dalaltriggs');
o_feat = double(imhog);
end

function [o_feat] = getFeature_HoG_DPM5( i_im, i_sqCellSize )
im = im2double(i_im);
imhog = features_wN(im, i_sqCellSize);
o_feat = double(imhog);
end


function [o_feat] = getFeature_HoG_DPM5_woCN( i_im, i_sqCellSize )
im = im2double(i_im);
imhog = features_woCN(im, i_sqCellSize);
o_feat = double(imhog);
end

function [o_feat] = getFeature_HoG_DPM5_woAN( i_im, i_sqCellSize )
im = im2double(i_im);
imhog = features_woAN(im, i_sqCellSize);
o_feat = double(imhog);
end