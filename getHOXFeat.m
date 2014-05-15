function [ o_feat ] = getHOXFeat( i_img, i_sqCellSize, i_type )
%GETHOXFEAT Summary of this function goes here
%   Detailed explanation goes here

switch i_type
    case 1
        o_feat = getHOGFeat( i_img, i_sqCellSize, i_type );
    case 2
        o_feat = getHOGFeat( i_img, i_sqCellSize, i_type );
    case 3
        o_feat = getHOGFeat( i_img, i_sqCellSize, i_type );
    case 4
        o_feat = getHObCFeat(i_img, i_sqCellSize);
    case 5
        o_feat = getHObCN0Feat(i_img, i_sqCellSize);
    case 6
        o_feat = getImgFeat(i_img, i_sqCellSize);
end
end

