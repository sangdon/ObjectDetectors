function [ o_featPyr ] = getFeatPyr( i_img, i_scales, i_featFnc )
%GETFEATPRY Summary of this function goes here
%   Detailed explanation goes here

featPyr = [];
for sInd=1:numel(i_scales)
    s = i_scales(sInd);
    curFeatPyr = [];
    curFeatPyr.scale = s;
    curFeatPyr.feat = i_featFnc(imresize(i_img, s));
%     curFeatPyr.feat = getAppFeat(i_params, struct('img', i_img, 'featPyr', []), i_mdl, [1; 1; s; 1]);
    featPyr = [featPyr; curFeatPyr];
end
o_featPyr = featPyr;
end

