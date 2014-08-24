function [ o_bbs ] = detect( i_params, i_img, i_mdl )
%DETECT Summary of this function goes here
%   Detailed explanation goes here
img = i_img;
i_objMdl = i_mdl;

%% build a feature pyramid
scales = i_params.test.scaleSearch;

if i_params.general.mdlType == 1
    featPyrScales = scales;
else
    featPyrScales = union(scales, scales*i_params.feat.partResRatio);
end
curMdl = i_objMdl;
curMdl.uv = [1; 1];
curMdl.wh = [size(img, 2); size(img, 1)];
featPyr = getFeatPyr( ...
    img, ...
    featPyrScales, ...
    @(img, s)(getAppFeat(i_params, struct('img', img, 'featPyr', []), curMdl, [1; 1; s; 1])));

%% search in the scale space
bbs = [];
imgSt = [];
imgSt.img = img;
imgSt.featPyr = featPyr;
for curScale=scales
    % find bounding boxes in the current scale space
    curBbs = slideWindow_mex(i_params, imgSt, updMdlUVSC(i_objMdl, [1; 1; curScale; 1]), zeros(4, 1));
    curBbs = normalizeBbs(i_params, i_objMdl, curBbs, curScale);
    curBbs = curBbs(curBbs(:, end)>i_params.test.scoreThres, :);
    bbs = [bbs; curBbs];
end
if ~isempty(bbs)
    bbs = sortrows(bbs, -size(bbs, 2));
    % nms
    if i_params.test.nms == 1;
        picked = nms(bbs, i_params.test.nmsOverlap);
        bbs = bbs(picked, :);
    end
end

%% return
o_bbs = bbs;

end

function [o_bbs] = normalizeBbs(i_params, i_mdl, i_bbs, i_scale)
% cell coordinate to image coordinate
% rescaling to the original image scale
o_bbs = i_bbs;
if isempty(o_bbs)
    return;
end

% root
o_bbs(:, 1:4) = cc2ic(o_bbs(:, 1:4)/i_scale, i_params.feat.HoG.SqCellSize);

% parts
next = 7;
for pInd=1:numel(i_mdl.parts)
    o_bbs(:, next:next+4-1) = cc2ic(o_bbs(:, next:next+4-1)/(i_scale*i_mdl.parts(pInd).ds), i_params.feat.HoG.SqCellSize);
    next = next + 6;
end

end


function [ o_bbs ] = cc2ic( i_bbs, i_sqCellSize )
%CC2IC cell coordinate to image coordinate
%   

o_bbs = [(i_bbs(:, [1 2])-1)*i_sqCellSize + 1, i_bbs(:, [3 4])*i_sqCellSize - 1];


% o_ic = (i_cc-1)*i_sqCellSize + i_sqCellSize/2 + 1;
% o_ic = (i_cc-1)*i_sqCellSize + 1;
end

