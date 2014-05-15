function [ o_bbs, o_bbs_wbg ] = detect( i_mdl, i_img  )
%DETECT Summary of this function goes here
%   Detailed explanation goes here
img = i_img;
objMdl = i_mdl.objMdl;
i_params = i_mdl.params;
sqCellSz = i_params.feat.HoG.SqCellSize;
HOGType = i_params.feat.HoG.type;
nmsFlag = i_params.test.nms;
nmsOverlap = i_params.test.nmsOverlap;
scoreThres = i_params.test.scoreThres;
bgContextSz = i_params.test.bgContextSz;
interval = i_params.test.interval;

%% build a feature pyramid
% fpTID = tic;
if interval == 0
    feats = {getHOXFeat(img, sqCellSz, HOGType)};
    scales = 1;
else
    [feats, scales] = featpyramid(img, sqCellSz, interval, @(img) getHOXFeat(img, sqCellSz, HOGType));
    
    feats = feats(1:2);
    scales = scales(1:2);
end

featPyr = [];
for sInd=1:numel(scales)
    curFeat = [];
    curFeat.scale = scales(sInd);
    curFeat.feat = padarray(feats{sInd}, [objMdl.wh_cc(2)-1 objMdl.wh_cc(1)-1 0]);
    featPyr = [featPyr; curFeat];
end

% fprintf('- building feature pyramid takes %s sec.\n', num2str(toc(fpTID)));

% how about part? is there scale*2 things?

%% search in the scale space
bbs = [];
bbs_wbg = [];
imgSt = [];
imgSt.img = img;
imgSt.featPyr = featPyr;
for curScale=scales(:)'
    % find bounding boxes in the current scale space
    curBbs = slideWindow_mex(sqCellSz, HOGType, imgSt, updMdlUVSC(objMdl, [1; 1; curScale; 1]), zeros(4, 1));
    
    % remove the effect of the feature padding
    curBbs(:, [1 3]) = curBbs(:, [1 3]) - (objMdl.wh_cc(1) - 1);
    curBbs(:, [2 4]) = curBbs(:, [2 4]) - (objMdl.wh_cc(2) - 1);
    
    % cell coord to pixel coord and rescale to original scale
    curBbs = normalizeBbs(sqCellSz, objMdl, curBbs, curScale);
    
    % discard bbs that exceed image space (only considers for root locations)
    thres = 0.8;
    bbW = (curBbs(:, 3) - curBbs(:, 1));
    bbH = (curBbs(:, 4) - curBbs(:, 2));
    bbW_in = (min(size(img, 2), curBbs(:, 3)) - max(1, curBbs(:, 1)));
    bbH_in = (min(size(img, 1), curBbs(:, 4)) - max(1, curBbs(:, 2)));
    invalInd = (bbW_in.*bbH_in)./(bbW.*bbH) < thres;
    curBbs(invalInd, :) = [];
    
    % discard bbs that exceed the score threshold
    curBbs = curBbs(curBbs(:, end)>scoreThres, :);

    
    bbs_wbg = [bbs_wbg; curBbs];
    % remove bg context of roots
    if bgContextSz > 0
        padding = objMdl.wh - objMdl.wh/(1+bgContextSz);
%         padWH = i_mdl.wh*(1-i_params.test.bgContextSz);
        curBbs(:, 1) = (curBbs(:, 1)*curScale + padding(1)/2)/curScale;
        curBbs(:, 2) = (curBbs(:, 2)*curScale + padding(2)/2)/curScale;
        
        curBbs(:, 3) = (curBbs(:, 3)*curScale - padding(1)/2)/curScale;
        curBbs(:, 4) = (curBbs(:, 4)*curScale - padding(2)/2)/curScale;
        
%         curBbs(:, [1 2]) = (curBbs(:, [1 2])*curScale + i_params.test.bgContextSz)/curScale;
%         curBbs(:, [3 4]) = (curBbs(:, [3 4])*curScale - i_params.test.bgContextSz)/curScale;
    end
    
    bbs = [bbs; curBbs];
    
end

if ~isempty(bbs)
    [bbs, IND] = sortrows(bbs, -size(bbs, 2));
    bbs_wbg = bbs_wbg(IND, :);
    % nms
    if nmsFlag == 1;
        picked = nms(bbs, nmsOverlap);
        bbs = bbs(picked, :);
        bbs_wbg = bbs_wbg(picked, :);
    end
end

%% return
o_bbs = bbs;
o_bbs_wbg = bbs_wbg;


end

function [o_bbs] = normalizeBbs(i_sqCellSz, i_mdl, i_bbs, i_scale)
% cell coordinate to image coordinate
% rescaling to the original image scale
o_bbs = i_bbs;
if isempty(o_bbs)
    return;
end

% root
% o_bbs(:, 1:4) = cc2ic(o_bbs(:, 1:4)/i_scale, i_params.feat.HoG.SqCellSize);
o_bbs(:, 1:4) = o_bbs(:, 1:4)*i_sqCellSz/i_scale;

% parts
next = 7;
for pInd=1:numel(i_mdl.parts)
%     o_bbs(:, next:next+4-1) = cc2ic(o_bbs(:, next:next+4-1)/(i_scale*i_mdl.parts(pInd).ds), i_params.feat.HoG.SqCellSize);
    o_bbs(:, next:next+4-1) = o_bbs(:, next:next+4-1)*i_sqCellSz/(i_scale*i_mdl.parts(pInd).ds);
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

