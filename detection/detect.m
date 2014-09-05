function [ o_bbs, o_bbs_wbg ] = detect( i_mdl, i_img  )
%DETECT Summary of this function goes here
%   Detailed explanation goes here
img = i_img;
objMdl = i_mdl.objMdl;
i_params = i_mdl.params;
sqCellSz = i_params.feat.HOX.SqCellSize;
HOGType = i_params.feat.HOX.type;
nmsFlag = i_params.test.nms;
nmsOverlap = i_params.test.nmsOverlap;
scoreThres = i_params.test.scoreThres;
bgContextSz = i_params.test.bgContextSz;
interval = i_params.test.interval;

%% build a feature pyramid
if interval == 0
    feats = {getHOXFeat(img, sqCellSz, HOGType)};
    scales = 1;
else
    [feats, scales] = featpyramid(img, sqCellSz, interval, @(img) getHOXFeat(img, sqCellSz, HOGType));
end
featPyr = struct('scale', [mat2cell(scales, ones(numel(scales), 1), 1)], 'feat', feats);

% featPyr = [];
% for sInd=1:numel(scales)
%     curFeat = [];
%     curFeat.scale = scales(sInd);
%     curFeat.feat = padarray(feats{sInd}, [objMdl.wh_cc(2)-1 objMdl.wh_cc(1)-1 0]);
%     featPyr = [featPyr; curFeat];
% end

%% FIXME: padarray??

%% FIXME: how about part? is there scale*2 things?

%% search in the scale space
bbs = [];
bbs_wbg = [];
imgSt = [];
imgSt.img = img;
imgSt.featPyr = featPyr;
for i=1:numel(scales)
    curScale = scales(i);
    
    % find bounding boxes in the current scale space
    if i_params.general.mdlType == 2
        curBbs = slideWindow_DPM_mex(i_params.general.mdlType, sqCellSz, HOGType, imgSt, updMdlUVSC(objMdl, [1; 1; curScale; 1]), zeros(4, 1));
    
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

        % remove bg context of roots
        bbs_wbg = [bbs_wbg; curBbs];
        if bgContextSz > 0
            padding = objMdl.wh - objMdl.wh/(1+bgContextSz);
    %         padWH = i_mdl.wh*(1-i_params.test.bgContextSz);
            curBbs(:, 1) = (curBbs(:, 1)*curScale + padding(1)/2)/curScale;
            curBbs(:, 2) = (curBbs(:, 2)*curScale + padding(2)/2)/curScale;

            curBbs(:, 3) = (curBbs(:, 3)*curScale - padding(1)/2)/curScale;
            curBbs(:, 4) = (curBbs(:, 4)*curScale - padding(2)/2)/curScale;
        end

        bbs = [bbs; curBbs];
    else
        % obtain bounding boxes
%         mdlMat = reshape(objMdl.w_app, [objMdl.wh_cc(2) objMdl.wh_cc(1) size(featPyr(i).feat, 3)]);
%         curBbs = slideWindow_conv( mdlMat, featPyr(i).feat, scoreThres );
        curBbs = slideWindow_conv( objMdl, featPyr, scoreThres );
        
        % cell coord to pixel coord and rescale to original scale
        curBbs = normalizeBbs(sqCellSz, objMdl, curBbs, curScale);
        
        % remove bg context of roots
        bbs_wbg = [bbs_wbg; curBbs];
        if bgContextSz > 0
            padding = objMdl.wh - objMdl.wh/(1+bgContextSz);
    
            curBbs(:, 1) = (curBbs(:, 1)*curScale + padding(1)/2)/curScale;
            curBbs(:, 2) = (curBbs(:, 2)*curScale + padding(2)/2)/curScale;
            curBbs(:, 3) = (curBbs(:, 3)*curScale - padding(1)/2)/curScale;
            curBbs(:, 4) = (curBbs(:, 4)*curScale - padding(2)/2)/curScale;
        end

        bbs = [bbs; curBbs];
    end
    
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
o_bbs(:, 1:4) = o_bbs(:, 1:4)*i_sqCellSz/i_scale;
% parts
next = 7;
for pInd=1:numel(i_mdl.parts)
    o_bbs(:, next:next+4-1) = o_bbs(:, next:next+4-1)*i_sqCellSz/(i_scale*i_mdl.parts(pInd).ds);
    next = next + 6;
end

end

function [o_bbs] = slideWindow_conv( i_objMdl, i_featPyr, i_scoreThres ) 
% save the bb: [xmin ymin xmax ymax appScore defScore]

%% precompute
scales = [i_featPyr(:).scale];

% obtain filter responses
resp = getAppFilterResp(i_featPyr, o_mdl);

% obtain scores for each root responses
for lInd=1:size(resp, 2)
    rootResp = resp(1, lInd);
    val = rootResp > i_scoreThres;
    [rows, cols] = find(val);
    x = cols(:);
    y = rows(:);
    s = rootResp(val); s = s(:);
    

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find root candidates
rootResp = resp


rootMdl = reshape(i_objMdl.w_app, [i_objMdl.wh_cc(2) i_objMdl.wh_cc(1) size(inMat, 3)]);
rootMdlWH = [size(rootMdl, 2); size(rootMdl, 1)];
% convolve the root

rootResp = convn(inMat, flipdim(flipdim(flipdim(rootMdl, 1), 2), 3), 'valid');


% generate bbs
val = rootResp > i_scoreThres;
[rows, cols] = find(val);
x = cols(:);
y = rows(:);
s = rootResp(val); s = s(:);

o_bbs = [x y x+rootMdlWH(1) y+rootMdlWH(2) s s];
end

function [ o_resp ] = getAppFilterResp( i_featPyr, i_mdl )
%GETAPPSPAFILTERRESP Summary of this function goes here
%   Detailed explanation goes here

map_IDTI = i_mdl.map_IDTI;
nAllParts = size(map_IDTI, 2);
nFeatLevel = numel(i_featPyr);

o_resp = cell(nAllParts, nFeatLevel);
for i=1:nAllParts
    curMdl = getNode(map_IDTI(:, i), i_mdl);
    filterSize = [curMdl.wh_cc(2) curMdl.wh_cc(1) curMdl.appFeatDim/prod(curMdl.wh_cc)];
    curFilter = reshape(curMdl.w_app, filterSize); %%FIXME: inefficient
    curFilter = flipdim(flipdim(flipdim(curFilter, 1), 2), 3);
    
    for l=1:nFeatLevel  
        % get responses
        o_resp{i, l} = convn(i_featPyr(l).feat, curFilter, 'valid');
    end
end
end

% function [o_bbs] = slideWindow_conv( i_objMdl, i_inMat, i_scoreThres ) 
% % save the bb: [xmin ymin xmax ymax appScore defScore]
% 
% %% precompute
% rootMdl = reshape(i_objMdl.w_app, [i_objMdl.wh_cc(2) i_objMdl.wh_cc(1) size(i_inMat, 3)]);
% rootMdlWH = [size(rootMdl, 2); size(rootMdl, 1)];
% % convolve the root
% 
% rootResp = convn(i_inMat, flipdim(flipdim(flipdim(rootMdl, 1), 2), 3), 'valid');
% 
% 
% % generate bbs
% val = rootResp > i_scoreThres;
% [rows, cols] = find(val);
% x = cols(:);
% y = rows(:);
% s = rootResp(val); s = s(:);
% 
% o_bbs = [x y x+rootMdlWH(1) y+rootMdlWH(2) s s];
% end
