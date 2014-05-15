function [ o_bbs, o_bbs_wbg ] = detect_SS( i_params, i_img, i_mdl, i_cacheID )
%DETECT Summary of this function goes here
%   Detailed explanation goes here
img = i_img;
sqCellSz = i_params.feat.HoG.SqCellSize;

%% obtain candidate windows
candiObj = getCandiWindow(img, i_mdl, sqCellSz);
% candiObj = candiObj(10:11, :);
nCandiObj = size(candiObj, 1);
if i_params.debug.verbose >= 2
    fprintf('- the number of candidate windows: %d\n', nCandiObj);
end

%% build feature pyramid, dummy for fair comparison
[feats, scales] = featpyramid(img, i_params.feat.HoG.SqCellSize, i_params.test.interval, @(img) getHOXFeat(img, i_params.feat.HoG.SqCellSize, 1));

featPyr = [];
for sInd=1:numel(scales)
    curFeat = [];
    curFeat.img = i_img;
    curFeat.scale = scales(sInd);
    curFeat.feat = padarray(feats{sInd}, [i_mdl.wh_cc(2)-1 i_mdl.wh_cc(1)-1 0]);
    featPyr = [featPyr; curFeat];
end

%% eval score
nLocalIter = 10;

bbs = zeros(nCandiObj, size(candiObj, 2)+2);
bbs_wbg = zeros(size(bbs));
invalInd = false(nCandiObj, 1);
scales = [featPyr(:).scale];
for bInd=1:size(bbs, 1)
    curBB = candiObj(bInd, :);
        
    % find corresponding (l, u, v)
    [distVal, curScaleInd] = min(abs(scales-curBB(end)));
    if distVal>0.1
        invalInd(bInd) = false;
        continue;
    end
    
    curScale = scales(curScaleInd);
    uv_cc = round(curBB(1:2)*curScale/sqCellSz)' + i_mdl.wh_cc - 1;
    luv_cc = [curScaleInd; uv_cc];
    
    % get a score
    score = getScore_cc(luv_cc, featPyr, i_mdl, i_params);
    
    for li=1:nLocalIter
        
        luv_cc1 = moveScale(luv_cc, featPyr, 1);
        luv_cc2 = moveScale(luv_cc, featPyr, -1);
        luv_cc3 = luv_cc + [0; 1; 0];
        luv_cc4 = luv_cc + [0; -1; 0];
        luv_cc5 = luv_cc + [0; 0; 1];
        luv_cc6 = luv_cc + [0; 0; -1];

        % get a score
        score1 = getScore_cc(luv_cc1, featPyr, i_mdl, i_params);
        score2 = getScore_cc(luv_cc2, featPyr, i_mdl, i_params);
        score3 = getScore_cc(luv_cc3, featPyr, i_mdl, i_params);
        score4 = getScore_cc(luv_cc4, featPyr, i_mdl, i_params);
        score5 = getScore_cc(luv_cc5, featPyr, i_mdl, i_params);
        score6 = getScore_cc(luv_cc6, featPyr, i_mdl, i_params);
        
        [score, maxInd] = max([score, score1, score2, score3, score4, score5, score6]);
        if maxInd == 1
            break;
        else
            eval(sprintf('luv_cc = luv_cc%d;', maxInd-1));
        end
    end
    if i_params.debug.verbose >= 2
        fprintf('- local search: %d/%d\n', li, nLocalIter);
    end
    curScale = scales(luv_cc(1));
    uvwh = [(luv_cc(2:3)-(i_mdl.wh_cc - 1))' i_mdl.wh_cc']*sqCellSz/curScale;
    
    % bb in image space
    curBb = [uvwh(1:2) uvwh(1:2)+uvwh(3:4) 0 0 score];

    % discard bbs that exceed image space (only considers for root locations)
    thres = 0.8;
    bbW = (curBb(:, 3) - curBb(:, 1));
    bbH = (curBb(:, 4) - curBb(:, 2));
    bbW_in = (min(size(img, 2), curBb(:, 3)) - max(1, curBb(:, 1)));
    bbH_in = (min(size(img, 1), curBb(:, 4)) - max(1, curBb(:, 2)));
    if (bbW_in.*bbH_in)/(bbW.*bbH) < thres
        invalInd(bInd) = false;
        continue;
    end
    
    % save bbs with background for the hard negative mining
    bbs_wbg(bInd, :) = curBb;
    
    % remove bg context of roots
    if i_params.test.bgContextSz > 0
        padding = i_mdl.wh - i_mdl.wh/(1+i_params.test.bgContextSz);

        curBb(:, 1) = (curBb(:, 1)*curScale + padding(1)/2)/curScale;
        curBb(:, 2) = (curBb(:, 2)*curScale + padding(2)/2)/curScale;
        
        curBb(:, 3) = (curBb(:, 3)*curScale - padding(1)/2)/curScale;
        curBb(:, 4) = (curBb(:, 4)*curScale - padding(2)/2)/curScale;
    end
    
    % save
    bbs(bInd, :) = curBb;
end
bbs(invalInd, :) = [];
bbs_wbg(invalInd, :) = [];

if ~isempty(bbs)
    [bbs, IND] = sortrows(bbs, -size(bbs, 2));
    bbs_wbg = bbs_wbg(IND, :);
    % nms
    if i_params.test.nms == 1;
        picked = nms(bbs, i_params.test.nmsOverlap);
        bbs = bbs(picked, :);
        bbs_wbg = bbs_wbg(picked, :);
    end
end

%% return
o_bbs = bbs;
o_bbs_wbg = bbs_wbg;

end

function [o_luv] = moveScale(i_luv_cc, i_featPyr, i_step)
nextL = i_luv_cc(1) + i_step;
if nextL < 1 || nextL > numel(i_featPyr)
    o_luv = i_luv_cc;
    o_luv(1) = nextL;
    return;
end

curFeatDim = size(i_featPyr(i_luv_cc(1)).feat);
nextFeatDim = size(i_featPyr(nextL).feat);

nextUV = round(i_luv_cc(2:3) + (nextFeatDim([2 1])' - curFeatDim([2 1])')/2);

o_luv = [nextL; nextUV];
end

function [o_score] = getScore_cc(i_luv_cc, i_featPyr, i_mdl, i_params)
nLevel = numel(i_featPyr);

% discard ones out of range
if i_luv_cc(1) < 1 || i_luv_cc(1) > nLevel
    o_score = -inf;
    return;
end
% find the corresponding image feature
curImgFeat = i_featPyr(i_luv_cc(1)).feat;

% discard ones out of range
if any(i_luv_cc(2:3)<1) || any(i_luv_cc(2:3)+i_mdl.wh_cc-1>[size(curImgFeat, 2); size(curImgFeat, 1)])
    o_score = -inf;
    return;
end

% get a score
curScale = i_featPyr(i_luv_cc(1)).scale;
img_rsz = imresize(i_featPyr(i_luv_cc(1)).img, curScale);
img_crop = imcrop(img_rsz, [i_luv_cc(2:3)' i_mdl.wh_cc']);
curFeat = getHOXFeat(img_crop, i_params.feat.HoG.SqCellSize, i_params.feat.HOG.type);
o_score = i_mdl.w_app'*curFeat(:);

end

% function [o_bbs] = normalizeBbs(i_params, i_mdl, i_bbs, i_scale)
% % cell coordinate to image coordinate
% % rescaling to the original image scale
% o_bbs = i_bbs;
% if isempty(o_bbs)
%     return;
% end
% 
% % root
% % o_bbs(:, 1:4) = cc2ic(o_bbs(:, 1:4)/i_scale, i_params.feat.HoG.SqCellSize);
% o_bbs(:, 1:4) = o_bbs(:, 1:4)*i_params.feat.HoG.SqCellSize/i_scale;
% 
% % parts
% next = 7;
% for pInd=1:numel(i_mdl.parts)
% %     o_bbs(:, next:next+4-1) = cc2ic(o_bbs(:, next:next+4-1)/(i_scale*i_mdl.parts(pInd).ds), i_params.feat.HoG.SqCellSize);
%     o_bbs(:, next:next+4-1) = o_bbs(:, next:next+4-1)*i_params.feat.HoG.SqCellSize/(i_scale*i_mdl.parts(pInd).ds);
%     next = next + 6;
% end
% 
% end
% 
% 
% function [ o_bbs ] = cc2ic( i_bbs, i_sqCellSize )
% %CC2IC cell coordinate to image coordinate
% %   
% 
% o_bbs = [(i_bbs(:, [1 2])-1)*i_sqCellSize + 1, i_bbs(:, [3 4])*i_sqCellSize - 1];
% 
% 
% % o_ic = (i_cc-1)*i_sqCellSize + i_sqCellSize/2 + 1;
% % o_ic = (i_cc-1)*i_sqCellSize + 1;
% end

function [o_bbs] = getCandiWindow(i_img, i_mdl, i_sqCellSz)
%% options
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1};

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:end); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.8;

%% run
boxes = Image2HierarchicalGrouping(i_img, sigma, k, minSize, colorType, simFunctionHandles);
boxes = BoxRemoveDuplicates(boxes);
% ShowRectsWithinImage(boxes, 5, 5, i_img);

%% no local maxima search. just fit to model size
nPert = 1;
overlapRatio = 0.5;
nBBs = size(boxes, 1)*nPert;
o_bbs = zeros(nBBs, 5);
invalInd = false(size(o_bbs, 1), 1);
mdlArea = prod(i_mdl.wh);
for  bInd=1:size(boxes, 1)
    oriRect = [boxes(bInd, [2, 1]) boxes(bInd, 4)-boxes(bInd, 2), boxes(bInd, 3)-boxes(bInd, 1)];
    % w = 0 | h = 0
    if oriRect(3) == 0 || oriRect(4) == 0
        invalInd((bInd-1)*nPert+1:(bInd-1)*nPert+nPert) = true;
        continue;
    end
    
    curArea = prod(oriRect(3:4));
    scaleFactor = sqrt(mdlArea/curArea);
    
    xy_rsz = oriRect(1:2)*scaleFactor;
    wh_rsz = oriRect(3:4)*scaleFactor;
    wh_rsz_new = i_mdl.wh';
    xy_rsz_new = xy_rsz - (wh_rsz_new-wh_rsz)/2;
    newRect = [xy_rsz_new/scaleFactor wh_rsz_new/scaleFactor];
    
    newBb = [newRect, scaleFactor];
    
    intArea = rectint(oriRect, newRect);
    if intArea/(prod(oriRect(3:4))+prod(newRect(3:4))-intArea) < overlapRatio
        invalInd((bInd-1)*nPert+1:(bInd-1)*nPert+nPert) = true;
        continue;
    end
    
%     figure(1);
%     imshow(i_img);
%     hold on; 
%     rectangle('Position', oriRect, 'EdgeColor', 'b');
%     rectangle('Position', newRect, 'EdgeColor', 'r');
%     hold off;
    
    if any(isnan(newBb) | isinf(newBb))
        invalInd((bInd-1)*nPert+1:(bInd-1)*nPert+nPert) = true;
    else
        newBbs = bsxfun(@times, ones(nPert, size(o_bbs, 2)), newBb);
%         newBbs(2, 1) = newBbs(2, 1) - i_sqCellSz/newBb(end);
%         newBbs(3, 2) = newBbs(3, 2) - i_sqCellSz/newBb(end);
%         newBbs(4, 1) = newBbs(4, 1) + i_sqCellSz/newBb(end);
%         newBbs(5, 2) = newBbs(5, 2) + i_sqCellSz/newBb(end);
    
        o_bbs((bInd-1)*nPert+1:(bInd-1)*nPert+nPert, :) = newBbs;
    end
end
o_bbs(invalInd, :) = [];
end
