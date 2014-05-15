function [ o_bbs, o_bbs_wbg ] = detect_new( i_params, i_img, i_mdl, i_cacheID )
%DETECT Summary of this function goes here
%   Detailed explanation goes here
img = i_img;
i_objMdl = i_mdl;
sqCellSz = i_params.feat.HoG.SqCellSize;
%% build a feature pyramid
fpTID = tic;
if i_params.feat.HOG.type < 4 || i_params.feat.HOG.type == 6
    [feats, scales] = featpyramid(img, i_params.feat.HoG.SqCellSize, i_params.test.interval, @(img) getHOXFeat(img, i_params.feat.HoG.SqCellSize, i_params.feat.HOG.type));

    featPyr = [];
    for sInd=1:numel(scales)
        curFeat = [];
        curFeat.scale = scales(sInd);
        curFeat.feat = padarray(feats{sInd}, [i_mdl.wh_cc(2)-1 i_mdl.wh_cc(1)-1 0]);
        featPyr = [featPyr; curFeat];
    end
else
    cacheFN = sprintf('%s/test_%s_feats.mat', i_params.feat.cachingDir, i_cacheID);
    if i_params.general.enableCaching && exist(cacheFN, 'file')
        load(cacheFN);
    else
        warning('tmp scale space');
        scales = 1:-0.1:0.3;

        featPyr = [];
        for s=scales(:)'

            img_rsz = imresize(img, s);

            [x, y] = meshgrid(1:sqCellSz:floor(size(img_rsz, 2)/sqCellSz)*sqCellSz-i_mdl.wh(1), 1:sqCellSz:floor(size(img_rsz, 1)/sqCellSz)*sqCellSz-i_mdl.wh(2));
            xy = [x(:)'; y(:)'];

            featInd = zeros(3, size(xy, 2));
            feats = zeros(i_mdl.appFeatDim, size(xy, 2));
            parfor ind=1:size(xy, 2)
                fprintf('- extracing features for detection of scale %.3f: %d/%d...', s, ind, size(xy, 2));
                feTID = tic;

                x = xy(1, ind);
                y = xy(2, ind);

                featInd(:, ind) = [s, x, y]';
                img_crop = imcrop(img_rsz, [x, y, i_mdl.wh']);
                feat = getHOXFeat(img_crop, i_params.feat.HoG.SqCellSize, i_params.feat.HOG.type);
                feats(:, ind) = feat(:);

                fprintf('%s\n', num2str(toc(feTID)));
            end

            featPyr.ind = [featPyr.ind featInd];
            featPyr.feat = [featPyr.feat feats];
        end
        
        save(cacheFN, 'featPyr');
    end
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
    if i_params.feat.HOG.type < 4  || i_params.feat.HOG.type == 6
        curBbs = slideWindow_mex(i_params.feat.HoG.SqCellSize, i_params.feat.HOG.type, imgSt, updMdlUVSC(i_objMdl, [1; 1; curScale; 1]), zeros(4, 1));
    else
        curBbs = slideWindow_simple(i_params.feat.HoG.SqCellSize, i_params.feat.HOG.type, imgSt, updMdlUVSC(i_objMdl, [1; 1; curScale; 1]), zeros(4, 1));
    end
    
    % remove the effect of the feature padding
    curBbs(:, [1 3]) = curBbs(:, [1 3]) - (i_mdl.wh_cc(1) - 1);
    curBbs(:, [2 4]) = curBbs(:, [2 4]) - (i_mdl.wh_cc(2) - 1);
    
    % cell coord to pixel coord and rescale to original scale
    curBbs = normalizeBbs(i_params, i_objMdl, curBbs, curScale);
    
    % discard bbs that exceed image space (only considers for root locations)
    thres = 0.8;
    bbW = (curBbs(:, 3) - curBbs(:, 1));
    bbH = (curBbs(:, 4) - curBbs(:, 2));
    bbW_in = (min(size(img, 2), curBbs(:, 3)) - max(1, curBbs(:, 1)));
    bbH_in = (min(size(img, 1), curBbs(:, 4)) - max(1, curBbs(:, 2)));
    invalInd = (bbW_in.*bbH_in)./(bbW.*bbH) < thres;
    curBbs(invalInd, :) = [];
    
    % discard bbs that exceed the score threshold
    curBbs = curBbs(curBbs(:, end)>i_params.test.scoreThres, :);

    
    bbs_wbg = [bbs_wbg; curBbs];
    % remove bg context of roots
    if i_params.test.bgContextSz > 0
        padding = i_mdl.wh - i_mdl.wh/(1+i_params.test.bgContextSz);
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

function [o_bbs] = normalizeBbs(i_params, i_mdl, i_bbs, i_scale)
% cell coordinate to image coordinate
% rescaling to the original image scale
o_bbs = i_bbs;
if isempty(o_bbs)
    return;
end

% root
% o_bbs(:, 1:4) = cc2ic(o_bbs(:, 1:4)/i_scale, i_params.feat.HoG.SqCellSize);
o_bbs(:, 1:4) = o_bbs(:, 1:4)*i_params.feat.HoG.SqCellSize/i_scale;

% parts
next = 7;
for pInd=1:numel(i_mdl.parts)
%     o_bbs(:, next:next+4-1) = cc2ic(o_bbs(:, next:next+4-1)/(i_scale*i_mdl.parts(pInd).ds), i_params.feat.HoG.SqCellSize);
    o_bbs(:, next:next+4-1) = o_bbs(:, next:next+4-1)*i_params.feat.HoG.SqCellSize/(i_scale*i_mdl.parts(pInd).ds);
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

