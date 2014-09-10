function [ o_bbs, o_bbs_wbg ] = detect_DPM( i_mdl, i_img  )
%DETECT Summary of this function goes here
%   Detailed explanation goes here
% warning('double img!');
% i_img = imresize(i_img, 2);

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
mdlType = i_params.general.mdlType;
partResolution = i_params.feat.HOX.partResRatio;
map_IDTI = objMdl.map_IDTI;
nAllParts = size(map_IDTI, 2);

%% build a feature pyramid
if interval == 0
    img_rz = imresize(img, partResolution); %%FIXME: artifical rescaling
    feats = {getHOXFeat(img_rz, sqCellSz, HOGType); getHOXFeat(img, sqCellSz, HOGType)};
    scales = [partResolution; 1];
else
    img_rz = imresize(img, partResolution); %%FIXME: artifical rescaling
    [feats, scales] = featpyramid(img_rz, sqCellSz, interval, @(img) getHOXFeat(img, sqCellSz, HOGType));
    scales = scales*2;
end
featPyr = struct('scale', [mat2cell(scales, ones(numel(scales), 1), 1)], 'feat', feats);

%% FIXME: padarray for performance, refer detect.m

%% precompute filter responses
resp = getAppFilterResp(featPyr, objMdl);

%% apply generalized distance transform
if nAllParts > 1
    resp_DT = applyDT(resp, objMdl);
end

%% aggregate bounding boxes
bbs = [];
bbs_wbg = [];
for sInd=1:numel(scales)
    curScale = scales(sInd);
    curPartScale = curScale*partResolution;
    % check available parts
    if nAllParts > 1
        psInd = ismember(scales, curPartScale);
        if ~any(psInd)
            continue;
        end
    end
    
    % obtain bbs for a root and parts
    curScaleBbs_cc = cell(1, nAllParts+1);
    
    pnInd = 1;
    curObjMdl = getNode(map_IDTI(:, pnInd), objMdl);
    curResp = resp{pnInd, sInd};
    [cols, rows] = meshgrid(1:size(curResp, 2), 1:size(curResp, 1));
    xy_cc = [cols(:) rows(:)];
    s = curResp(:);
    curScaleBbs_cc{pnInd} = [xy_cc(:, 1) xy_cc(:, 2) xy_cc(:, 1)+curObjMdl.wh_cc(1)-1 xy_cc(:, 2)+curObjMdl.wh_cc(2)-1 s];
    
    for pnInd=2:nAllParts 
        curPart = getNode(map_IDTI(:, pnInd), objMdl);
        
        % update the score
        curLevel = psInd;
        respSz = size(resp{pnInd, psInd});
        assert(all(respSz == size(resp_DT{pnInd, curLevel}.score)));

        % getAnchor: i_child.ds*i_parent.uv_cc + i_child.dudv_cc - 1;
        anchorPos_part_cc = bsxfun(@plus, partResolution*xy_cc - 1, curPart.dudv_cc') - 1;
        valInd = ... %%FIXME: really required?? bugs??
                anchorPos_part_cc(:, 1) >= 1 & anchorPos_part_cc(:, 1) <= respSz(2) & ...
                anchorPos_part_cc(:, 2) >= 1 & anchorPos_part_cc(:, 2) <= respSz(1);
        ind = sub2ind(size(resp_DT{pnInd, curLevel}.score), anchorPos_part_cc(valInd, 2), anchorPos_part_cc(valInd, 1));
        
        % uv
        xy_part_cc = ones(size(anchorPos_part_cc, 1), 2);
        xy_part_cc(valInd, :) = double([...
            resp_DT{pnInd, curLevel}.Ix(ind) resp_DT{pnInd, curLevel}.Iy(ind)]);
        
        % scores
        s = ones(size(anchorPos_part_cc, 1), 1)*-inf;
        ind = sub2ind(size(resp_DT{pnInd, curLevel}.score), xy_part_cc(valInd, 2), xy_part_cc(valInd, 1));
        s(valInd) = resp_DT{pnInd, curLevel}.score(ind);
        
        % bbs
        curScaleBbs_cc{pnInd} = [xy_part_cc(:, 1) xy_part_cc(:, 2) xy_part_cc(:, 1)+curPart.wh_cc(1)-1 xy_part_cc(:, 2)+curPart.wh_cc(2)-1 s];
    end
    
    % sum scores of parts and a root
    ss = zeros(size(xy_cc, 1), 1);
    for pnInd=1:nAllParts 
        ss = ss + curScaleBbs_cc{pnInd}(:, end);
    end
    curScaleBbs_cc{end} = ss;
    
    % cell coord to pixel coord and rescale to original scale
    curBbs = normalizeBbs(sqCellSz, objMdl, curScaleBbs_cc, curScale, partResolution);
    curBbs = cell2mat(curBbs);
    
    % threshold scores
    curBbs = curBbs(curBbs(:, end)>scoreThres, :);

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

%% reorganize bbs
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

function [o_bbs_cell] = normalizeBbs(i_sqCellSz, i_mdl, i_bbs_cell, i_scale, i_partResolution)
% cell coordinate to image coordinate
% rescaling to the original image scale
o_bbs_cell = i_bbs_cell;
if isempty(o_bbs_cell)
    return;
end
pnInd = 1;
o_bbs_cell{pnInd}(:, 1:4) = o_bbs_cell{pnInd}(:, 1:4)*i_sqCellSz/i_scale;

% parts
for pnInd=2:size(o_bbs_cell, 2)-1
%     pInd = pnInd - 1;
%     o_bbs_cell{pnInd}(:, 1:4) = o_bbs_cell{pnInd}(:, 1:4)*i_sqCellSz/(i_scale*i_mdl.parts(pInd).ds);
    o_bbs_cell{pnInd}(:, 1:4) = o_bbs_cell{pnInd}(:, 1:4)*i_sqCellSz/(i_scale*i_partResolution);
end

end


