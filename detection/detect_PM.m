function [ o_bbs, o_bbs_wbg ] = detect_PM( i_mdl, i_img  )
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
mdlType = i_params.general.mdlType;
partResolution = i_params.feat.HOX.partResRatio;
map_IDTI = objMdl.map_IDTI;
nAllParts = size(map_IDTI, 2);

%% build a feature pyramid
if interval == 0
    feats = {getHOXFeat(img, sqCellSz, HOGType)};
    scales = 1;
else
    [feats, scales] = featpyramid(img, sqCellSz, interval, @(img) getHOXFeat(img, sqCellSz, HOGType));
end
featPyr = struct('scale', [mat2cell(scales, ones(numel(scales), 1), 1)], 'feat', feats);

%% FIXME: padarray for performance, refer detect.m

%% precompute filter responses
resp = getAppFilterResp(featPyr, objMdl);

%% aggregate bounding boxes
bbs = [];
bbs_wbg = [];
for sInd=1:numel(scales)
    curScale = scales(sInd);
    if mdlType == 3
        % check available parts
        psInd = ismember(curScale*partResolution, scales);
        if ~psInd
            continue;
        end
    end
    % obtain bbs for a root and parts
    curScaleBbs_cc = cell(1, nAllParts+1);
    
    pnInd = 1;
    curObjMdl = getNode(map_IDTI(:, pnInd), objMdl);
    curResp = resp{pnInd, sInd};
    [cols, rows] = meshgrid(1:size(curResp, 2), 1:size(curResp, 1));
    x_cc = cols(:);
    y_cc = rows(:);
    s = curResp(:);
    curScaleBbs_cc{pnInd} = [x_cc y_cc x_cc+curObjMdl.wh_cc(1)-1 y_cc+curObjMdl.wh_cc(2)-1 s];
    
    for pnInd=2:nAllParts 
        curObjMdl = getNode(map_IDTI(:, pnInd), objMdl);
        curResp = resp{pnInd, psInd};
        
        xy_part_cc = bsxfun(@plus, partResolution*[x_cc y_cc] - 1, curObjMdl.uv_cc');
        
        %%FIXME: handle boundary errors
        valInd = xy_part_cc(:, 1) <= size(curResp, 2) & xy_part_cc(:, 2) <= size(curResp, 1);
        ind = sub2ind(size(curResp), xy_part_cc(valInd, 2), xy_part_cc(valInd, 1));
        s = ones(size(xy_part_cc, 1), 1)*-inf;
        s(valInd) = curResp(ind);
        
        xy_part_cc_max = xy_part_cc;
        s_max = s;
        for i=1:16 % perturb
            switch i
                case 1
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) - 1;
                case 2
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) - 1;
                case 3
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) - 1;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) - 1;
                case 4
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) + 1;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) - 1;
                case 5
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) - 1;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) + 1;
                case 6
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) + 1;
                case 7
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) + 1;
                case 8
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) + 1;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) + 1;
                case 9
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) - 2;
                case 10
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) - 2;
                case 11
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) - 2;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) - 2;
                case 12
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) + 2;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) - 2;
                case 13
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) - 2;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) + 2;
                case 14
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) + 2;
                case 15
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) + 2;
                case 16
                    xy_part_cc_var = xy_part_cc;
                    xy_part_cc_var(:, 1) = xy_part_cc_var(:, 1) + 2;
                    xy_part_cc_var(:, 2) = xy_part_cc_var(:, 2) + 2;
                    
            end
            
            %%FIXME: handle boundary errors
            valInd = ...
                xy_part_cc_var(:, 1) >= 1 & xy_part_cc_var(:, 1) <= size(curResp, 2) & ...
                xy_part_cc_var(:, 2) >= 1 & xy_part_cc_var(:, 2) <= size(curResp, 1);
            ind = sub2ind(size(curResp), xy_part_cc_var(valInd, 2), xy_part_cc_var(valInd, 1));
            s_var = ones(size(xy_part_cc_var, 1), 1)*-inf;
            s_var(valInd) = curResp(ind);
            
            % take max
           [C, I] = max([s_max, s_var], [], 2);
           s_max = C;
           xy_part_cc_max(I == 2, :) = xy_part_cc_var(I == 2, :);
            
        end
        
        xy_part_cc = xy_part_cc_max;
        s = s_max;
        
%         s = curResp(ind);
        curScaleBbs_cc{pnInd} = [xy_part_cc(:, 1) xy_part_cc(:, 2) xy_part_cc(:, 1)+curObjMdl.wh_cc(1)-1 xy_part_cc(:, 2)+curObjMdl.wh_cc(2)-1 s];
    end
    
    % sum scores of parts and a root
    ss = zeros(numel(x_cc), 1);
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
    pInd = pnInd - 1;
%     o_bbs_cell{pnInd}(:, 1:4) = o_bbs_cell{pnInd}(:, 1:4)*i_sqCellSz/(i_scale*i_mdl.parts(pInd).ds);
    o_bbs_cell{pnInd}(:, 1:4) = o_bbs_cell{pnInd}(:, 1:4)*i_sqCellSz/(i_scale*i_partResolution);
end

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

