function [ o_score, o_mdl ] = measPart_DT( i_imgSt, i_mdl, i_uvsc ) %% FIXME: should be combined with slideWindow....computationally inefficient...
%MEASOBJCLS Summary of this function goes here
%   Detailed explanation goes here

%%FIXME: assume a shallow model

i_mdl = updMdlUVSC(i_mdl, i_uvsc);
o_mdl = i_mdl;
if o_mdl.c == 0
    o_score = 0; %%FIXME: correct way?
    return;
end
    
map_IDTI = o_mdl.map_IDTI;
nAllParts = size(map_IDTI, 2);
featPyr = i_imgSt.featPyr;

%% get filter responses
resp = getAppFilterResp(featPyr, o_mdl);

%% apply generalized distance transform
resp_DT = applyDT(resp, o_mdl);

%% obtain a score
score = 0;

% obtin scores for the parent
rootPart = getNode(map_IDTI(:, 1), o_mdl);
curLevel = ismember([featPyr(:).scale], rootPart.s);
validResp = resp{1, curLevel}(...
    1+round((rootPart.wh_cc(2)-1)/2):end-round((rootPart.wh_cc(2)-1)/2), ...
    1+round((rootPart.wh_cc(1)-1)/2):end-round((rootPart.wh_cc(1)-1)/2));
[C, I] = max(validResp, [], 1);
[resp_max, x_opt] = max(C, [], 2);
y_opt = I(x_opt);
cent_xy_opt = [x_opt; y_opt];

appScore = resp_max;
score = score + appScore;

rootPart.appScore = appScore;
rootPart.defScore = 0;

% update a root location
rootPart.uv_cc = cent_xy_opt; % round((rootPart.wh_cc-1)/2) - round((rootPart.wh_cc-1)/2)

% update the node
o_mdl = setNode(o_mdl, map_IDTI(:, 1), rootPart);

% obtain scores for children
for i=2:nAllParts 
    curPart = getNode(map_IDTI(:, i), o_mdl);
    if curPart.c == 0
        continue;
    end
    
    % update the score
    curLevel = ismember([featPyr(:).scale], curPart.s);
    if isempty(find(curLevel, 1))
        score = -inf;
        break;
    end
    validResp = resp{i, curLevel}(...
        1+round((curPart.wh_cc(2)-1)/2):end-round((curPart.wh_cc(2)-1)/2), ...
        1+round((curPart.wh_cc(1)-1)/2):end-round((curPart.wh_cc(1)-1)/2));
    anchorPos_partcc = getAnchor( rootPart, curPart );
    [C, I] = max(validResp, [], 1);
    [resp_max, x_opt] = max(C, [], 2);
    y_opt = I(x_opt);
    cent_xy_app_opt = [x_opt; y_opt];
    
    curScore = resp_DT{i, curLevel}.score(anchorPos_partcc(2), anchorPos_partcc(1));
    score = score + curScore;
    
    curPart.appScore = resp_max;
    curPart.defScore = curScore - curPart.appScore; %%FIXME: correct?

    % update part locations
    curPart.uv_cc = double([...
        resp_DT{i, curLevel}.Ix(anchorPos_partcc(2), anchorPos_partcc(1)); ...
        resp_DT{i, curLevel}.Iy(anchorPos_partcc(2), anchorPos_partcc(1))]);
    
    % update the node
    o_mdl = setNode(o_mdl, map_IDTI(:, i), curPart);
end
% obtain a bias score
score = score + rootPart.w_b(1)*1; 

%% return
o_score = score;

end

function [ o_resp ] = getAppFilterResp( i_featPyr, i_mdl ) %% FIXME: old, check detect.m
%GETAPPSPAFILTERRESP Summary of this function goes here
%   Detailed explanation goes here

map_IDTI = i_mdl.map_IDTI;
nAllParts = size(map_IDTI, 2);
nFeatLevel = numel(i_featPyr);

o_resp = cell(nAllParts, nFeatLevel);
for i=1:nAllParts
    curMdl = getNode(map_IDTI(:, i), i_mdl);
    filterSize = [curMdl.wh_cc(2) curMdl.wh_cc(1) curMdl.appFeatDim/prod(curMdl.wh_cc)];
    curFilter = reshape(curMdl.w_app, filterSize);
    
    for l=1:nFeatLevel  
        % get responses
        resp = convn(i_featPyr(l).feat, curFilter, 'same');
        o_resp{i, l} = resp(:, :, 1+round((size(resp, 3)-1)/2));
    end
end
end

function [o_resp_DT] = applyDT(i_resp, i_mdl)

map_IDTI = i_mdl.map_IDTI;
nAllParts = size(map_IDTI, 2);
nFeatLevel = size(i_resp, 2);

o_resp_DT = cell(nAllParts, nFeatLevel);
for i=2:nAllParts % run except for the root
    curMdl = getNode(map_IDTI(:, i), i_mdl);
    for l=1:nFeatLevel
        resp = struct('score', 0, 'Ix', [], 'Iy', []); 
        
        [resp.score, resp.Ix, resp.Iy] = bounded_dt(i_resp{i, l}, ...
            curMdl.w_def(1), curMdl.w_def(2), curMdl.w_def(3), curMdl.w_def(4), 4); %%FIXME: constant 4 here !!!!!!!!!!!!!
        o_resp_DT{i, l} = resp;                                    
    end
end


end






