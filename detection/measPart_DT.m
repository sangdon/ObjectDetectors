function [ o_score, o_mdl ] = measPart_DT( i_imgSt, i_mdl, i_uvsc ) 
%MEASOBJCLS Summary of this function goes here
%   Detailed explanation goes here

%%%%% assume a shallow model

i_mdl = updMdlUVSC(i_mdl, i_uvsc);
o_mdl = i_mdl;
if i_mdl.c == 0
    o_score = 0;
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
appScore = resp{1, curLevel}(round((rootPart.wh_cc(2)-1)/2), round((rootPart.wh_cc(1)-1)/2));

score = score + appScore;
o_mdl(1).appScore = appScore;
    
% obtain scores for children
for i=2:nAllParts 
    curPart = getNode(map_IDTI(:, i), o_mdl);
    if curPart.c == 0
        continue;
    end
    
    % update the score
    curLevel = ismember([featPyr(:).scale], curPart.s);
    parPart = rootPart;
%     anchorPos = curPart.ds*parPart.uv_cc - 1 + curPart.dudv_cc;
    anchorPos = getAnchor( parPart, curPart );
    
    curScore = resp_DT{i, curLevel}.score(anchorPos(2), anchorPos(1));
    score = score + curScore;
%     o_mdl(i).appScore = resp{i, curLevel}(curPart.uv_cc(2), curPart.uv_cc(1));
%     o_mdl(i).defScore = curScore - o_mdl(i).appScore;

    if curScore~= 0
        keyboard;
    end

    % update part locations
%     curPart.uv_cc = double([resp_DT{i, curLevel}.Ix(anchorPos(2), anchorPos(1)); resp_DT{i, curLevel}.Iy(anchorPos(2), anchorPos(1))]);
%     o_mdl = setNode(o_mdl, map_IDTI(:, i), curPart);
end
% obtain a bias score
score = score + rootPart.w_b(1)*1; 

%% return
o_score = score;

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
    curFilter = reshape(curMdl.w_app, filterSize);
    
    for l=1:nFeatLevel  
        % get responses          
        o_resp{i, l} = sum(convn(i_featPyr(l).feat, curFilter, 'same'), 3);
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
            curMdl.w_def(1), curMdl.w_def(2), curMdl.w_def(3), curMdl.w_def(4), 4); %% constant 4 here !!!!!!!!!!!!!
        o_resp_DT{i, l} = resp;                                    
    end
end


end


function [o_node] = getNode(i_treeInd, i_tree)
nDepth = numel(i_treeInd);

curNode = i_tree;
for d=1:nDepth
    if i_treeInd(d) == 0
        break;
    else
        curNode = curNode.parts(i_treeInd(d));
    end
    
end
o_node = curNode;
end

function [o_tree] = setNode(i_tree, i_treeInd, i_node)
o_tree = i_tree;

if i_treeInd(1) == 0
    o_tree = i_node;
else
    o_tree.parts(i_treeInd(1)) = i_node;
end
end

