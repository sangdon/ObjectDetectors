%% temporary codes for codegen
function [o_bbs] = slideWindow_part(i_sqCellSize, i_type, i_imgSt, i_objMdl, i_valRectProp_cc)
mdlCellWH = i_objMdl.wh_cc;

% invalid featurepyramid level   
level = findFeatPyrLevel(i_imgSt.featPyr,  i_objMdl.s);
if level == 0
    o_bbs = zeros(0, getNParts(i_objMdl)*6+1);
    return;
end

% find a search space
featW = size(i_imgSt.featPyr(level).feat, 2);
featH = size(i_imgSt.featPyr(level).feat, 1);
defaultRect = [1; 1; featW; featH];
if all(i_valRectProp_cc == 0)
    validRect_cc = defaultRect;
else
    valSPnt = max(defaultRect(1:2), i_valRectProp_cc(1:2)); 
    valEPnt = min(defaultRect(1:2)+defaultRect(3:4)-1, i_valRectProp_cc(1:2)+i_valRectProp_cc(3:4)-1);
    
    validRect_cc = [valSPnt; valEPnt-valSPnt+1];
end

% search space should be bigger than the model size
if any(mdlCellWH > validRect_cc(3:4))
    o_bbs = zeros(0, getNParts(i_objMdl)*4+1);
    return;
end

% find start points
[Xs, Ys] = meshgrid(...
    validRect_cc(1):validRect_cc(1)+validRect_cc(3)-mdlCellWH(1), ...
    validRect_cc(2):validRect_cc(2)+validRect_cc(4)-mdlCellWH(2));
sPnts = [Xs(:)'; Ys(:)'];

%% check the measurement of each candidates
bbs = zeros(size(sPnts, 2), getNParts(i_objMdl)*6+1);
% parfor
parfor (spInd=1:size(sPnts, 2), 20)

    % get the score of the model
    [score, curMdl] = measPart_part(...
        i_sqCellSize, ...
        i_type, ...
        i_imgSt, ...
        i_objMdl, ...
        [sPnts(:, spInd); i_objMdl.s; 1]);
    
    % save the bb: [xmin ymin xmax ymax appScore defScore]
    bbs(spInd, :) = [getBbs(curMdl) score];
end
o_bbs = bbs;
end


function [o_bb] = getBbs(i_mdl)

next = 1;
size = 6;
o_bb = zeros(1, getNParts(i_mdl)*size);


sPnt = i_mdl.uv_cc;
ePnt = sPnt + i_mdl.wh_cc - 1;
o_bb(next:next+size-1) = [sPnt(1) sPnt(2) ePnt(1) ePnt(2) i_mdl.appScore i_mdl.defScore];
next = next + size;

for pInd=1:numel(i_mdl.parts)
    sPnt = i_mdl.parts(pInd).uv_cc;
    ePnt = sPnt + i_mdl.parts(pInd).wh_cc - 1;
    o_bb(next:next+size-1) = [sPnt(1) sPnt(2) ePnt(1) ePnt(2) i_mdl.parts(pInd).appScore i_mdl.parts(pInd).defScore];
    next = next + size;
end
end

function [o_np] = getNParts(i_mdl)
o_np = 1;
o_np = o_np + numel(i_mdl.parts);
end


function [ o_score, o_mdl ] = measPart_part( i_sqCellSize, i_type, i_imgSt, i_mdl, i_uvsc ) %#codegen
%MEASOBJCLS Summary of this function goes here
%   Detailed explanation goes here

i_mdl = updMdlUVSC(i_mdl, i_uvsc);
i_mdl.uv = i_uvsc(1:2)*i_sqCellSize; %%%%%%%%%%%%%%%%

o_mdl = i_mdl;
score = 0;
%% root appearance score
appScore = measAppScore(i_sqCellSize, i_type, i_imgSt, o_mdl, []);
o_mdl.appScore = appScore;
score = score + appScore;

% %% root bias score
% score = score + measBScore(o_mdl);

%% return
o_score = score;

end

function [o_score] = measAppScore(i_sqCellSize, i_type, i_imgSt, i_mdl, i_uvsc)
if i_mdl.c == 0 % for efficienty
    o_score = 0;
else
    appFeat = getAppFeat(i_sqCellSize, i_type, i_imgSt, i_mdl, i_uvsc);
    o_score = appFeat(:)'*i_mdl.w_app;
end
end

function [o_score] = measBScore(i_partMdl)
o_score = i_partMdl.w_b;
end


