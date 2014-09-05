function [ o_score, o_mdl ] = measPart( i_sqCellSize, i_type, i_imgSt, i_mdl, i_uvsc ) 
%MEASOBJCLS Summary of this function goes here
%   Detailed explanation goes here

% warning('legacy!!');

i_mdl = updMdlUVSC(i_mdl, i_uvsc);
i_mdl.uv = i_uvsc(1:2)*i_sqCellSize; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% what is that??

o_mdl = i_mdl;
score = 0;
%% root appearance score
appScore = measAppScore(i_sqCellSize, i_type, i_imgSt, o_mdl, []);
o_mdl.appScore = appScore;
score = score + appScore;

%% root deformation score
defScore = 0;
o_mdl.defScore = defScore;
score = score + defScore;

%% parts scores
children = o_mdl.parts;
for cInd=1:numel(children)
    childMdl = children(cInd);

    if childMdl.c == 0
        continue;
    end
    
    % find optimal part locations    
%     candiParts = slideWindow(i_params, ...
%         i_imgSt, ...
%         childMdl, ...
%         childMdl.ds*[i_mdl.uv_cc; i_mdl.wh_cc]);
    % for Matlab coder
    candiParts = slideWindow_part(i_sqCellSize, i_type, ...
        i_imgSt, ...
        childMdl, ...
        childMdl.ds*[i_mdl.uv_cc; i_mdl.wh_cc]);
    
    % argmax
    maxScore = -inf;
    maxChild = childMdl; % for initialization
    for bInd=1:size(candiParts, 1) 
        
        curChildMdl = childMdl;
        
        % deformation score
        curChildMdl.uv_cc = reshape(candiParts(bInd, 1:2), [2 1]);
        defScore = measDefScore(o_mdl, curChildMdl);        
        curChildMdl.defScore = defScore;
        
        % appearance score
        appScore = candiParts(bInd, end);
        curChildMdl.appScore = appScore;

        % total score
        totalScore = defScore + appScore;
        
        if maxScore < totalScore
            maxScore = totalScore;
            maxChild = curChildMdl;    
        end
    end
    
    score = score + maxScore;
    children(cInd) = maxChild;
end
o_mdl.parts = children;

%% root bias score
bScore = measBScore(o_mdl);
score = score + bScore;

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

function [o_score] = measDefScore(i_parentMdl, i_childMdl)
if i_parentMdl.c == 0 % for efficienty
    o_score = 0;
else
    defFeat = getDefFeat(i_parentMdl, i_childMdl);
    o_score = -defFeat'*i_childMdl.w_def;
%     o_score = defFeat'*i_childMdl.w_def; % sign: depending on the
%     library...
end
end

function [o_score] = measBScore(i_partMdl)
o_score = i_partMdl.w_b(1)*1;
end


