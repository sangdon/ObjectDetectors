function [ o_feat, o_mdl ] = getFeat( i_params, i_imgSt, i_mdl, i_uvsc )
%GETFEAT Summary of this function goes here
%   Detailed explanation goes here

o_mdl = updMdlUVSC(i_mdl, i_uvsc);

%% get appearance feature
appFeat = getAppFeat(i_params.feat.HOX.SqCellSize, i_params.feat.HOX.type, i_imgSt, o_mdl, []);
o_feat = appFeat(:);

o_mdl.appFeatDim = numel(appFeat(:));
o_mdl.wh_cc = [size(appFeat, 2); size(appFeat, 1)];
% rect_cc = ic2cc([o_mdl.uv; o_mdl.wh], i_params.feat.HoG.SqCellSize, o_mdl.wh_cc);
% o_mdl.uv_cc = rect_cc(1:2);

%% get features of child 
for cInd=1:numel(o_mdl.parts)
    
    % get features
    childAppFeat = getAppFeat(i_params.feat.HoG.SqCellSize, i_params.feat.HoG.type, i_imgSt, o_mdl.parts(cInd), []);
    childDefFeat = getDefFeat(o_mdl, o_mdl.parts(cInd));
    o_feat = [o_feat; childAppFeat(:); childDefFeat(:)];
    
    % add additional information
    o_mdl.parts(cInd).appFeatDim = numel(childAppFeat(:)); % appFeatDim
    o_mdl.parts(cInd).defFeatDim = numel(childDefFeat(:)); % defFeatDim
    o_mdl.parts(cInd).wh_cc = [size(childAppFeat, 2); size(childAppFeat, 1)]; % wh_cc
%     rect_ic = [o_mdl.parts(cInd).uv; o_mdl.parts(cInd).wh]*o_mdl.parts(cInd).ds;
%     rect_cc = ic2cc(rect_ic, i_params.feat.HoG.SqCellSize, o_mdl.parts(cInd).wh_cc);
%     o_mdl.parts(cInd).uv_cc = rect_cc(1:2); % uv_cc
    rect_ic = [o_mdl.parts(cInd).dudv; o_mdl.parts(cInd).wh]*o_mdl.parts(cInd).ds;
    rect_cc = ic2cc(rect_ic, i_params.feat.HoG.SqCellSize, o_mdl.parts(cInd).wh_cc);
    o_mdl.parts(cInd).dudv_cc = rect_cc(1:2); % dudv_cc
    
%     o_mdl.parts(cInd).bFeatDim = 1;
end

%% add bias term
o_feat = [o_feat; 1];
o_mdl.bFeatDim = 1;

end

function [ o_rect_cc ] = ic2cc( i_rect_ic, i_sqCellSize,  i_wh_cc)
%IC2CC image coordinate to cell coordinate
%   i_rect_ic: [x, y, w, h]

o_rect_cc = floor(i_rect_ic/i_sqCellSize) + 1;
o_rect_cc(3:4) = i_wh_cc; % not good!
end