function [ o_feat ] = getAppFeat( i_sqCellSize, i_type, i_imgSt, i_mdl, i_uvsc )
%GETFEATURE Summary of this function goes here
%   Detailed explanation goes here

i_mdl = updMdlUVSC( i_mdl, i_uvsc );

% if i_mdl.c == 0
%     o_feat = zeros([i_mdl.wh_cc(2) i_mdl.wh_cc(1) i_mdl.appFeatDim/prod(i_mdl.wh_cc)]);
%     return;
% end

if isempty(i_imgSt.featPyr)
    %% don't have precomputed features
    img = i_imgSt.img;
    
    rect = [i_mdl.uv' i_mdl.wh'-1]; % in scale space
    img = imcrop(img, rect);
%     img = imresize(imcrop(img, rect), i_mdl.s);
    
    o_feat = getHOXFeat(img, i_sqCellSize, i_type);
else
    %% has precomputed features
    level = findFeatPyrLevel(i_imgSt.featPyr, i_mdl.s);
%     if level == 0
%         o_feat = zeros([i_mdl.wh_cc(2) i_mdl.wh_cc(1) i_mdl.appFeatDim/prod(i_mdl.wh_cc)]);
%         return;
%     end
    feat = i_imgSt.featPyr(level).feat;
    
% %     rect = ic2cc([i_mdl.uv' i_mdl.wh']*i_mdl.s, i_params.feat.HoG.SqCellSize, i_mdl.wh_cc);
% %     rect = ic2cc([i_mdl.uv' i_mdl.wh'], i_params.feat.HoG.SqCellSize, i_mdl.wh_cc);
% %     rect = [i_mdl.uv_cc' i_mdl.wh_cc'];
% 
%     if any(i_mdl.wh_cc == 0) % dimension is not yet decided.
%         o_feat = feat;
%         warning('don''t know wh_cc. Is it ok?');
%     else
%         rect = [max([1 1], i_mdl.uv_cc') i_mdl.wh_cc'];
%         
%         o_feat = crop3dmat(feat, rect);
%     end
    
    rect = [i_mdl.uv_cc' i_mdl.wh_cc'];
%     offset_s = min(rect(1:2) - [1 1], [0 0]);
%     rect(1:2) = rect(1:2) - offset_s;
%     ePnt = rect(1:2) + rect(3:4) - 1;
%     offset_e = max(ePnt - [size(feat, 2) size(feat, 1)], [0 0]);
%     rect(1:2) = rect(1:2) - offset_e;
    
    
    
    o_feat = crop3dmat(feat, rect);
end
end




