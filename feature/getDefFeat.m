function [ o_feat ] = getDefFeat( i_parentMdl, i_childMdl )
%GETDEFFEAT Summary of this function goes here
%   Detailed explanation goes here

% if i_parentMdl.c == 1
    pq = i_childMdl.uv_cc - getAnchor( i_parentMdl, i_childMdl );

%     o_feat = [pq(:); pq(:).^2]; % depending on the implementation: I'm using bounded_dt.mex and official SSVM
    o_feat = -[pq(:); pq(:).^2];
% else
%     o_feat = zeros(4, 1);
% end

end

