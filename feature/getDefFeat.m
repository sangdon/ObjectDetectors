function [ o_feat ] = getDefFeat( i_parentMdl, i_childMdl )
%GETDEFFEAT Summary of this function goes here
%   Detailed explanation goes here

if i_parentMdl.c
%     pq = i_childMdl.uv_cc - (i_childMdl.ds*i_parentMdl.uv_cc - 1 + i_childMdl.dudv_cc);
    pq = i_childMdl.uv_cc - getAnchor( i_parentMdl, i_childMdl );

    o_feat = -[pq(:); pq(:).^2];
else
    o_feat = zeros(4, 1);
end

end

