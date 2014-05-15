function [ o_mdl ] = updMdlUVSC( i_mdl, i_uvsc )
%UPDMDLUVSC Summary of this function goes here
%   Detailed explanation goes here
o_mdl = i_mdl;
if isempty(i_uvsc)
    return;
end

o_mdl = updMdlUV(o_mdl, i_uvsc(1:2));
o_mdl = updMdlS(o_mdl, i_uvsc(3));
o_mdl = updMdlC(o_mdl, i_uvsc(4));

end

function [ o_mdl ] = updMdlUV( i_mdl, i_uv_cc )
%UPDMDLUV Summary of this function goes here
%   Detailed explanation goes here
o_mdl = i_mdl;
o_mdl.uv_cc = i_uv_cc(:);
end

function [ o_mdl ] = updMdlS( i_mdl, i_s )
%UPDMDLS Summary of this function goes here
%   Detailed explanation goes here
o_mdl = i_mdl;
o_mdl.s = i_s;
for pInd=1:numel(o_mdl.parts)
    o_mdl.parts(pInd).s = o_mdl.s*o_mdl.parts(pInd).ds;
end

end

function [o_mdl] = updMdlC(i_mdl, i_c)
o_mdl = i_mdl;
o_mdl.c = i_c;
for pInd=1:numel(o_mdl.parts)
    o_mdl.parts(pInd).c = i_c;
end
end



