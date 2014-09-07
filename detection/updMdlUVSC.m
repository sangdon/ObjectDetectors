function [ o_mdl ] = updMdlUVSC( i_mdl, i_uv, i_s, i_c )
%UPDMDLUVSC Summary of this function goes here
%   Detailed explanation goes here
o_mdl = i_mdl;

if ~isempty(i_uv)
    o_mdl = updMdlUV(o_mdl, i_uv);
end
if ~isempty(i_s)
    o_mdl = updMdlS(o_mdl, i_s);
end
if ~isempty(i_c)
    o_mdl = updMdlC(o_mdl, i_c);
end

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



