function [ o_bb ] = evalCandiObj( i_uv_cc, i_objMdl, i_sqCellSize, i_type, i_imgSt )
%EVALWINDOW Summary of this function goes here
%   Detailed explanation goes here
  
% get the score of the model
[score, curMdl] = measPart(...
    i_sqCellSize, ...
    i_type, ...
    i_imgSt, ...
    i_objMdl, ...
    [i_uv_cc; i_objMdl.s; 1]);

% save the bb: [xmin ymin xmax ymax appScore defScore]
o_bb = [getBbs(curMdl) score];

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