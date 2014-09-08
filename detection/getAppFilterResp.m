function [ o_resp ] = getAppFilterResp( i_featPyr, i_mdl )
%GETAPPSPAFILTERRESP Summary of this function goes here
%   Detailed explanation goes here

map_IDTI = i_mdl.map_IDTI;
nAllParts = size(map_IDTI, 2);
nFeatLevel = numel(i_featPyr);
c = i_mdl.c;
o_resp = cell(nAllParts, nFeatLevel);
for i=1:nAllParts
    curMdl = getNode(map_IDTI(:, i), i_mdl);
    filterSize = [curMdl.wh_cc(2) curMdl.wh_cc(1) curMdl.appFeatDim/prod(curMdl.wh_cc)];
    curFilter = reshape(curMdl.w_app, filterSize); %%FIXME: inefficient
%     curFilter = ((c==1)*1 + (c==0)*(-1))*curFilter;
    curFilter = ((c==1)*1 + (c==0)*(0))*curFilter;
    curFilter = flipdim(flipdim(flipdim(curFilter, 1), 2), 3);
    
    for l=1:nFeatLevel  
        % get responses
        o_resp{i, l} = convn(i_featPyr(l).feat, curFilter, 'valid');
    end
end
end