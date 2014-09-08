function [o_resp_DT] = applyDT(i_resp, i_mdl)

map_IDTI = i_mdl.map_IDTI;
nAllParts = size(map_IDTI, 2);
nFeatLevel = size(i_resp, 2);
c = i_mdl.c;

o_resp_DT = cell(nAllParts, nFeatLevel);
for i=2:nAllParts % run except for the root
    curMdl = getNode(map_IDTI(:, i), i_mdl);
    for l=1:nFeatLevel
        resp = struct('score', 0, 'Ix', [], 'Iy', []); 
        
%         w_def = ((c==1)*1 + (c==0)*(-1))*curMdl.w_def; %%FIXME: sign!
        w_def = ((c==1)*1 + (c==0)*(0))*curMdl.w_def; %%FIXME: sign!
        [resp.score, resp.Ix, resp.Iy] = bounded_dt(i_resp{i, l}, ...
            w_def(3), w_def(1), w_def(4), w_def(2), 4); %%FIXME: constant 4 here !!!!!!!!!!!!!
        o_resp_DT{i, l} = resp;                                    
    end
end
end