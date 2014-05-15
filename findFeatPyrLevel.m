function [o_level] = findFeatPyrLevel(i_featPyr, i_s)
% o_level = ismember([i_featPyr(:).scale], i_s);

% for codegen
o_level = 0;
for lInd=1:numel(i_featPyr)
    if i_featPyr(lInd).scale == i_s
        o_level = lInd;
        break;
    end
end
end