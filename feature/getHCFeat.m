function [o_feat] = getHCFeat(i_im, i_sqCellSz, i_nColBin)

nrows = floor(size(i_im, 1)/i_sqCellSz);
ncols = floor(size(i_im, 2)/i_sqCellSz);
feat = mat2cell(i_im, ...
    [i_sqCellSz*ones(1, nrows-1) size(i_im, 1)-i_sqCellSz*(nrows-1)], ...
    [i_sqCellSz*ones(1, ncols-1) size(i_im, 2)-i_sqCellSz*(ncols-1)], ...
    [1 1 1]);
for k=1:size(feat, 3)
    for i=1:size(feat, 1)
        for j=1:size(feat, 2)
            feat{i, j, k} = histc(feat{i, j, k}(:), 0:1/i_nColBin:1); 
            feat{i, j, k} = reshape(feat{i, j, k}, [1 1 numel(feat{i, j, k})]);
            % color normalization????????????????????????????
        end
    end
end
o_feat = cell2mat(feat);
end