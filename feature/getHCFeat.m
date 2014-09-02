function [o_feat] = getHCFeat(i_im, i_sqCellSz, i_nColBin)

nrows = floor(size(i_im, 1)/i_sqCellSz);
ncols = floor(size(i_im, 2)/i_sqCellSz);
ndep = size(i_im, 3)*i_nColBin;
% feat = mat2cell(i_im, ...
%     [i_sqCellSz*ones(1, nrows-1) size(i_im, 1)-i_sqCellSz*(nrows-1)], ...
%     [i_sqCellSz*ones(1, ncols-1) size(i_im, 2)-i_sqCellSz*(ncols-1)], ...
%     [1 1 1]);
o_feat = zeros(nrows, ncols, ndep);
for k=1:size(i_im, 3)
    d_ind = 1+(k-1)*i_nColBin:k*i_nColBin;
    for i=1:nrows
        i_ind = 1+(i-1)*i_sqCellSz:i*i_sqCellSz;
        for j=1:ncols
            j_ind = 1+(j-1)*i_sqCellSz:j*i_sqCellSz;
%             feat{i, j, k} = histc(feat{i, j, k}(:), 0:1/i_nColBin:1); 
%             feat{i, j, k} = reshape(feat{i, j, k}, [1 1 numel(feat{i, j, k})]);

            % color normalization????????????????????????????
            
            curFeat = i_im(i_ind, j_ind, k);
            hc = histc(curFeat(:), 0:1/i_nColBin:1); 
            hc(end-1) = hc(end-1)+hc(end);
            for l=1:i_nColBin
                o_feat(i, j, d_ind(l)) = hc(l);
            end
        end
    end
end
% o_feat = cell2mat(feat);
end