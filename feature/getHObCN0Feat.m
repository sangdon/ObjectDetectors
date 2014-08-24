function [ o_feat ] = getHObCN0Feat( i_img, i_cellSz, i_cacheFN )
%HOCFEAT Summary of this function goes here
%   Detailed explanation goes here

nOrient = 8;
warning('change N ori');
nvec = 12;
stdSz = [50; 50];
imsz_ori = [size(i_img, 1) size(i_img, 2)];
rsz_Pb = sqrt(prod(stdSz)/prod(imsz_ori));

if nargin < 3
    i_cacheFN = [];
end

i_img = im2double(i_img);
% i_img = min(1, max(0, i_img)); % to avoid segmentation fault

img_rsz = imresize(i_img, rsz_Pb);
img_rsz = min(1, max(0, img_rsz)); % to avoid segmentation fault


%% bPb
% bPb_c = mex_pb_parts_final_selected_b_3(img_rsz(:,:,1),img_rsz(:,:,2),img_rsz(:,:,3));
% bPb = cell2mat(reshape(bPb_c, [1, 1, nOrient]));

[bPb1_c, bPb2_c, bPb3_c] = mex_pb_parts_final_selected_b(img_rsz(:,:,1),img_rsz(:,:,2),img_rsz(:,:,3));
bPbs{1} = nonmax_channels(cell2mat(reshape(bPb1_c, [1, 1, nOrient])));
bPbs{2} = imresize(nonmax_channels(cell2mat(reshape(bPb2_c, [1, 1, nOrient]))), 0.5);
bPbs{3} = imresize(nonmax_channels(cell2mat(reshape(bPb3_c, [1, 1, nOrient]))), 0.5^2);

%% sPb
% sPbImg_in = imresize(nonmax_channels(bPb), rsz_sPb);
% sPbImg_in = nonmax_channels(bPb);
% sPbs = spectralPb_BSR(nonmax_channels(bPb), imsz_ori, '', nvec);
sPbs = spectralPb_multigrid(bPbs, nvec);
%% binning
% Pbs = [lPbs; sPbs];
Pbs = sPbs(3);
o_feat = convPix2Cell(Pbs, floor(i_cellSz*rsz_Pb), [size(img_rsz, 2) size(img_rsz, 1)], nOrient);

end

function [sPbs] = spectralPb_multigrid(pb_arr, nvec)
% compute intervening contour
[C_arr, Theta_arr, U_arr] = multiscale_ic(pb_arr);
% progressive multigrid multiscale eigensolver
opts = struct( ...
   'k', [1 1 1], ...
   'k_rate', sqrt(3), ...
   'tol_err', 10.^-1, ...
   'disp', false ...
);

% uncomment this line to use ISPC sparse matrix * dense matrix implementation
% opts.use_ispc = 1;
% tic;
[evecs, evals, info] = ae_multigrid(C_arr, Theta_arr, U_arr, nvec, opts);
% time = toc;
% disp(['Wall clock time for eigensolver: ' num2str(time) ' seconds']);
% spectral pb extraction from eigenvectors
[spb_arr, sPbs, spb, spbo, spb_nmax] = multiscale_spb(evecs, evals, pb_arr);
end

function [o_feat] = convPix2Cell(i_Pbs, i_cellSz, i_imgWH, i_nOrient)
cellWH = floor(i_imgWH/i_cellSz);
nPbs = numel(i_Pbs);
o_feat = zeros(cellWH(2), cellWH(1), nPbs*i_nOrient);

for pInd=1:nPbs
    Pb = i_Pbs{pInd};
    Pb = Pb(1:cellWH(2)*i_cellSz, 1:cellWH(1)*i_cellSz, :);
    Pb_cc = mat2cell(Pb, ones(1, cellWH(2))*i_cellSz, ones(1, cellWH(1))*i_cellSz, size(Pb, 3));

    for i=1:cellWH(2)
        for j=1:cellWH(1)
            oriHist = sum(sum(Pb_cc{i, j}, 1), 2);
            o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = oriHist;
        end
    end
end
end

% function [o_feat] = convPix2Cell(i_Pbs, i_cellSz, i_imgWH, i_nOrient)
% cellWH = floor(i_imgWH/i_cellSz);
% nPbs = numel(i_Pbs);
% o_feat = zeros(cellWH(2), cellWH(1), nPbs*i_nOrient);
% 
% for pInd=1:nPbs
%     Pb = i_Pbs{pInd};
%     Pb = Pb(1:cellWH(2)*i_cellSz, 1:cellWH(1)*i_cellSz, :);
%     Pb_cc = mat2cell(Pb, ones(1, cellWH(2))*i_cellSz, ones(1, cellWH(1))*i_cellSz, size(Pb, 3));
% 
%     cellMaxL2Norm = -inf;
%     for i=1:cellWH(2)
%         for j=1:cellWH(1)
%             cellMaxL2Norm = max(cellMaxL2Norm, sqrt(sum(sum(sum(Pb_cc{i, j}, 1), 2).^2) + eps));
%         end
%     end
%     for i=1:cellWH(2)
%         for j=1:cellWH(1)
%             o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = sum(sum(Pb_cc{i, j}, 1), 2)./cellMaxL2Norm;
%             
%         end
%     end
% end
% end


% function [o_feat] = convPix2Cell(i_Pbs, i_cellSz, i_imgWH, i_nOrient)
% cellWH = floor(i_imgWH/i_cellSz);
% nPbs = numel(i_Pbs);
% o_feat = zeros(cellWH(2), cellWH(1), nPbs*i_nOrient);
% 
% for pInd=1:nPbs
%     Pb = i_Pbs{pInd};
%     Pb = Pb(1:cellWH(2)*i_cellSz, 1:cellWH(1)*i_cellSz, :);
%     Pb_cc = mat2cell(Pb, ones(1, cellWH(2))*i_cellSz, ones(1, cellWH(1))*i_cellSz, size(Pb, 3));
%     for i=1:cellWH(2)
%         for j=1:cellWH(1)
%             o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = sum(sum(Pb_cc{i, j}, 1), 2);
%         end
%     end
% end
% 
% end
% 
% 
