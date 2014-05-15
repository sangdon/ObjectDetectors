function [ o_feat ] = getHOmbCFeat( i_img, i_cellSz, i_cacheFN )
%HOCFEAT Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    i_cacheFN = [];
end

rsz = 1;
img = min(1, max(0, im2double(i_img))); % to avoid segmentation fault

nPbChannel = 1;
nPbScale = 3;
nOrient = 8;
nvec = 8;

%% mPb
% compute cues - standard scales
[Pb11, Pb12, Pb13] = det_mbPb(img, i_cacheFN);
% % smooth cues
% gtheta = ...
%      [1.5708    1.1781    0.7854    0.3927 ...
%   0         2.7489    2.3562    1.9635];
% for o = 1:nOrient
%     Pb11(:,:,o) = fitparab(Pb11(:,:,o),3,3/4,gtheta(o));
%     Pb12(:,:,o) = fitparab(Pb12(:,:,o),5,5/4,gtheta(o));
%     Pb13(:,:,o) = fitparab(Pb13(:,:,o),10,10/4,gtheta(o));
% end

% compute grid sizes
sz_im = [size(img,1) size(img,2)];
sigma = 2;
sz1 = sz_im;
sz2 = round(1./sigma.*sz1);
sz3 = round(1./sigma.*sz2);
n1 = prod(sz1);
% compute sampling matrices
S2 = grid_sample(sz2, sz1);
S3 = grid_sample(sz3, sz1);
% resize pb signals
pb1_all_rsz = Pb11;
pb2_all_rsz = reshape(S2*reshape(Pb12, [n1 8]), [sz2 8]);
pb3_all_rsz = reshape(S3*reshape(Pb13, [n1 8]), [sz3 8]);
% nonmax suppress
pb1_nmax = nonmax_channels(pb1_all_rsz,pi/16);
pb1_nmax = max(0, min(1, 1.2*pb1_nmax));
pb2_nmax = nonmax_channels(pb2_all_rsz,pi/16);
pb2_nmax = max(0, min(1, 1.2*pb2_nmax));
pb3_nmax = nonmax_channels(pb3_all_rsz,pi/16);
pb3_nmax = max(0, min(1, 1.2*pb3_nmax));
% store pb
lPbs{1} = pb1_nmax;
lPbs{2} = pb2_nmax;
lPbs{3} = pb3_nmax;

% [chInd, scInd] = meshgrid(1:nPbChannel, 1:nPbScale);
% chscIndSet = [chInd(:)'; scInd(:)'];
% lPbs = cell(nPbChannel*nPbScale, 1);
% for csInd=1:size(chscIndSet, 2)
%     eval(sprintf('Pb = Pb%d%d;', chscIndSet(1, csInd), chscIndSet(2, csInd)));
%     lPbs{csInd} = Pb;
% end


%% sPb
[C_arr, Theta_arr, U_arr] = multiscale_ic(lPbs);
% progressive multigrid multiscale eigensolver
opts = struct( ...
   'k', [1 1 1], ...
   'k_rate', sqrt(3), ... %sqrt(2)
   'tol_err', 1e-1, ...
   'disp', false ...
);
[evecs, evals, info] = ae_multigrid(C_arr, Theta_arr, U_arr, nvec, opts);
% spectral pb extraction from eigenvectors
[spb_arr, spbo_arr, spb, spbo, spb_nmax] = multiscale_spb(evecs, evals, lPbs);

% % display spectral pb results
% figure(1); imagesc(img); axis image; axis off; title('Image');
% figure(22);
% subplot(1,3,1); imagesc(spb_arr{1}); axis image; axis off; title('sPb (coarse)');
% subplot(1,3,2); imagesc(spb_arr{2}); axis image; axis off; title('sPb (medium)');
% subplot(1,3,3); imagesc(spb_arr{3}); axis image; axis off; title('sPb (fine)');

% rescale sPbs
sPbs = [];
for ind=1:numel(spbo_arr)
    sPbs{ind} = imresize(spbo_arr{ind}, sz_im, 'bilinear');
end

% sPbs = cell(nPbChannel*nPbScale, 1);
% for csInd=1:size(chscIndSet, 2)
%     Pb = lPbs{csInd};
%     if rsz < 1
%         Pb_rszd = imresize(Pb(:, :, 1), rsz);
%         for oInd=2:nOrient
%             Pb_rszd(:, :, oInd) = imresize(Pb(:, :, oInd), rsz);
%         end
%     else
%         Pb_rszd = Pb;
%     end
%     sPbs{csInd} = spectralPb(nonmax_channels(Pb_rszd), size(Pb), '', nvec); 
% end

% sPbs = cell(nPbChannel, 1);
% for csInd=1
%     Pb = lPbs{csInd};
%     if rsz < 1
%         Pb_rszd = imresize(Pb(:, :, 1), rsz);
%         for oInd=2:nOrient
%             Pb_rszd(:, :, oInd) = imresize(Pb(:, :, oInd), rsz);
%         end
%     else
%         Pb_rszd = Pb;
%     end
%     sPbs{csInd} = spectralPb(nonmax_channels(Pb_rszd), size(Pb), '', nvec); 
% end

%% binning
% Pbs = [lPbs; sPbs];
Pbs = sPbs(:);
o_feat = convPix2Cell(Pbs, i_cellSz, [size(img, 2), size(img, 1)], nOrient);

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
            o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = oriHist/sqrt(sum(oriHist.^2));
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
