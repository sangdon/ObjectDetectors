function [ o_feat ] = getHObCFeat( i_img, i_cellSz, i_cacheFN )
%HOCFEAT Summary of this function goes here
%   Detailed explanation goes here

nOrient = 9;
nvec = 16;
stdSz = [100; 100];
imsz_ori = [size(i_img, 1) size(i_img, 2)];
rsz_Pb = min(1, sqrt(prod(stdSz)/prod(imsz_ori)));

if nargin < 3
    i_cacheFN = [];
end

i_img = im2double(i_img);
% i_img = min(1, max(0, i_img)); % to avoid segmentation fault

if rsz_Pb == 1
    img_rsz = i_img;
else
    img_rsz = imresize(i_img, rsz_Pb);
end
img_rsz = min(1, max(0, img_rsz)); % to avoid segmentation fault


%% bPb
% bPb_c = mex_pb_parts_final_selected_b_3(img_rsz(:,:,1),img_rsz(:,:,2),img_rsz(:,:,3));
% bPb = cell2mat(reshape(bPb_c, [1, 1, nOrient]));

[bPb1_c, bPb2_c, bPb3_c] = mex_pb_parts_final_selected_b_9ori(img_rsz(:,:,1),img_rsz(:,:,2),img_rsz(:,:,3));
bPbs{1} = cell2mat(reshape(bPb1_c, [1, 1, nOrient]));
% bPbs{1} = bPbs{1}./(max(bPbs{1}(:))+eps);
bPbs{2} = cell2mat(reshape(bPb2_c, [1, 1, nOrient]));
% bPbs{2} = bPbs{2}./(max(bPbs{2}(:))+eps);
bPbs{3} = cell2mat(reshape(bPb3_c, [1, 1, nOrient]));
% bPbs{3} = bPbs{3}./(max(bPbs{3}(:))+eps);

%% sPb
bPbs_nns{1} = nonmax_channels(bPbs{1});
bPbs_nns{2} = imresize(nonmax_channels(bPbs{2}), 0.5);
bPbs_nns{3} = imresize(nonmax_channels(bPbs{3}), 0.5^2);
% sPbImg_in = imresize(nonmax_channels(bPb), rsz_sPb);
% sPbImg_in = nonmax_channels(bPb);
% sPbs = spectralPb_BSR(nonmax_channels(bPb), imsz_ori, '', nvec);
sPbs = spectralPb_multigrid(bPbs_nns, nvec, nOrient);

%% binning
% Pbs = [bPbs(:); sPbs(:)];
Pbs = [sPbs(3); bPbs{1}];
o_feat = convPix2Cell(Pbs, i_cellSz, rsz_Pb, [size(i_img, 2) size(i_img, 1)], nOrient);

end

function [sPbs] = spectralPb_multigrid(pb_arr, nvec, nOrient)
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

% [spb_arr, sPbs, spb, spbo, spb_nmax] = multiscale_spb(evecs, evals, pb_arr, nOrient);
sPbs = multiscale_spb_HOC(evecs, evals, pb_arr, nOrient);
end

function [o_feat] = convPix2Cell(i_Pbs, i_cellSz, i_scale, i_imgWH, i_nOrient)
cellWH = floor(i_imgWH/i_cellSz);
cellSz_rsz = floor(i_cellSz*i_scale);

nPbs = numel(i_Pbs);
o_feat = zeros(cellWH(2), cellWH(1), nPbs*i_nOrient);


for pInd=1:nPbs
    Pb = i_Pbs{pInd};
    Pb = Pb(1:cellWH(2)*cellSz_rsz, 1:cellWH(1)*cellSz_rsz, :);
    [maxMag, maxOri] = max(Pb, [], 3);
    
    
    for i=1:cellWH(2)
        for j=1:cellWH(1)
            curOri = maxOri((i-1)*cellSz_rsz+1:i*cellSz_rsz, (j-1)*cellSz_rsz+1:j*cellSz_rsz);
            curMag = maxMag((i-1)*cellSz_rsz+1:i*cellSz_rsz, (j-1)*cellSz_rsz+1:j*cellSz_rsz);
            for o=1:i_nOrient
                o_feat(i, j, o) = sum(curMag(curOri == o));
            end
        end
    end
    % normalization per each channel
    o_feat(:, :, 1+(pInd-1)*i_nOrient:pInd*i_nOrient) = o_feat(:, :, 1+(pInd-1)*i_nOrient:pInd*i_nOrient)/(max(max(sum(o_feat(:, :, 1+(pInd-1)*i_nOrient:pInd*i_nOrient), 3)))+eps); 
end
end

% function [o_feat] = convPix2Cell(i_Pbs, i_cellSz, i_scale, i_imgWH, i_nOrient)
% cellWH = floor(i_imgWH/i_cellSz);
% cellSz_rsz = floor(i_cellSz*i_scale);
% 
% nPbs = numel(i_Pbs);
% o_feat = zeros(cellWH(2), cellWH(1), nPbs*i_nOrient);
% 
% for pInd=1:nPbs
%     Pb = i_Pbs{pInd};
%     Pb = Pb(1:cellWH(2)*cellSz_rsz, 1:cellWH(1)*cellSz_rsz, :);
%     Pb_cc = mat2cell(Pb, ones(1, cellWH(2))*cellSz_rsz, ones(1, cellWH(1))*cellSz_rsz, size(Pb, 3));
% 
% %     maxE = -inf;
%     for i=1:cellWH(2)
%         for j=1:cellWH(1)
%             
%             oriHist = sum(sum(Pb_cc{i, j}, 1), 2);
%             
% %             maxE = max(maxE, sqrt(sum(oriHist.^2)));
% %             o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = oriHist;
%             
%             topN = 1;
%             [sortedVal, sortedOriInd] = sort(oriHist, 'descend');
%             o_feat(i, j, sortedOriInd(1:topN)+(pInd-1)*i_nOrient) = 1;
% %             o_feat(i, j, sortedOriInd(1:topN)+(pInd-1)*i_nOrient) = sortedVal(1:topN);
%             
% %             o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = oriHist/sqrt(sum(oriHist.^2));
% %             o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = oriHist/sum(oriHist);
%         end
%     end
% %     o_feat(:, :, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = o_feat(:, :, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient)./maxE;
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
