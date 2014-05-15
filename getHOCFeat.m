function [ o_feat ] = getHOCFeat( i_img, i_cellSz, i_cachFN )
%HOCFEAT Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    i_cachFN = [];
end

rsz = 1;
img = im2double(i_img);
nPbChannel = 4;
nPbScale = 3;
nOrient = 8;

%% mPb
[Pb11, Pb12, Pb13, Pb21, Pb22, Pb23, Pb31, Pb32, Pb33, Pb41, Pb42, Pb43, ~] = det_mPb(img, i_cachFN);
[chInd, scInd] = meshgrid(1:nPbChannel, 1:nPbScale);
chscIndSet = [chInd(:)'; scInd(:)'];
lPbs = cell(nPbChannel*nPbScale, 1);
for csInd=1:size(chscIndSet, 2)
    eval(sprintf('Pb = Pb%d%d;', chscIndSet(1, csInd), chscIndSet(2, csInd)));
    lPbs{csInd} = Pb;
end

% %% sPb
% 
% sPbs = cell(nPbChannel*nPbScale, 1);
% for csInd=1:size(chscIndSet, 2)
%     Pb = lPbs{csInd};
%     sPbs{csInd} = spectralPb(nonmax_channels(Pb), size(Pb), '', 8); 
% end

lPbs = [];
sPbs = cell(1, 1);

Pb = Pb11;
sPbs{1} = spectralPb(nonmax_channels(Pb), size(Pb), '', 8); 

%% binning
Pbs = [lPbs; sPbs];
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
            o_feat(i, j, 1+(pInd-1)*i_nOrient:i_nOrient+(pInd-1)*i_nOrient) = sum(sum(Pb_cc{i, j}, 1), 2);
        end
    end
end

end


