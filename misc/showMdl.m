function showMdl( i_params, i_objMdl )
%SHOWMDL Summary of this function goes here
%   Detailed explanation goes here
bs = 40;

% draw a root filter
w = reshape(i_objMdl.w_app, [i_objMdl.wh_cc(2), i_objMdl.wh_cc(1), numel(i_objMdl.w_app)/prod(i_objMdl.wh_cc)]);
if i_params.feat.HOX.type < 4
    w = w(:,:,1:9);
    scale = max(max(w(:)),max(-w(:)));
    mdlImg = HOGpicture(w, bs) * 255/scale;
else
    w = w(:,:,1:9);
    scale = max(max(w(:)),max(-w(:)));
    mdlImg = HOGpicture(w, bs, 9) * 255/scale;
end

title('Visualzation of the learnt model');
imagesc(mdlImg); 
colormap gray;
axis image;

% draw parts
for pInd=1:numel(i_objMdl.parts)
    curMdl = i_objMdl.parts(pInd);
    w = reshape(curMdl.w_app, [curMdl.wh_cc(2), curMdl.wh_cc(1), numel(curMdl.w_app)/prod(curMdl.wh_cc)]);
    
    if i_params.feat.HOX.type < 4
        w = w(:,:,1:9);
        scale = max(max(w(:)),max(-w(:)));
        partMdlImg = HOGpicture(w, bs) * 255/scale;
    else
        warning('not yet implemented');
        keyboard;
    end
    
    partMdlImg = imresize(im2double(partMdlImg), 1./(curMdl.ds*i_objMdl.ds));
    partMdlImg = makeimborder(partMdlImg, ceil(bs/20), inf);
    resSPnt = (curMdl.uv/i_params.feat.HOX.SqCellSize)*bs;
%     resEPnt = ((curMdl.uv+curMdl.wh-1)/i_params.feat.HOX.SqCellSize)*bs;
%     offPos = max(resEPnt - [size(mdlImg, 2); size(mdlImg, 1)], [0; 0]);
%     resSPnt = resSPnt - offPos;
    
    hold on; imagesc(resSPnt(1), resSPnt(2), partMdlImg); hold off;
    
end

end



