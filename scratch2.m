srcDir = '/data/v50/sangdonp/FacadeRecognition/data/RCTA_shop_loop_doors_VOC/JPEGImages/';
resDir = '~/UPenn/Dropbox/5-12-14/trainedOnNormalBackdoor_sangdon/';
bbsDir = '/data/v50/sangdonp/objectDetection/exp_NPM_RCTA_shop_loop_doors/door/resultsBbs/';

nTop = 3;

filelist = dir(srcDir);
for fInd=1:numel(filelist)
    if filelist(fInd).isdir
        continue;
    end
    [~, imgID] = fileparts(filelist(fInd).name);
    imgName = filelist(fInd).name;
    
    img = im2double(imread(sprintf('%s/%s', srcDir, imgName))); 
    bbs = detect(params, img, objMdl); 
    
    % save
    save(sprintf('%s/%s.mat', bbsDir, imgID), 'bbs');
    
    % show
    sfigure(2); clf;
    topbbs = bbs(1:min(nTop, size(bbs, 1)), :); 
    showbbs(img, topbbs, 0, [min(topbbs(:, end))-10 max(topbbs(:, end))+10], colormap(jet));
    
    saveas(2, sprintf('%s/%s', resDir, imgName));
end