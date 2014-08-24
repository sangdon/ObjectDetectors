figure(1);

% for i=6:20
%     subplot_tight(3, 5, i-5);
%     img = im2double(imread(sprintf('~/UPenn/Dropbox/Research/SocialObject/testImg/image00006%02d.bmp', i))); 
%     [bbs, bbs_wbg] = detect(objMdl, img); topBBs = bbs(1:min(size(bbs, 1), 1), :);showbbs(img, topBBs, 0, [-10 10], colormap(jet));
%     if i==6
%         title(sprintf('#fr: 6%02d (also training img)', i));
%     else
%         title(sprintf('#fr: 6%02d', i));
%     end
% end

figure(2);
img = im2double(imread('~/UPenn/Research/Data/KidsData/9/No3/image/image0001049.bmp')); 
[bbs, bbs_wbg] = detect(objMdl, img); 
topBBs = bbs(1:min(size(bbs, 1), 2), :); showbbs(img, topBBs, 0, [0 3000], colormap(jet));

