function im = HOGpicture(w, bs, nOri)
% Make picture of positive HOG weights.
%   im = HOGpicture(w, bs)

if nargin < 3
    nOri = 9; % the number of orient of HOG
end

% construct a "glyph" for each orientaion
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) nOri]);
bim(:,:,1) = bim1;
for i = 2:nOri
  bim(:,:,i) = imrotate(bim1, -(i-1)*180/nOri, 'crop');
end

% make pictures of positive weights bs adding up weighted glyphs
s = size(w);    
w(w < 0) = 0;    
im = zeros(bs*s(1), bs*s(2));
for i = 1:s(1),
  iis = (i-1)*bs+1:i*bs;
  for j = 1:s(2),
    jjs = (j-1)*bs+1:j*bs;          
    for k = 1:nOri
      im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
    end
  end
end
