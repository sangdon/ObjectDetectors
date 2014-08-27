function [o_bbs] = slideWindow_conv( i_mdlMat, i_inMat, i_scoreThres ) %%%%%%%%%%%%% need c++
% save the bb: [xmin ymin xmax ymax appScore defScore]

mdlWH = [size(i_mdlMat, 2); size(i_mdlMat, 1)];
    
% convolve
resp = convn(i_inMat, i_mdlMat, 'same');

% generate bbs
val = resp > i_scoreThres;
[rows, cols] = find(val);
x = cols(:);
y = rows(:);
s = resp(val); s = s(:);

o_bbs = [x-(mdlWH(1)-1)/2 y-(mdlWH(2)-1)/2 x+(mdlWH(1)-1)/2 y+(mdlWH(2)-1)/2 s s];
end