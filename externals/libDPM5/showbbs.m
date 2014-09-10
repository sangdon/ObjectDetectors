% function showbbs(im, boxes, range, thres, cm, i_crop, i_showcb)
% function showbbs(im, boxes, nTop, type, range, cm)
function showbbs(im, boxes, nTop, varargin)
% Draw bounding boxes on top of an image.
%   showbbs(im, boxes, nTop, type, range, cm)
%   If out is given, a pdf of the image is generated (requires export_fig).
%       type
%           0: root
%           1: root + part

switch nargin
    case 3
        type = 0;
        scoreRange = [];
        cm = colormap(jet);
    case 4
        type = varargin{1};
        scoreRange = [];
        cm = colormap(jet);
    case 5
        type = varargin{1};
        scoreRange = varargin{2};
        cm = colormap(jet);
    case 6
        type = varargin{1};
        scoreRange = varargin{2};
        cm = varargin{3};
    otherwise
        error('Invalid args!')
end

boxes = boxes(1:min(size(boxes, 1), nTop), :); 

nCM = size(cm, 1);
lineWidth = 3;
i_showcb = 1;
bbsize = 5;

image(im); 
axis image;
if isempty(scoreRange)
    switch type
        case 0
            scoreRange = [floor(min(boxes(:, end))-eps), ceil(max(boxes(:, end))+eps)];
        case 1
            scoreRange = [-1, 1];
    end
end


if isempty(boxes)
    return;
end

hold on;
nParts = (type==1)*((size(boxes, 2)-1)/bbsize) + (1-(type==1));
for bInd=size(boxes, 1):-1:1
    for pInd=1:nParts
        x1 = boxes(bInd, bbsize*(pInd-1)+1);
        y1 = boxes(bInd, bbsize*(pInd-1)+2);
        x2 = boxes(bInd, bbsize*(pInd-1)+3);
        y2 = boxes(bInd, bbsize*(pInd-1)+4);
        appScore = boxes(bInd, bbsize*(pInd-1)+5);
%         defScore = boxes(bInd, bbsize*(pInd-1)+6);
        defScore = 0;  %%FIXME: no def scores...
        totalScore = boxes(bInd, end);
        
        % appeaerance
        switch type
            case 0
                score = totalScore;
            case 1
                score = appScore/abs(totalScore); % contribution of appearance score
        end
        score = min(scoreRange(2), max(scoreRange(1), score));

        colorInd = round((score-scoreRange(1))/(scoreRange(2) - scoreRange(1))*(nCM-1) + 1);
        c = cm(colorInd, :);
        
        line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', c, 'linewidth', lineWidth, 'linestyle', '-');
        
        % deformation
        switch type
            case 0
                
            case 1
                score = defScore/abs(totalScore); % contribution of deformation score
                score = min(scoreRange(2), max(scoreRange(1), score));
                colorInd = round((score-scoreRange(1))/(scoreRange(2) - scoreRange(1))*(nCM-1) + 1);
                c = cm(colorInd, :);
                
                plot(mean([x1, x2]), mean([y1, y2]), 'color', c, 'Marker', 'o', 'MarkerFaceColor', c, 'MarkerSize', 8);
        end        
    end
end
hold off;
if i_showcb == 1
    cb_h = colorbar;
    set(cb_h, 'YTick', [1 10:10:nCM]);
    set(cb_h, 'YTickLabel', [scoreRange(1):(scoreRange(2)-scoreRange(1))/6:scoreRange(2)]);
end

end