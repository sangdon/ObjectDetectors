function showLabels_mdl( i_img, i_obj )
%SHOWLABELS Summary of this function goes here
%   Detailed explanation goes here
img = i_img;
imshow(img);
axis on; axis image; 


objs = i_obj;
for oInd=1:numel(objs)
    bndbox = objs(oInd).bndbox;

    hold on;
    rectangle('Position', [bndbox.xmin bndbox.ymin bndbox.xmax-bndbox.xmin bndbox.ymax-bndbox.ymin], 'EdgeColor', 'g');
    hold off;
    
    parts = objs(oInd).parts;
    for pInd=1:numel(parts)
        bndbox = parts(pInd).bndbox;
        
        hold on;
        rectangle('Position', [bndbox.xmin bndbox.ymin bndbox.xmax-bndbox.xmin bndbox.ymax-bndbox.ymin], 'EdgeColor', 'g');
        hold off;
        
    end
end

end

