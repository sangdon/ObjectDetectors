function [ o_im ] = imresize_scale_fast( i_im, i_scale )
%IMRESIZE Summary of this function goes here
%   Detailed explanation goes here
i_scale
o_im = resize(i_im, i_scale);
end

