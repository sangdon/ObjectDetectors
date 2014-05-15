function [ o_pq ] = getDeformation( i_parent, i_child )
%GETDEFORMATION Summary of this function goes here
%   Detailed explanation goes here
o_pq = i_child.ds*i_parent.uv_cc - 1 + i_child.dudv_cc;
end

