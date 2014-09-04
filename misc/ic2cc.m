function [ o_pnt_cc ] = ic2cc( i_pnt_ic, i_sqCellSize )
%IC2CC Summary of this function goes here
%   Detailed explanation goes here

o_pnt_cc = ceil(i_pnt_ic/i_sqCellSize);

end

