function [o_loss] = lossFnc(yi, y)
o_loss = double(yi.c ~= y.c);
end