function [o_np] = getNParts(i_mdl)
o_np = 1;
o_np = o_np + numel(i_mdl.parts);
end