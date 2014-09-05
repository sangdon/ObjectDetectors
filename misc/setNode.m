function [o_tree] = setNode(i_tree, i_treeInd, i_node)
o_tree = i_tree;

if i_treeInd(1) == 0
    o_tree = i_node;
else
    o_tree.parts(i_treeInd(1)) = i_node;
end
end