function [o_node] = getNode(i_treeInd, i_tree)
nDepth = numel(i_treeInd);

curNode = i_tree;
for d=1:nDepth
    if i_treeInd(d) == 0
        break;
    else
        curNode = curNode.parts(i_treeInd(d));
    end
    
end
o_node = curNode;
end