function [ o_bbs, o_bbs_wbg ] = detect( i_mdl, i_img  )
%DETECT Summary of this function goes here
%   Detailed explanation goes here


if i_mdl.params.general.mdlType == 2
    [bbs, bbs_wbg] = detect_DPM(i_mdl.objMdl, i_img);
else
    [bbs, bbs_wbg] = detect_PM(i_mdl.objMdl, i_img);
end

%% return
o_bbs = bbs;
o_bbs_wbg = bbs_wbg;


end
