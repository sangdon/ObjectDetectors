function [o_stat, o_pasDB_tp, o_pasDB_fp] = evaluate( i_params, i_objCls, i_pasDB_gt, i_pasDB_det )
%EVALUATE Summary of this function goes here
%   Detailed explanation goes here

% %% find cahched results
% cacheFN = sprintf('%s/eval.mat', i_params.results.cachingDir);
% if i_params.general.enableCaching && exist(cacheFN, 'file') && i_params.training.hardNegMining == 0
%     load(cacheFN);
%     return;
% end

%% evaluate
if i_params.debug.verbose>=0
    figure(30001);
end
[rec_gt, prec, ap, o_pasDB_tp, o_pasDB_fp] = VOCevaldet_simple(i_pasDB_gt, i_pasDB_det, i_objCls, i_params.eval.minOverlap, i_params.debug.verbose>=0);
if i_params.debug.verbose>=0
    fprintf('- AP: %.3f\n', ap);
    saveas(30001, [i_params.results.intResDir '/prCurve'], 'png');
end

o_stat = [];
o_stat.recall = rec_gt;
o_stat.prec = prec;
o_stat.ap = ap;
o_stat.objCls = i_objCls;


if i_params.general.enableCaching
    cacheFN = sprintf('%s/eval.mat', i_params.results.cachingDir);
    save(cacheFN, 'o_stat', 'o_pasDB_tp', 'o_pasDB_fp');
end
end

