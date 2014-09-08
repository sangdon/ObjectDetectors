function [o_objMdl] = trainSSVM(i_params, i_patterns, i_labels, i_objMdl)
assert(numel(i_patterns) > 0);

%% learn using SSVM with SVM settings
ssvmParams.patterns = i_patterns;
ssvmParams.labels = i_labels;
ssvmParams.lossFn = @lossCB;
ssvmParams.constraintFn = @constraintCBFnc;
ssvmParams.featureFn = @featureCB;
% ssvmParams.kernelFn = @kernelCB;
ssvmParams.dimension = i_objMdl.featDim;
ssvmParams.verbose = i_params.debug.verbose;
ssvmParams.globalParams = squeezeParams(i_params);

if i_params.debug.verbose >= 1
    ssvmTicID = tic;
end
% -c C, -o <rescaling> -v <verbose> -w <algorith>
% -o 2: margin rescaling
% -w 3: 1-slack algorithm (dual) described in [5]
% -w 4: 1-slack algorithm (dual) with constraint cache [5]
% model_ssvm = svm_struct_learn(sprintf('-c %f -o 2 -v %d -w 4 -e %f', i_params.training.C, i_params.debug.verbose, i_params.training.tol), ssvmParams) ;
% model_ssvm = svm_struct_learn(sprintf('-c %f -o 2 -v %d -w 4 -e %f -t 2 -g 100', i_params.training.C, i_params.debug.verbose, i_params.training.tol), ssvmParams);
model_ssvm = svm_struct_learn_Vedaldi(sprintf('-c %f -o 2 -v %d -w 4 -e %f', i_params.training.C, i_params.debug.verbose, i_params.training.tol), ssvmParams);
% model_ssvm = svm_struct_learn(sprintf('-c %f -o 2 -v %d -w 4 -e %f', i_params.training.C, i_params.debug.verbose, i_params.training.tol), ssvmParams);

if i_params.debug.verbose >= 1
    fprintf('- running time of SSVM: %s sec.\n', num2str(toc(ssvmTicID)));
end

%% return
o_objMdl = updMdlUVSC(i_objMdl, [], [], 1);
w = calcW(model_ssvm);
o_objMdl = updMdlW(i_params.general.mdlType, o_objMdl, w);
% o_objMdl = updMdlW(i_params.general.mdlType, o_objMdl, model_ssvm.w);


end

function [o_params] = squeezeParams(i_params)
o_params = [];
o_params.general.mdlType = i_params.general.mdlType;
o_params.feat.HOX.SqCellSize = i_params.feat.HOX.SqCellSize;
o_params.feat.HOX.type = i_params.feat.HOX.type;
o_params.feat.HOX.partResRatio = i_params.feat.HOX.partResRatio;
end

function delta = lossCB(param, yi, ybar)

delta = lossFnc(yi, ybar);

end

function [o_loss] = lossFnc(yi, y)

o_loss = lossFn_root(yi, y) + lossFn_parts(yi, y);
end

function [o_loss, o_n] = lossFn_root(yi, y)
o_loss = double(yi.c ~= y.c);
o_n = 1;
end

function [o_loss, o_n] = lossFn_parts(yi, y)
o_n = numel(yi.parts);
loss = 0;
for pInd=1:numel(yi.parts)
    loss = loss + lossFn_part(yi.parts(pInd), y.parts(pInd));
end
o_loss = loss;
end

function [o_loss] = lossFn_part(yi_part, y_part)
overlapThres = 0.5; %%FIXME: constant here!
loss = 0;
if yi_part.c ~= y_part.c
    loss = loss + 1;
else
    uvwh_i = [yi_part.uv_cc; yi_part.wh_cc];
    uvwh = [y_part.uv_cc; y_part.wh_cc];
    oa_int = rectint(uvwh_i', uvwh');
    oa_uni = uvwh_i(3)*uvwh_i(4) + uvwh(3)*uvwh(4) - oa_int;
    oa_norm = oa_int/oa_uni;
    if yi_part.c == 1
        % c = 1: check similarity
        loss = loss + double(oa_norm<overlapThres);
    else
        % c = 0: check dissimilarity
        loss = loss + 0; %%FIXME: correct?
    end
end
o_loss = loss;
end

function psi = featureCB(param, x, y)
    
psi = sparse(getFeat(param.globalParams, x, y, []));
if y.c == 0 %%FIXME: put into the getFeat.m?
%     psi = -psi;
    psi = sparse(zeros(numel(psi)));
end
end

function k = kernelCB(param, x, y, xp, yp)
gamma = 10;
u = getFeat(param.globalParams, x, y, []);
v = getFeat(param.globalParams, xp, yp, []);
k = exp(-gamma*norm(u-v));
end


function yhat = constraintCBFnc(param, model, xi, yi)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y)-(<psi(x,yi), w> - <psi(x,y), w>)
% margin rescaling: argmax_{ybar} loss(y,ybar)+psi(x,ybar)
% 
%   margin rescaling
%   note that this code should be consistent with detect_DPM.m

globalParams = param.globalParams;
w = model.w;
partResolution = globalParams.feat.HOX.partResRatio;
map_IDTI = yi.map_IDTI;
nAllParts = size(map_IDTI, 2);

%% update w
yi_w = updMdlW(globalParams.general.mdlType, yi, w);

%% precompute filter responses and generalized distance transform

%% <w, psi(xi, yi)>: don't need to compute explicilty
uv_cc_root = yi_w.uv_cc;
assert(all(uv_cc_root == [1; 1]));
% psi_i = getFeat( globalParams, xi, yi, [] );
% wspi_i = w(1:numel(psi_i))'*psi_i;

%% max_y delta(yi, y) + <w, psi(xi, y)>
score_max = -inf;
yhat_max = yi_w; % initialize
for c=[0 1]
    score_tot = 0;
    % update c
    y = updMdlUVSC(yi_w, [], [], c); % uv and s are fixed
    % compute app response
    resp_app_c = getAppFilterResp(xi.featPyr, y); % be careful of the value c
    % check a root
    pnInd = 1;
    sInd = 1;
    rootMdl = getNode(map_IDTI(:, pnInd), y);
    curResp = resp_app_c{pnInd, sInd};
    assert(numel(curResp) == 1);
    xy_cc = [1 1];
    score_root = curResp;
    score_tot = score_tot + score_root + lossFn_root(yi, y);
    % update y
    rootMdl.uv_cc = xy_cc';
    y = setNode(y, map_IDTI(:, pnInd), rootMdl);
    % enumerate all possible part locations
    [~, resp_apploss_c] = computePartLossMap(resp_app_c, y); % be careful of the value c
    resp_applossdef_c = applyDT(resp_apploss_c, y); % be careful of the value c
    % find maximul part locations
    for pnInd=2:nAllParts 
        curPart = getNode(map_IDTI(:, pnInd), y);
        psInd = 2; %%FIXME: constant
        % update the score
        curLevel = psInd;
        respSz = size(resp_app_c{pnInd, psInd});
        assert(all(respSz == size(resp_applossdef_c{pnInd, curLevel}.score)));

        %%FIXME: consistent with getAnchor.m: i_child.ds*i_parent.uv_cc + i_child.dudv_cc - 1;
        anchorPos_part_cc = bsxfun(@plus, partResolution*xy_cc - 1, curPart.dudv_cc') - 1;
        valInd = ... %%FIXME: really required?? bugs??
                anchorPos_part_cc(:, 1) >= 1 & anchorPos_part_cc(:, 1) <= respSz(2) & ...
                anchorPos_part_cc(:, 2) >= 1 & anchorPos_part_cc(:, 2) <= respSz(1);
        ind = sub2ind(size(resp_applossdef_c{pnInd, curLevel}.score), anchorPos_part_cc(valInd, 2), anchorPos_part_cc(valInd, 1));
        % uv
        xy_part_cc = ones(size(anchorPos_part_cc, 1), 2);
        xy_part_cc(valInd, :) = double([...
            resp_applossdef_c{pnInd, curLevel}.Ix(ind) resp_applossdef_c{pnInd, curLevel}.Iy(ind)]);
        % scores
        score_part = ones(size(anchorPos_part_cc, 1), 1)*-inf;
        ind = sub2ind(size(resp_applossdef_c{pnInd, curLevel}.score), xy_part_cc(valInd, 2), xy_part_cc(valInd, 1));
        score_part(valInd) = resp_applossdef_c{pnInd, curLevel}.score(ind);
        assert(numel(score_part) == 1);
        % update yi_w_c and score
        curPart.uv_cc = xy_part_cc(:);
        y = setNode(y, map_IDTI(:, pnInd), curPart);
        score_tot = score_tot + score_part;
    end
   
    % update y_max 
    if score_max < score_tot
        score_max = score_tot;
        yhat_max = y;
    end 
end

%% return
yhat = yhat_max;
end

% function yhat = constraintCBFnc_old(param, model, xi, yi)
% % slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% % margin rescaling: argmax_y delta(yi, y)-(<psi(x,yi), w> - <psi(x,y), w>)
% % margin rescaling: argmax_{ybar} loss(y,ybar)+psi(x,ybar)
% 
% 
% globalParams = param.globalParams;
% 
% %% margin rescaling
% 
% % if sum(model.w) ~= 0
% %     keyboard;
% % end
% 
% % update w
% if size(model.svPatterns, 2) == 0
%     w = zeros(param.dimension, 1);
% else
%     w = calcW(model);
% end
% y = updMdlW(globalParams.general.mdlType, yi, w);
% % y = updMdlW(globalParams.general.mdlType, yi, model.w);
% 
% % find yhat
% maxScore = -inf;
% maxPartMdl = y; % initialize
% for c=[0 1]
%     y_c = updMdlUVSC(y, [1; 1; 1; c]); % uv, s, is fixed
%     
%     loss = lossFnc(y, y_c);
% %     [meas, y_c_opt] = measPart( globalParams.feat.HOX.SqCellSize, globalParams.feat.HOX.type, xi, y_c, [1; 1; 1; c]);
%     [meas, y_c_opt] = measPart_DT( xi, y_c, []);
%         
%     score = meas + loss; 
%     if maxScore < score
%         maxScore = score;
%         maxPartMdl = y_c_opt;
%     end
% end
% yhat = maxPartMdl;
% 
% end

function w = calcW(model)
if isempty(model.w)
    w = [model.svPatterns{:}] * (model.alpha .* [model.svLabels{:}]') / 2 ;
else
    w = model.w;
end
end


function [o_map, o_apploss] = computePartLossMap(i_resp_app, yi)
map_IDTI = yi.map_IDTI;
nAllParts = size(map_IDTI, 2);
nFeatLevel = size(i_resp_app, 2);

%% compute a loss map
o_map = cell(nAllParts, nFeatLevel);
for pnInd=2:nAllParts
    curPart = getNode(map_IDTI(:, pnInd), yi);
    for l=2:nFeatLevel %%FIXME: assume part is at the second level
        map = zeros(size(i_resp_app{pnInd, l}));
        for u=1:size(i_resp_app{pnInd, l}, 2)
            for v=1:size(i_resp_app{pnInd, l}, 1)
                y_part = curPart;
                y_part.uv_cc = [u; v];
                map(v, u) = lossFn_part(curPart, y_part);
            end
        end
        o_map{pnInd, l} = map;
    end
end

%% add a loss map with an app resp
o_apploss = i_resp_app;
for pnInd=2:nAllParts
    for l=2:nFeatLevel %%FIXME: assume part is at the second level
        o_apploss{pnInd, l} =  o_apploss{pnInd, l} + o_map{pnInd, l};
    end
end
end


