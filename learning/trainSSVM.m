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
model_ssvm = svm_struct_learn_Vedaldi(sprintf('-c %f -o 2 -v %d -w 4 -e %f', i_params.training.C, i_params.debug.verbose, i_params.training.tol), ssvmParams);
% model_ssvm = svm_struct_learn(sprintf('-c %f -o 2 -v %d -w 4 -e %f -t 2 -g 100', i_params.training.C, i_params.debug.verbose, i_params.training.tol), ssvmParams);

if i_params.debug.verbose >= 1
    fprintf('- running time of SSVM: %s sec.\n', num2str(toc(ssvmTicID)));
end

%% return
o_objMdl = i_objMdl;
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
overlapThres = 0.5; %%FIXME: constant here!

o_n = numel(yi.parts);
loss = 0;
for pInd=1:numel(yi.parts)
    if yi.parts(pInd).c ~= y.parts(pInd).c
        loss = loss + 1;
    else
        uvwh_i = [yi.parts(pInd).uv_cc; yi.parts(pInd).wh_cc];
        uvwh = [y.parts(pInd).uv_cc; y.parts(pInd).wh_cc];
        oa_int = rectint(uvwh_i, uvwh);
        oa_uni = uvwh_i(3)*uvwh_i(4) + uvwh(3)*uvwh(4) - oa_int;
        oa_norm = oa_int/oa_uni;
        if yi.parts(pInd).c == 1
            % c = 1: check similarity
            loss = loss + oa_norm<overlapThres;
        else
            % c = 0: check dissimilarity
            loss = loss + oa_norm>verlapThres;
        end
    end
end
o_loss = loss;
end

function psi = featureCB(param, x, y)
    
psi = sparse(getFeat(param.globalParams, x, y, []));
if y.c == 0 %%FIXME: put into the getFeat.m?
    psi = -psi;
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
%%FIXME: inefficient to call every time, but no chice
resp = getAppFilterResp(xi.featPyr, yi_w);
if nAllParts > 1
    resp_DT = applyDT(resp, yi_w);
end

%% <w, psi(xi, yi)>: don't need to compute explicilty
uv_cc_root = yi_w.uv_cc;
assert(all(uv_cc_root == [1; 1]));
% psi_i = getFeat( globalParams, xi, yi, [] );
% wspi_i = w(1:numel(psi_i))'*psi_i;


%% max_y delta(yi, y) + <w, psi(xi, y)>
score_max = -inf;
yhat_max = yi_w; % initialize
for c=[0 1]
    % update c
    yi_w_c = updMdlUVSC(yi_w, [], [], c); % uv and s are fixed
    % enumerate all possible part locations
    
    
    
end



%% return
yhat = yhat_max;







% find yhat
maxScore = -inf;
maxPartMdl = y; % initialize
for c=[0 1]
    y_c = updMdlUVSC(y, [1; 1; 1; c]); % uv, s, is fixed
    
    loss = lossFnc(y, y_c);
%     [meas, y_c_opt] = measPart( globalParams.feat.HOX.SqCellSize, globalParams.feat.HOX.type, xi, y_c, [1; 1; 1; c]);
    [meas, y_c_opt] = measPart_DT( xi, y_c, []);
        
    score = meas + loss; 
    if maxScore < score
        maxScore = score;
        maxPartMdl = y_c_opt;
    end
end
yhat = maxPartMdl;

end

function yhat = constraintCBFnc_old(param, model, xi, yi)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y)-(<psi(x,yi), w> - <psi(x,y), w>)
% margin rescaling: argmax_{ybar} loss(y,ybar)+psi(x,ybar)


globalParams = param.globalParams;

%% margin rescaling

% if sum(model.w) ~= 0
%     keyboard;
% end

% update w
if size(model.svPatterns, 2) == 0
    w = zeros(param.dimension, 1);
else
    w = calcW(model);
end
y = updMdlW(globalParams.general.mdlType, yi, w);
% y = updMdlW(globalParams.general.mdlType, yi, model.w);

% find yhat
maxScore = -inf;
maxPartMdl = y; % initialize
for c=[0 1]
    y_c = updMdlUVSC(y, [1; 1; 1; c]); % uv, s, is fixed
    
    loss = lossFnc(y, y_c);
%     [meas, y_c_opt] = measPart( globalParams.feat.HOX.SqCellSize, globalParams.feat.HOX.type, xi, y_c, [1; 1; 1; c]);
    [meas, y_c_opt] = measPart_DT( xi, y_c, []);
        
    score = meas + loss; 
    if maxScore < score
        maxScore = score;
        maxPartMdl = y_c_opt;
    end
end
yhat = maxPartMdl;

end

function w = calcW(model)
w = [model.svPatterns{:}] * (model.alpha .* [model.svLabels{:}]') / 2 ;
end


function [o_map] = computePartLossMap()
end


