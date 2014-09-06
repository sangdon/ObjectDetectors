function [o_objMdl] = trainSSVM(i_params, i_patterns, i_labels, i_objMdl)
assert(numel(i_patterns) > 0);

%% learn using SSVM with SVM settings
ssvmParams.patterns = i_patterns;
ssvmParams.labels = i_labels;
ssvmParams.lossFn = @lossCB;
ssvmParams.constraintFn = @constraintCBFnc;
ssvmParams.featureFn = @featureCB;
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

% tmpverbose = 1;
% model_ssvm = svm_struct_learn(sprintf('-c %f -o 2 -v %d -w 3', i_params.training.C, tmpverbose), ssvmParams) ;

if i_params.debug.verbose >= 1
    fprintf('- running time of SSVM: %s sec.\n', num2str(toc(ssvmTicID)));
end

%% return
o_objMdl = i_objMdl;
o_objMdl = updMdlW(i_params.general.mdlType, o_objMdl, model_ssvm.w);


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
    
%     if param.verbose
%         fprintf('delta = loss(%3d, %3d) = %f\n', yi, ybar, delta) ;
%     end
end

function [o_loss] = lossFnc(yi, y)
o_loss = double(yi.c ~= y.c);
end

function psi = featureCB(param, x, y)
    
psi = sparse(getFeat(param.globalParams, x, y, []));
end


function yhat = constraintCBFnc(param, model, xi, yi)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>

globalParams = param.globalParams;

%% margin rescaling

if sum(model.w) ~= 0
    keyboard;
end
% update w
y = updMdlW(globalParams.general.mdlType, yi, model.w);

% find yhat
maxScore = -inf;
maxPartMdl = y; % initialize
for c=[0 1]
    y_c = updMdlUVSC(y, [1; 1; 1; c]);
    
    loss = lossFnc(y, y_c);
%     [meas, y_c_opt] = measPart( globalParams.feat.HOX.SqCellSize, globalParams.feat.HOX.type, xi, y_c, [1; 1; 1; c]);
    [meas, y_c_opt] = measPart_DT( xi, y_c, []);
        
    score = meas + loss; % wierd but correct based on svm_struct_api.c
    if maxScore < score
        maxScore = score;
        maxPartMdl = y_c_opt;
    end
end
yhat = maxPartMdl;

end






