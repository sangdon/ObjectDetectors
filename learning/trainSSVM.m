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
o_objMdl = updMdlW(o_objMdl, model_ssvm.w);


end

function [o_params] = squeezeParams(i_params)
o_params = [];
o_params.feat.HoG.SqCellSize = i_params.feat.HoG.SqCellSize;
o_params.feat.HoG.type = i_params.feat.HoG.type;
end

function delta = lossCB(param, yi, ybar)

    delta = lossFnc(yi, ybar);
    
%     if param.verbose
%         fprintf('delta = loss(%3d, %3d) = %f\n', yi, ybar, delta) ;
%     end
end

function psi = featureCB(param, x, y)

    psi = sparse(getFeat(param.globalParams, x, y, [1; 1; 1; y.c]));
    
%     if param.verbose
%         fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
%             x, y, full(psi(1)), full(psi(2))) ;
%     end
end






