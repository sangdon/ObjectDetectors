function [o_objMdl] = trainSVMLight(i_params, i_patterns, i_labels, i_objMdl)
assert(numel(i_patterns) > 0);

exampleFN = 'example_file';
modelFN = 'model_file';

exFID = fopen(exampleFN, 'w');

% write examples
for pInd=1:numel(i_patterns)
    pattern = i_patterns{pInd};
    label = i_labels{pInd};
    
    fprintf(exFID, '%d ', label);
    for dInd=1:numel(pattern)
        if pattern(dInd) == 0
            continue;
        end
        fprintf(exFID, '%d:%f ', dInd, pattern(dInd));
    end
    fprintf(exFID, '\n');
end

if i_params.debug.verbose >= 1
    ssvmTicID = tic;
end
system(sprintf('svm_learn %s %s', exampleFN, modelFN));
if i_params.debug.verbose >= 1
    fprintf('- running time of SVM: %s sec.\n', num2str(toc(ssvmTicID)));
end



%% return
o_objMdl = i_objMdl;
o_objMdl.w_app = model_ssvm.w;
end

function delta = lossCB(param, y, ybar)
  delta = double(y ~= ybar) ;
%   if param.verbose
%     fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
%   end
end

function psi = featureCB(param, x, y)
  psi = sparse(y*x/2) ;
%   if param.verbose
%     fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
%             x, y, full(psi(1)), full(psi(2))) ;
%   end
end

function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end
%   if param.verbose
%     fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
%             model.w, x, y, yhat) ;
%   end
end


