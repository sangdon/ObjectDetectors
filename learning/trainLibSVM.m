function [o_objMdl] = trainLibSVM(i_params, i_patterns, i_labels, i_objMdl)
assert(numel(i_patterns) > 0);

training_label_vector = cell2mat(i_labels);
training_instance_matrix = cell2mat(i_patterns')';

nPos = sum(training_label_vector == 1);
nNeg = sum(training_label_vector == -1);

if i_params.debug.verbose >= 1
    ssvmTicID = tic;
end
model = svmtrain(training_label_vector, training_instance_matrix, sprintf('-c %f t 0 -m 1000 -h 0 -w1 %f', i_params.training.C, nNeg/nPos));
if i_params.debug.verbose >= 1
    fprintf('- running time of libSVM: %s sec.\n', num2str(toc(ssvmTicID)));
end


%% return
o_objMdl = i_objMdl;
o_objMdl.w_app = (model.sv_coef' * full(model.SVs))';
end
