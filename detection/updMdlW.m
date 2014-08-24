function [o_mdl] = updMdlW(i_mdl, i_w )
w = i_w;
y = i_mdl;

% y.w = w;

y.w_app = zeros(y.appFeatDim, 1);
y.w_app(1:y.appFeatDim) = w(1:y.appFeatDim);
w = w(y.appFeatDim+1:end);

for pInd=1:numel(y.parts)
    y.parts(pInd).w_app = zeros(y.parts(pInd).appFeatDim, 1);
    y.parts(pInd).w_app(1:y.parts(pInd).appFeatDim) = w(1:y.parts(pInd).appFeatDim);
    w = w(y.parts(pInd).appFeatDim+1:end);
    
    y.parts(pInd).w_def = zeros(y.parts(pInd).defFeatDim, 1);
    y.parts(pInd).w_def(1:y.parts(pInd).defFeatDim) = w(1:y.parts(pInd).defFeatDim);
    w = w(y.parts(pInd).defFeatDim+1:end);
    
%     y.parts(pInd).w_b = w(1);
%     w = w(2:end);
end
y.w_b = w(1);
w = w(2:end);

assert(isempty(w));
o_mdl = y;
end

