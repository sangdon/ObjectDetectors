function yhat = constraintCBFnc(param, model, xi, yi)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>

% globalParams = param; % trick for Matlab coder
globalParams = param.globalParams;

%% margin rescaling

% update w
% y = updMdlW(globalParams.objMdl, model.w);
y = updMdlW(yi, model.w);

% find yhat
maxScore = -inf;
maxPartMdl = y; % initialize
for c=[0 1]
    y_c = updMdlUVSC(y, [1; 1; 1; c]);
    
    loss = lossFnc(yi, y_c);
%     [meas, y_c_opt] = measPart( globalParams.feat.HoG.SqCellSize, globalParams.feat.HoG.type, xi, y_c, [1; 1; 1; c]);
    [meas, y_c_opt] = measPart_DT( xi, y_c, [1; 1; 1; c]);
        
    score = meas + loss;
    if maxScore < score
        maxScore = score;
        maxPartMdl = y_c_opt;
    end
end
yhat = maxPartMdl;

end