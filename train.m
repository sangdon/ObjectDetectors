function [ o_model ] = train( i_params, i_objCls, i_pasDB )
%TRAIN Summary of this function goes here
%   Detailed explanation goes here

cacheFN = sprintf('%s/train.mat', i_params.results.cachingDir);
if i_params.general.enableCaching && exist(cacheFN, 'file')
    load(cacheFN);
else
    %% supervised no-part model (NPM) or hyper-supervised DPM
    % train using easy-negatives
    [mdl_easy, pasDB_tr] = train_DPM(i_params, i_objCls, i_pasDB);
    if i_params.training.hardNegMining == 1
        % test on the training set
        [pasDB_tr, pasDB_tr_det] = test(i_params, mdl_easy, pasDB_tr);
        % choose hard negatives
        [~, ~, pasDB_fp] = evaluate(i_params, i_objCls, pasDB_tr, pasDB_tr_det);
        % train using hard-negatives
        newPasDB = chooseHardNegs(pasDB_fp, i_params.training.nMaxNegPerImg);
        newPasDB = convPasObjCls(newPasDB, i_objCls, [i_objCls '_neg']);
        newPasDB = mergePascalDB(pasDB_tr, newPasDB);

        [objMdl, ~] = train_DPM(i_params, i_objCls, newPasDB);
    else
        objMdl = mdl_easy;
    end
    o_model = struct(...
        'objMdl', objMdl, ...
        'params', i_params);
    
    if i_params.general.enableCaching
        save(cacheFN, 'o_model');
    end
end


if i_params.debug.verbose >= 1
    sfigure(10002); clf;
%     fr = o_model.w_app - min(o_model.w_app); fr = fr./max(fr); imagesc(reshape(fr, [o_model.wh_cc(2) o_model.wh_cc(1)]))
%     colormap(gray); axis image;
    showMdl(i_params, o_model.objMdl);
    if ~isempty(i_params.results.intResDir)
        saveFN = [i_params.results.intResDir '/learntHOG'];
        saveas(10002, saveFN, 'png');
        print(10002, '-depsc2', saveFN);
    end
    pause(0.1);
end

end

function [o_model, o_pasDB] = train_DPM(i_params, i_objCls, i_pasDB)
%% load db
if isempty(i_pasDB)
    pasDB = addObjPading(loadPascalDB(i_params, i_params.db.trainingSet), i_objCls, i_params.training.padSize);
else
    pasDB = i_pasDB;
    if isfield(pasDB, 'annotation')
        pasDB = pasDB.annotation;
    end
end

% warning('small number of training');
% pasDB = pasDB([1:100 3001:3010] );
oriPasDB = pasDB;


%% initialize
[pasDB, objMdl] = initParts(pasDB, i_objCls);

% get align parameters for bounding boxes
[pasDB, objMdl] = transObjsToMdlSpace(i_params, pasDB, objMdl);

% obtain hyper-supervision
[pasDB, objMdl] = getPartAnnotations(i_params, pasDB, objMdl);

% convert a tree data structure into a array data structure
objMdl = genMap_IDTI(objMdl);

% check feature dim to gather information
[pasDB, objMdl] = getFeatDim(i_params, pasDB, objMdl);

% check the total number of bbs for parfor
[nTotBB, dbobjIndMap] = getTotalNumBB(pasDB);


%% extract features for efficiency
cacheFN = sprintf('%s/train_feats.mat', i_params.feat.cachingDir);
if i_params.general.enableCaching && exist(cacheFN, 'file')
    load(cacheFN);
else
    patterns = cell(nTotBB, 1);
    patterns_ref = cell(nTotBB, 1);
    labels = cell(nTotBB, 1);
    bPosInd = false(nTotBB, 1);
    if i_params.debug.verbose >= 1
        fprintf('- extracting features for %d examples\n', size(dbobjIndMap, 2));
        showInterval = size(dbobjIndMap, 2)*0.1;
    end
    parfor dbobjInd=1:size(dbobjIndMap, 2)
%     for dbobjInd=1:size(dbobjIndMap, 2)

        if i_params.debug.verbose >= 1 %&& mod(dbobjInd-1, showInterval) == 0
            fprintf('- extract features: %d/%d...', dbobjInd, size(dbobjIndMap, 2));
            feTID = tic;
        end

        inds = dbobjIndMap(:, dbobjInd);
        dbInd = inds(1);
        oInd = inds(2);

        % get patterns and labels
        [curImg, curMdl] = getAlignedBBImg(...
            getPascalImg(i_params, pasDB(dbInd)), ...
            pasDB(dbInd).objects(oInd), ...
            objMdl.wh, ...
            pasDB(dbInd).objects(oInd).scale_psi, pasDB(dbInd).objects(oInd).resizedBB_psi);

        if i_params.general.mdlType == 1
            %% NPM

            pattern = getHOXFeat( curImg, i_params.feat.HoG.SqCellSize, i_params.feat.HoG.type );
            pattern = pattern(:);
            
%             if i_params.feat.HOG.type == 5
%                 pattern = pattern./norm(pattern)
%             end

            patterns{dbobjInd} = pattern;
            labels{dbobjInd} = (1)*(curMdl.c==1) + (-1)*(curMdl.c==0);

            if i_params.training.reflect == 1
                pattern = double(getHOXFeat(flipdim(im, 2), i_params.feat.HoG.SqCellSize, i_params.feat.HOGType));
                patterns_ref{dbobjInd} = pattern(:);
            end
        else
            %% DPM
            
            pattern = [];
            pattern.featPyr = getFeatPyr( ...
                curImg, ...
                [1 i_params.feat.partResRatio], ...
                @(img) getHOXFeat(img, i_params.feat.HoG.SqCellSize, i_params.feat.HoG.type));

            patterns{dbobjInd} = pattern;
            labels{dbobjInd} = updMdlW(squeezeMdl(curMdl), zeros(objMdl.featDim, 1));
        end

        bPosInd(dbobjInd) = curMdl.c == 1;


        if i_params.debug.verbose >= 1 %&& mod(dbobjInd-1, showInterval) == 0
            fprintf('%s sec. \n', num2str(toc(feTID)));
        end    
    end
    if i_params.general.enableCaching
        save(cacheFN, 'patterns', 'patterns_ref', 'labels', 'bPosInd');
    end
end

% merge flipped features
if i_params.training.reflect == 1
    patterns = [patterns; patterns_ref(bPosInd)];
    labels = [labels; labels(bPosInd)];
    bPosInd = [bPosInd; bPosInd(bPosInd)];
end

% balance a training set
if i_params.training.exampleDuplication == 1
    nPos = sum(bPosInd);
    nNeg = sum(~bPosInd);
    if nPos < nNeg
        nDup = nNeg - nPos;
        posInd = find(bPosInd);
        for d=1:nDup
            curPInd = posInd(mod(d-1, numel(posInd))+1);

            patterns = [patterns; patterns(curPInd)];
            labels = [labels; labels(curPInd)];
        end
    else
        warning('not yet impl.');
        keyboard;
    end
end

%% train
if i_params.debug.verbose >= 1
    fprintf('- training...\n');
end
if i_params.general.mdlType == 1
    % train a linear SVM
%     objMdl = trainSVMLight(i_params, patterns, labels, objMdl);
%     objMdl = trainSSVM_NPM(i_params, patterns, labels, objMdl);
%     objMdl = trainLinSVM(i_params, patterns, labels, objMdl);
    objMdl = trainLibSVM(i_params, patterns, labels, objMdl);
else
    % train a SSVM
    objMdl = trainSSVM(i_params, patterns, labels, objMdl);
end


%% return
o_model = objMdl;
o_pasDB = oriPasDB;

end

function [o_mdl] = squeezeMdl(i_mdl)
o_mdl = [];
if isfield(i_mdl, 'c')
    o_mdl.c = double(i_mdl.c);
else
    o_mdl.c = 0;
end
if isfield(i_mdl, 'uv')
    o_mdl.uv = i_mdl.uv;
else
    o_mdl.uv = zeros(2, 1);
end
if isfield(i_mdl, 'uv_cc')
    o_mdl.uv_cc = i_mdl.uv_cc;
else
    o_mdl.uv_cc = zeros(2, 1);
end
if isfield(i_mdl, 'wh')
    o_mdl.wh = i_mdl.wh;
else
    o_mdl.wh = zeros(2, 1);
end
if isfield(i_mdl, 'wh_cc')
    o_mdl.wh_cc = i_mdl.wh_cc;
else
    o_mdl.wh_cc = 0;
end
if isfield(i_mdl, 'ds')
    o_mdl.ds = i_mdl.ds;
else
    o_mdl.ds = 0;
end
if isfield(i_mdl, 's')
    o_mdl.s = i_mdl.s;
else
    o_mdl.s = 0;
end
if isfield(i_mdl, 'dudv')
    o_mdl.dudv = i_mdl.dudv;
else
    o_mdl.dudv = zeros(2, 1);
end
if isfield(i_mdl, 'dudv_cc')
    o_mdl.dudv_cc = i_mdl.dudv_cc;
else
    o_mdl.dudv_cc = zeros(2, 1);
end
if isfield(i_mdl, 'appFeatDim')
    o_mdl.appFeatDim = i_mdl.appFeatDim;
else
    o_mdl.appFeatDim = 0;
end
if isfield(i_mdl, 'defFeatDim')
    o_mdl.defFeatDim = i_mdl.defFeatDim;
else
    o_mdl.defFeatDim = 0;
end
if isfield(i_mdl, 'appScore')
    o_mdl.appScore = i_mdl.appScore;
else
    o_mdl.appScore = 0;
end
if isfield(i_mdl, 'defScore')
    o_mdl.defScore = i_mdl.defScore;
else
    o_mdl.defScore = 0;
end
if isfield(i_mdl, 'parts')
    o_mdl.parts = [];
    for pInd=1:numel(i_mdl.parts)
        o_mdl.parts = [o_mdl.parts; squeezeMdl(i_mdl.parts(pInd))];
    end
else
    o_mdl.parts = [];
end
if isfield(i_mdl, 'map_IDTI')
    o_mdl.map_IDTI = i_mdl.map_IDTI;
else
    o_mdl.map_IDTI = [];
end
end


function [o_nTotBB, o_dbobjIndMap] = getTotalNumBB(i_pasDB)
nTotBBs = zeros(numel(i_pasDB), 1);

parfor dbInd=1:numel(i_pasDB)
    nTotBBs(dbInd) = numel(i_pasDB(dbInd).objects);
end
o_nTotBB = sum(nTotBBs);

o_dbobjIndMap = zeros(2, o_nTotBB);
bbInd = 1;
for dbInd=1:numel(i_pasDB)
    objs = i_pasDB(dbInd).objects;
    for oInd=1:numel(objs)
        o_dbobjIndMap(:, bbInd) = [dbInd; oInd];
        bbInd = bbInd + 1;
    end
end
end

function [o_pasDB, o_objMdl] = getFeatDim(i_params, i_pasDB, i_objMdl)

o_objMdl = i_objMdl;
validObjInst = [];
for dbInd=1:numel(i_pasDB)
    curPasRec = i_pasDB(dbInd);
    poInd = find(strcmp(i_objMdl.class, {curPasRec.objects(:).class}));
    if numel(poInd) > 0
        oInd = poInd(1);
        
        [alignedBbImg, curLabel] = getAlignedBBImg(...
            getPascalImg(i_params, curPasRec), ...
            curPasRec.objects(oInd), ...
            o_objMdl.wh, ...
            curPasRec.objects(oInd).scale_psi, curPasRec.objects(oInd).resizedBB_psi);
        
%         [feats, scales] = featpyramid(alignedBbImg, i_params.feat.HoG.SqCellSize, 1, @(img) getHOGFeat(img, i_params.feat.HoG.SqCellSize, i_params.feat.HOG.type));
%         featPyr = [];
%         for sInd=1:numel(scales)
%             curFeat = [];
%             curFeat.scale = scales(sInd);
%             curFeat.feat = feats{sInd};
%             featPyr = [featPyr; curFeat];
%         end

%         featPyr = getFeatPyr(alignedBbImg, [1, i_params.feat.partResRatio], @(img) getHOGFeat(img, i_params.feat.HoG.SqCellSize, i_params.feat.HOG.type));

        [tmpFeat, validObjInst] = getFeat(i_params, ...
            struct('img', alignedBbImg, 'featPyr', []), ...
            curLabel, ...
            [1; 1; 1; 1]);
        
        o_objMdl.featDim = numel(tmpFeat(:));

        break;
    end
end
assert(~isempty(validObjInst));

% update feat dim of mdl
% o_objMdl.uv_cc = validObjInst.uv_cc;
o_objMdl.wh_cc = validObjInst.wh_cc;
o_objMdl.appFeatDim = validObjInst.appFeatDim;
o_objMdl.w_app = zeros(o_objMdl.appFeatDim, 1);
o_objMdl.w_b = 0;

parts = o_objMdl.parts;
newParts = [];
for pInd=1:numel(parts)
    newPart = parts(pInd);
    corrPart = validObjInst.parts(strcmp(newPart.class, {validObjInst.parts(:).class}));
%     newPart.uv_cc = corrPart.uv_cc;
    newPart.wh_cc = corrPart.wh_cc;
    newPart.dudv_cc = corrPart.dudv_cc;
    
    newPart.appFeatDim = corrPart.appFeatDim;
    newPart.w_app = zeros(newPart.appFeatDim, 1);
    newPart.defFeatDim = corrPart.defFeatDim;
    newPart.w_def = zeros(newPart.defFeatDim, 1);
    newPart.w_b = 0;
    
    newParts = [newParts; newPart];
end
o_objMdl.parts = newParts;

% update db
o_pasDB = num2cell(i_pasDB);
parfor dbInd=1:numel(o_pasDB)
    objs = o_pasDB{dbInd}.objects;
    newObjs = [];
    for oInd=1:numel(objs)
        
        newObj = objs(oInd);
%         newObj.uv_cc = o_objMdl.uv_cc;
        newObj.wh_cc = o_objMdl.wh_cc;
        newObj.dudv_cc = o_objMdl.dudv_cc;
        
        newObj.appFeatDim = o_objMdl.appFeatDim;
        newObj.w_app = zeros(newObj.appFeatDim, 1);
        newObj.w_b = 0;
        newObj.map_IDTI = o_objMdl.map_IDTI;
        % parts
        parts = objs(oInd).parts;
        newParts = [];
        for pInd=1:numel(parts)
            part = parts(pInd);
            partMdl = o_objMdl.parts(strcmp(part.class, {o_objMdl.parts(:).class}));
            
%             part.uv_cc = partMdl.uv_cc;
            part.wh_cc = partMdl.wh_cc;
            part.dudv_cc = partMdl.dudv_cc;
            
            part.appFeatDim = partMdl.appFeatDim;
            part.w_app = zeros(part.appFeatDim, 1);
            part.defFeatDim = partMdl.defFeatDim;
            part.w_def = zeros(part.defFeatDim, 1);
            part.w_b = 0;
            
            newParts = [newParts; part];
        end
        newObj.parts = newParts;
        newObjs = [newObjs; newObj];
    end
    o_pasDB{dbInd}.objects = newObjs;
end
o_pasDB = cell2mat(o_pasDB);

end


function [o_pasDB, o_objMdl] = transObjsToMdlSpace(i_params, i_pasDB, i_objMdl)
% function [o_bbScales, o_paddedBBs, o_objMdl] = alignBBImgByCenter(i_params, i_pasDB, i_objCls, i_objMdl)
% align BB images by their centers
% add c, u, v, w, h, ds, du, dv to objects
nTrDB = numel(i_pasDB);
objCls = i_objMdl.class;
o_objMdl = i_objMdl;
%% find max scale bbs
if i_params.feat.HoG.maxMdlImgArea == 0
    areas_pos = zeros(nTrDB, 1);
    parfor dbInd=1:nTrDB  
        curPasRec = i_pasDB(dbInd);
        posBBInds = find(strcmp(objCls, {curPasRec.objects(:).class}));

        % positive
        curMaxArea_pos = -inf;
        for oInd=posBBInds(:)'
            curBB = curPasRec.objects(oInd).bndbox;
            curW = curBB.xmax-curBB.xmin;
            curH = curBB.ymax-curBB.ymin;
            curBBArea = curW*curH;

            curMaxArea_pos = max(curMaxArea_pos, curBBArea);
        end
        areas_pos(dbInd) = curMaxArea_pos;
    end
    maxArea = max(areas_pos);
else
    maxArea = i_params.feat.HoG.maxMdlImgArea;
end

%% estimate the size of model image

% % find max wh separately
% imgWHs_pos = zeros(nTrDB, 2);
% parfor dbInd=1:nTrDB
%     curPasRec = i_pasDB(dbInd);
%     posBBInds = find(strcmp(objCls, {curPasRec.objects(:).class}));
%     
%     % positives
%     maxW = -inf;
%     maxH = -inf;
%     for oInd=posBBInds(:)'
%         curBB = curPasRec.objects(oInd).bndbox;
%         curW = curBB.xmax-curBB.xmin;
%         curH = curBB.ymax-curBB.ymin;
%         curScale = sqrt(maxArea/(curW*curH));
%         
%         curW_resc = curW*curScale;
%         curH_resc = curH*curScale;
%         
%         maxW = max(maxW, curW_resc);
%         maxH = max(maxH, curH_resc);
%     end
%     imgWHs_pos(dbInd, :) = [maxW, maxH];
%    
% end
% modelImgWH_pos = ceil(max(imgWHs_pos, [], 1));
% o_objMdl.wh = modelImgWH_pos(:);


if all(i_params.feat.HoG.mdlWH == 0)
    imgWHSums_pos = zeros(nTrDB, 2);
    nObjPerImg = zeros(nTrDB, 1);
    parfor dbInd=1:nTrDB
        curPasRec = i_pasDB(dbInd);
        posBBInds = find(strcmp(objCls, {curPasRec.objects(:).class}));

        % positives
        curWHSums = [0 0];
        for oInd=posBBInds(:)'
            curBB = curPasRec.objects(oInd).bndbox;
            curW = curBB.xmax-curBB.xmin;
            curH = curBB.ymax-curBB.ymin;
            curScale = sqrt(maxArea/(curW*curH));

            curW_resc = curW*curScale;
            curH_resc = curH*curScale;
            curWH = [curW_resc, curH_resc];

            curWHSums = curWHSums + curWH;
        end
        imgWHSums_pos(dbInd, :) = curWHSums;
        nObjPerImg(dbInd) = numel(posBBInds);

    end
    modelImgWH_pos = ceil(sum(imgWHSums_pos, 1)/sum(nObjPerImg));
    
else
    modelImgWH_pos = ceil(i_params.feat.HoG.mdlWH);
end
o_objMdl.wh = modelImgWH_pos(:);

if i_params.debug.verbose >= 1
    fprintf('- model WH: (%d, %d)\n', o_objMdl.wh(1), o_objMdl.wh(2));
end

%% find the proper scales and resize bbs to the pivot image
o_pasDB = i_pasDB;
parfor dbInd=1:nTrDB 
    for oInd=1:numel(o_pasDB(dbInd).objects)        
        curBB = o_pasDB(dbInd).objects(oInd).bndbox;
        curW = curBB.xmax-curBB.xmin;
        curH = curBB.ymax-curBB.ymin;
        
        % scale
        curScale = sqrt(maxArea/(curW*curH));
        o_pasDB(dbInd).objects(oInd).scale_psi = curScale;
       
        % resized bb        
        scaledBB = curBB;
        scaledBB.xmin = scaledBB.xmin*curScale;
        scaledBB.ymin = scaledBB.ymin*curScale;
        scaledBB.xmax = scaledBB.xmax*curScale;
        scaledBB.ymax = scaledBB.ymax*curScale;
        xPadLen = modelImgWH_pos(1) - (scaledBB.xmax-scaledBB.xmin);
        yPadLen = modelImgWH_pos(2) - (scaledBB.ymax-scaledBB.ymin);       
        
        resizedBB = scaledBB;
        resizedBB.xmin = round((resizedBB.xmin - xPadLen/2)/curScale);
        resizedBB.xmax = round((resizedBB.xmax + xPadLen/2)/curScale);
        resizedBB.ymin = round((resizedBB.ymin - yPadLen/2)/curScale);
        resizedBB.ymax = round((resizedBB.ymax + yPadLen/2)/curScale);
        
        o_pasDB(dbInd).objects(oInd).resizedBB_psi = resizedBB;
        
        % c
        o_pasDB(dbInd).objects(oInd).c = strcmp(objCls, o_pasDB(dbInd).objects(oInd).class);
        
        % uv
        o_pasDB(dbInd).objects(oInd).uv = [1; 1];
        
        % wh
        o_pasDB(dbInd).objects(oInd).wh = modelImgWH_pos(:);
        
        % ds
        o_pasDB(dbInd).objects(oInd).ds = 1;
        
        % dudv
        o_pasDB(dbInd).objects(oInd).dudv = [0; 0];
        
        % parts
        o_pasDB(dbInd).objects(oInd).parts = [];
    end
end

%% visualize
if i_params.debug.verbose >= 2
    aveImg_pos = getAverageImg(i_params, o_pasDB, objCls, o_objMdl);
    
    sfigure(10001); clf;
    imshow(aveImg_pos);
    axis on;
    axis image;
    if ~isempty(i_params.results.intResDir)
        saveas(10001, [i_params.results.intResDir '/avePosImg'], 'png');
    end
    pause(0.5);
end


end

function [o_aveImg] = getAverageImg(i_params, i_pasDB, i_objCls, i_objMdl)

% nPoss = zeros(numel(i_pasDB), 1);
% parfor dbInd=1:numel(i_pasDB)
%     nPoss(dbInd) = sum(strcmp(i_objCls, {i_pasDB(dbInd).objects(:).class}));
% end
% nPos = sum(nPoss);

aveImg_pos = [];
nAdded = 0;
for dbInd=1:numel(i_pasDB)
    curPasRec = i_pasDB(dbInd);
    img = getPascalImg(i_params, curPasRec);

    for oInd=1:numel(curPasRec.objects)
        obj = curPasRec.objects(oInd);
        if strcmp(i_objCls, curPasRec.objects(oInd).class)
            % positive
            alignedBbImg = getAlignedBBImg(img, curPasRec.objects(oInd), i_objMdl.wh, obj.scale_psi, obj.resizedBB_psi);

            if isempty(aveImg_pos)
                aveImg_pos = alignedBbImg;
            else
                aveImg_pos = imadd(aveImg_pos, alignedBbImg);
            end
            nAdded = nAdded + 1;
        end
    end
end

o_aveImg = aveImg_pos./nAdded;
    
end

function [o_alImg, o_pascalObj] = getAlignedBBImg(i_img, i_pascalObj, i_wh, i_scale, i_paddedBB)
% function [o_alImg, o_obj] = getAlignedBBImg(i_img, i_obj, i_objMdl, i_scale, i_paddedBB)
pivotWHSize = i_wh;
paddedImg = i_img;
paddedBB = i_paddedBB;
o_pascalObj = i_pascalObj;

if paddedBB.xmin < 0
    padSize = abs(paddedBB.xmin);
    paddedImg = padarray(paddedImg, [0, padSize], 0, 'pre');
    paddedBB.xmin = paddedBB.xmin + padSize;
    paddedBB.xmax = paddedBB.xmax + padSize;
end
if paddedBB.ymin < 0
    padSize = abs(paddedBB.ymin);
    paddedImg = padarray(paddedImg, [padSize, 0], 0, 'pre');
    paddedBB.ymin = paddedBB.ymin + padSize;
    paddedBB.ymax = paddedBB.ymax + padSize;
end
if paddedBB.xmax > size(i_img, 2)
    padSize = paddedBB.xmax-size(i_img, 2);
    paddedImg = padarray(paddedImg, [0, padSize], 0, 'post');
end
if paddedBB.ymax > size(i_img, 1)
    padSize = paddedBB.ymax-size(i_img, 1);
    paddedImg = padarray(paddedImg, [padSize, 0], 0, 'post');
end

% transform the image
scale = i_scale;
bbImg = imcrop(paddedImg, [paddedBB.xmin paddedBB.ymin paddedBB.xmax-paddedBB.xmin paddedBB.ymax-paddedBB.ymin]);
bbImg_scaled = imresize(bbImg, scale);
o_alImg = im2double(imresize(bbImg_scaled, [pivotWHSize(2), pivotWHSize(1)]));

% % transformlocational info.
% x0y0 = [paddedBB.xmin; paddedBB.ymin];
% o_obj.bndbox.xmin = (o_obj.bndbox.xmin - x0y0(1))*scale;
% o_obj.bndbox.ymin = (o_obj.bndbox.ymin - x0y0(2))*scale;
% o_obj.bndbox.xmax = (o_obj.bndbox.xmax - x0y0(1))*scale;
% o_obj.bndbox.ymax = (o_obj.bndbox.ymax - x0y0(2))*scale;
% 
% for cInd=1:numel(o_obj.parts)
%     children = o_obj.parts(cInd);
% 
%     children.bndbox.xmin = (children.bndbox.xmin - x0y0(1))*scale;
%     children.bndbox.ymin = (children.bndbox.ymin - x0y0(2))*scale;
%     children.bndbox.xmax = (children.bndbox.xmax - x0y0(1))*scale;
%     children.bndbox.ymax = (children.bndbox.ymax - x0y0(2))*scale;
% 
%     o_obj.parts(cInd) = children;
% end

end


function [o_newObj] = invTransferPartLabels(i_params, i_poly, i_toObj, bbScale, paddedBB)
o_newObj = i_toObj;
nParts = numel(i_poly);
o_newObj.parts = [];
for pInd=1:nParts
    curPoly = i_poly{pInd};
    transPoly = curPoly/bbScale;
    transPoly = bsxfun(@plus, transPoly, [paddedBB.xmin; paddedBB.ymin]);
    
    part = [];
    part.class = sprintf('part_%d', pInd);
    part.bndbox.xmin = min(transPoly(1, :));
    part.bndbox.xmax = max(transPoly(1, :));
    part.bndbox.ymin = min(transPoly(2, :));
    part.bndbox.ymax = max(transPoly(2, :));
    part.parts = [];
    
    o_newObj.parts = [o_newObj.parts; part];
end
end


function [o_pasDB, o_objMdl] = getPartAnnotations(i_params, i_pasDB, i_objMdl)
o_pasDB = i_pasDB;
o_objMdl = i_objMdl;

if i_params.training.activelearning == 0    
    return;
end

objCls = i_objMdl.class;
switch i_params.training.activelearning
    case 1
        % get labels
        aveImg = getAverageImg(i_params, i_pasDB, objCls, i_objMdl);
        figure(10005);
        imshow(aveImg);
        axis on; axis image;

        polys = [];
        while 1
            [x, y] = ginput(4);
            if isempty(x)
                break;
            end

            hold on;
            rectangle('Position', [min(x) min(y) max(x)-min(x) max(y)-min(y)], 'EdgeColor', 'g')
            hold off;

            polys = [polys; {ceil([x'; y'])}];
        end
%         warning('load tmp data');
%         load('annoPoly.mat');

    case 2 % choose parts for rectangular objects, like window
        aveImg = getAverageImg(i_params, i_pasDB, objCls, i_objMdl);
        figure(10005);
        imshow(aveImg);
        axis on; axis image;
        
        nPart = 4;
        wSize = size(aveImg, 2);
        hSize = size(aveImg, 1);
        partRatio = 0.30;
        sqCellSize = i_params.feat.HoG.SqCellSize;
        partTmplt = [...
            1 1 wSize*partRatio wSize*partRatio;
            1 hSize*partRatio hSize*partRatio 1
            ];
        
        ax = sqCellSize/2+1:(wSize-wSize*partRatio-sqCellSize)/(sqrt(nPart)-1):wSize; ax = ax(1:round(sqrt(nPart)));
        ay = sqCellSize/2+1:(hSize-hSize*partRatio-sqCellSize)/(sqrt(nPart)-1):hSize; ay = ay(1:round(sqrt(nPart)));
        [ax, ay] = meshgrid(ax, ay);
        anchors = [ax(:)'; ay(:)'];
        
        polys = [];
        for pInd=1:nPart
            x = min(wSize, max(1, partTmplt(1, :) + anchors(1, pInd) - 1));
            y = min(hSize, max(1, partTmplt(2, :) + anchors(2, pInd) - 1));
            
            polys = [polys; {ceil([x(:)'; y(:)'])}];
            
            % visualize
            hold on;
            rectangle('Position', [min(x) min(y) max(x)-min(x) max(y)-min(y)], 'EdgeColor', 'g')
            hold off;
            pause(0.1);
        end
end

% add parts to the model
o_objMdl.parts = [];
o_objMdl.c = 1;
for pInd=1:numel(polys)
    curPoly = ceil(polys{pInd});
    xmin = min(curPoly(1, :));
    ymin = min(curPoly(2, :));
    xmax = max(curPoly(1, :));
    ymax = max(curPoly(2, :));
    xmean = mean(curPoly(1, :));
    ymean = mean(curPoly(2, :));
    
    children = initPart(...
        sprintf('part_%d', pInd), [], [], [], ...
        1, ...
        [xmin; ymin], [], ...
        [xmax-xmin; ymax-ymin], [], ...
        2, [], ...
        [xmean; ymean], [], [], [], []);
%     children = [];
%     children.c = 1;
%     children.uv = [xmin; ymin];
%     children.wh = [xmax-xmin; ymax-ymin];
%     children.ds = 2;
%     children.dudv = [xmean; ymean];
%     children.class =  sprintf('part_%d', pInd);
%     children.parts = [];
    
    o_objMdl.parts = [o_objMdl.parts; children];
end

% inversely trasnfer part labels and add to the objects structure
parfor dbInd=1:numel(o_pasDB)
    % obj
    objs = o_pasDB(dbInd).objects;
    posObjInd = find(strcmp(objCls, {objs(:).class}));
    negObjInd = find(~strcmp(objCls, {objs(:).class}));
    
    newObjs = [];
    for poInd=posObjInd(:)'
        newObj = invTransferPartLabels(i_params, polys, objs(poInd), objs(poInd).scale_psi, objs(poInd).resizedBB_psi);
        newObjs = [newObjs; newObj];
    end
    
    newObjs2 = [];
    for noInd=negObjInd(:)'
        newObj = objs(noInd);
        newObj.parts = [];
        for pInd=1:numel(o_objMdl.parts)
            part = initPart(o_objMdl.parts(pInd).class, o_objMdl.parts(pInd).bndbox, 0, 0, 0, [0; 0], [0; 0], [0; 0], [0; 0], o_objMdl.parts(pInd).ds, 0, [0; 0], [], [], [], []);
%             part = [];
%             part.c = 0;
%             part.class = o_objMdl.parts(pInd).class;
%             part.ds = o_objMdl.parts(pInd).ds;
%             part.parts = [];
            
            newObj.parts = [newObj.parts; part];
        end
        newObjs2 = [newObjs2; newObj];
    end
    
    o_pasDB(dbInd).objects = [newObjs; newObjs2];
    
%     sfigure(10006);
%     showLabels(i_params, o_pasDB(dbInd));
end

% add part labels
parfor dbInd=1:numel(o_pasDB)
    posObjInd = find(strcmp(objCls, {o_pasDB(dbInd).objects(:).class}));
    for oInd=posObjInd(:)'
        
        newParts = [];
        for pInd=1:numel(o_pasDB(dbInd).objects(oInd).parts)
            objMdlPartsInd = find(strcmp(o_pasDB(dbInd).objects(oInd).parts(pInd).class, {o_objMdl.parts(:).class}));
            
            newPart = ...
                initPart(...
                o_pasDB(dbInd).objects(oInd).parts(pInd).class,...
                o_pasDB(dbInd).objects(oInd).parts(pInd).bndbox, ...
                [], ...
                [], ...
                1, ...
                o_objMdl.parts(objMdlPartsInd).uv, ...
                [], ...
                o_objMdl.parts(objMdlPartsInd).wh, ...
                [], ...
                o_objMdl.parts(objMdlPartsInd).ds, ...
                [], ...
                o_objMdl.parts(objMdlPartsInd).dudv, ...
                [], ...
                [], ...
                [], ...
                []);
            newParts = [newParts; newPart];
            
%             o_pasDB(dbInd).objects(oInd).parts(pInd).c = 1;
%             o_pasDB(dbInd).objects(oInd).parts(pInd).uv = o_objMdl.parts(objMdlPartsInd).uv;
%             o_pasDB(dbInd).objects(oInd).parts(pInd).wh = o_objMdl.parts(objMdlPartsInd).wh;
%             o_pasDB(dbInd).objects(oInd).parts(pInd).ds = o_objMdl.parts(objMdlPartsInd).ds;
%             o_pasDB(dbInd).objects(oInd).parts(pInd).dudv = o_objMdl.parts(objMdlPartsInd).dudv;
        end
        o_pasDB(dbInd).objects(oInd).parts = newParts;
    end
end

end

function [o_pasDB, o_objMdl] = initParts(i_pasDB, i_objCls)

%% init model
o_objMdl = initPart(i_objCls, [], 0, 0, 0, [0; 0], [1; 1], [0; 0], [0; 0], 1, 1, [0; 0], [], [], [], []);

%% init db
o_pasDB = i_pasDB;
parfor dbInd=1:numel(o_pasDB)
    newObjs = [];
    for oInd=1:numel(o_pasDB(dbInd).objects)
        newObj = initPart(o_pasDB(dbInd).objects(oInd).class, o_pasDB(dbInd).objects(oInd).bndbox, 0, 0, 0, [0; 0], [1; 1], [0; 0], [0; 0], 0, 0, [0; 0], [], [], [], []);
        newObjs = [newObjs; newObj];
    end
    o_pasDB(dbInd).objects = newObjs;
end

end

function [o_part] = initPart(class, bndbox, appFeatDim, defFeatDim, c, uv, uv_cc, wh, wh_cc, ds, s, dudv, dudv_cc, appScore, defScore, parts)

% part.class
% part.bndbox
% part.appFeatDim
% part.defFeatDim
% part.c
% part.uv
% part.uv_cc
% part.wh
% part.wh_cc
% part.ds
% part.s
% part.dudv
% part.dudv_cc
% part.appScore
% part.defScore
% part.parts

emptyBndbox = [];
emptyBndbox.xmin = 0;
emptyBndbox.xmax = 0;
emptyBndbox.ymin = 0;
emptyBndbox.ymax = 0;

o_part = [];
o_part.class = class;
if isempty(bndbox)
    o_part.bndbox = emptyBndbox;
else
    o_part.bndbox = bndbox;
end

if isempty(appFeatDim)
    o_part.appFeatDim = 0;
else
    o_part.appFeatDim = appFeatDim;
end

if isempty(defFeatDim)
    o_part.defFeatDim = 0;
else
    o_part.defFeatDim = appFeatDim;
end

if isempty(c)
    o_part.c = 0;
else
    o_part.c = c;
end

if isempty(uv)
    o_part.uv = [0; 0];
else
    o_part.uv = uv;
end
if isempty(uv_cc)
    o_part.uv_cc = [1; 1];
else
    o_part.uv_cc = uv_cc;
end

if isempty(wh)
    o_part.wh = [0; 0];
else
    o_part.wh = wh;
end

if isempty(wh_cc)
    o_part.wh_cc = [0; 0];
else
    o_part.wh_cc = wh_cc;
end

if isempty(ds)
    o_part.ds = 0;
else
    o_part.ds = ds;
end

if isempty(s)
    o_part.s = 0;
else
    o_part.s = s;
end

if isempty(dudv)
    o_part.dudv = [0; 0];
else
    o_part.dudv = dudv;
end
if isempty(dudv_cc)
    o_part.dudv_cc = [0; 0];
else
    o_part.dudv_cc = dudv_cc;
end

if isempty(appScore)
    o_part.appScore = 0;
else
    o_part.appScore = appScore;
end
if isempty(defScore)
    o_part.defScore = 0;
else
    o_part.defScore = defScore;
end

o_part.parts = parts;

end


function [o_pasRec] = chooseHardNegs(i_pasRec, i_nNegPerImg)
o_pasRec = i_pasRec;
parfor dbInd=1:numel(o_pasRec)
    scores = [o_pasRec(dbInd).objects(:).score];
    [~, Ind] = sort(scores, 'descend');
    minNegPerImg = min(i_nNegPerImg, numel(o_pasRec(dbInd).objects));
    o_pasRec(dbInd).objects = o_pasRec(dbInd).objects(Ind(1:minNegPerImg));
    
    % take bbs with bg
    for oInd=1:numel(o_pasRec(dbInd).objects)
        o_pasRec(dbInd).objects(oInd).bbox = o_pasRec(dbInd).objects(oInd).bbox_wbg;
        o_pasRec(dbInd).objects(oInd).bndbox = o_pasRec(dbInd).objects(oInd).bndbox_wbg;
    end
end
end

function [o_objMdl] = convObjMdl_tmp(i_objMdl)

parts = [];
curParts_tree = {i_objMdl};
while ~isempty(curParts_tree)
    part = [];
    
    curPart_tree = curParts_tree{1};
    part.c = double(curPart_tree.c);
    part.uv = curPart_tree.uv;
    part.uv_cc = curPart_tree.uv_cc;
    part.wh = curPart_tree.wh;
    part.wh_cc = curPart_tree.wh_cc;
    part.ds = curPart_tree.ds;
    part.s = curPart_tree.s;
    part.dudv = curPart_tree.dudv;
    part.dudv_cc = curPart_tree.dudv_cc;
    part.appFeatDim = curPart_tree.appFeatDim;
    part.defFeatDim = curPart_tree.defFeatDim;
    part.appScore = curPart_tree.appScore;
    part.defScore = curPart_tree.defScore;
    part.w_app = curPart_tree.w_app;
    if isfield(curPart_tree, 'w_def')
        part.w_def = curPart_tree.w_def;
    else
        part.w_def = zeros(4, 1);
    end
    if isfield(curPart_tree, 'w_b')
        part.w_b = curPart_tree.w_b;
    else
        part.w_b = 0;
    end
    
    parts = [parts; part];
    
    curParts_tree(1) = [];
    if ~isempty(curPart_tree.parts)
        curParts_tree = [curParts_tree; mat2cell(curPart_tree.parts, ones(numel(curPart_tree.parts), 1), 1)];
    end
end
parents = ones(numel(parts), 1);
parents(1) = 0;
%% return
o_objMdl = [];
% o_objMdl.featDim = i_objMdl.featDim;
o_objMdl.parts = parts;
o_objMdl.parents = parents;
end


function [o_mdl] = genMap_IDTI(i_mdl)

nNode = 1 + numel(i_mdl.parts); % assume a shallow model
nLevel = 2;

map_IDTI = zeros(nLevel, nNode);
map_IDTI(1, 2:end) = 1:numel(i_mdl.parts);

%% return
o_mdl = i_mdl;
o_mdl.map_IDTI = map_IDTI;

end

