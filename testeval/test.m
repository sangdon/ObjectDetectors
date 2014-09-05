function [o_pasDB_gt, o_pasDB_det] = test(i_params, i_objMdl, i_pasDB_test)

%% find cahched results
cacheFN = sprintf('%s/test.mat', i_params.results.cachingDir);
% warning('not load caches');
if i_params.general.enableCaching && exist(cacheFN, 'file') && i_params.training.hardNegMining == 0
    load(cacheFN);
    return;
end

%% load db
if isempty(i_pasDB_test)
    pasDB_gt = loadPascalDB(i_params, i_params.db.testSet);
else
    pasDB_gt = i_pasDB_test;
end

% warning('small test');
% pasDB_gt = pasDB_gt(1:10);

nTeDB = numel(pasDB_gt);
pasDB_det = pasDB_gt;

if i_params.debug.verbose >= 1
    testTID = tic;
end

maxScores = ones(nTeDB, 1)*(-inf);
minScores = ones(nTeDB, 1)*(inf);
topBbs_cell = cell(nTeDB, 1);

% showInterval = 60;
if i_params.debug.verbose >= 1
%     showTic = tic;
    showInterval = round(nTeDB*0.1);
else
%     showTic = 0;
end
% show = false;

fprintf('- test %d images\n', nTeDB);

warning('no handling on resizing part labels caused by paddings');

% parfor dbInd=1:nTeDB
for dbInd=1:nTeDB
    
    curPasRec = pasDB_gt(dbInd);
    img = getPascalImg(i_params, curPasRec);  
    [~, imgIDStr, ~] = fileparts(getPascalImgFFN( i_params, curPasRec ));
    if i_params.debug.verbose >= 1 && mod(dbInd-1, showInterval) == 0
%     if i_params.debug.verbose >= 1 && toc(showTic)>showInterval
%         show = true;
        testTicID = tic;
        fprintf('- test: %d/%d...', dbInd, nTeDB);
    end
    
    % detect
    if i_params.test.searchType == 1
        if i_params.general.mdlType == 2
            [bbs, bbs_wbg] = detect(i_objMdl, img);
        else
            [bbs, bbs_wbg] = detect_PM(i_objMdl, img);
        end
    else      
        [bbs, bbs_wbg] = detect_SS_py(i_params, img, i_objMdl, imgIDStr);
    end
    
%     if i_params.debug.verbose >= 1 && show
    if i_params.debug.verbose >= 1 && mod(dbInd-1, showInterval) == 0
        fprintf('%s sec.\n', num2str(toc(testTicID)));
%         showTic = tic;
%         show = false;
    end
    
    if isempty(bbs)
        warning('empty bbs! dbInd: %d/ is the WH too small? WH: (%d, %d)', dbInd, size(img, 2), size(img, 1));
    end


    %% for visualization and evaluation
    if ~isempty(bbs)
        maxScores(dbInd) = max(bbs(1:min(size(bbs, 1), i_params.test.topN), end));
        minScores(dbInd) = min(bbs(1:min(size(bbs, 1), i_params.test.topN), end));
    end
    
    pasDB_det(dbInd) = curPasRec;
    pasDB_det(dbInd).objects = convBB2PVObjs(bbs, i_objMdl.objMdl.class, bbs_wbg);
       
    if i_params.debug.verbose >= 2
        topBBs = bbs(1:min(size(bbs, 1), i_params.test.topN), :);
        topBbs_cell{dbInd} = topBBs;
    end
end

maxScore = max(maxScores);
minScore = min(minScores);

if i_params.debug.verbose >= 1
    fprintf('* total test time: %s sec.\n', num2str(toc(testTID)));
end

%% return
o_pasDB_gt = pasDB_gt;
o_pasDB_det = pasDB_det;

% warning('no save')
if i_params.general.enableCaching
    save(cacheFN, 'o_pasDB_gt', 'o_pasDB_det');
end

%% show results
if i_params.debug.verbose >= 2
    for dbInd=1:nTeDB
        topBBs = topBbs_cell{dbInd};
        
        curPasRec = pasDB_gt(dbInd);
        img = getPascalImg(i_params, curPasRec);
    
        sfigure(20002); clf;
%         showbbs(img, topBBs, 1, [], colormap(jet));
        showbbs(img, topBBs, 5, 0, [minScore maxScore], colormap(jet));
        if i_params.general.mdlType == 2
            sfigure(20003); clf;
            showbbs(img, topBBs, 5, 1, [minScore maxScore], colormap(jet));
        end
        
        saveas(20002, [i_params.results.detFigureDir '/' curPasRec.filename(1:end-4)], 'png');
    end
    
end

end


function [o_objs] = convBB2PVObjs(i_bbs, i_objCls, i_bbs_wbg)

objs = [];
for oInd=1:size(i_bbs, 1)
    obj = [];
    obj.class = i_objCls;
    
    obj.bbox = i_bbs(oInd, 1:4);
    obj.bndbox.xmin = obj.bbox(1);
    obj.bndbox.ymin = obj.bbox(2);
    obj.bndbox.xmax = obj.bbox(3);
    obj.bndbox.ymax = obj.bbox(4);
    
    if ~isempty(i_bbs_wbg)
        obj.bbox_wbg = i_bbs_wbg(oInd, 1:4);
        obj.bndbox_wbg.xmin = obj.bbox_wbg(1);
        obj.bndbox_wbg.ymin = obj.bbox_wbg(2);
        obj.bndbox_wbg.xmax = obj.bbox_wbg(3);
        obj.bndbox_wbg.ymax = obj.bbox_wbg(4);
    end    
    
    obj.score = i_bbs(oInd, end);
    objs = [objs; obj];
end
o_objs = objs; 
end

