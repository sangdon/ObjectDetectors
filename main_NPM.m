%% user settings

% expID = 'exp_NPMwoCN'; % AP: 0.471
% expID = 'exp_NPMwCN'; % AP: 0.690
% expID = 'exp_NPM_HObC_tmp'; % AP: ? 
% expID = 'exp_HObC_N0';
expID = 'exp_Img';

resultDir = '~/UPenn/Research/Data/OD/';

% object names
% objCls = 'inriaperson';
objCls = 'obj';
% objCls = 'car_side';
% objCls = 'window';


% INRIA person
train_inria = 0;

clearCache = false;
% clearCache = true;

% parameter settings
params = [];
% general
params.general.objCls = objCls;
params.general.enableCaching = true;
params.general.mdlType = 1; % 1: NPM, 2: DPM
% libs
params.lib.libDir = {...
    '~/UPenn/Research/VisionTools/libDPM5', ...
    '~/UPenn/Research/VisionTools/libMatlabHelper', ...
    '~/UPenn/Research/VisionTools/PVOC', ...
    '~/UPenn/Research/VisionTools/libsvm'};
% db
params.db.rootDir = '~/UPenn/Dropbox/Research/SocialObject/trainImg_kids_VOC'; % pascal format
% params.db.rootDir = '/data/v50/sangdonp/objectDetection/DB/VOC_Paris'; % pascal format
% params.db.rootDir = '/data/v50/sangdonp/objectDetection/DB/VOC_INRIA_person'; % pascal format
params.db.annDir = [params.db.rootDir '/Annotations']; % pascal format
params.db.imgDir = [params.db.rootDir '/JPEGImages']; % pascal format
params.db.imagesetDir = [params.db.rootDir '/ImageSets/Main']; % pascal format
params.db.trainingSet = [params.db.imagesetDir '/train.txt']; % pascal format
params.db.testSet = [params.db.imagesetDir '/test.txt']; % pascal format
params.db.valSet = [params.db.imagesetDir '/val.txt']; % pascal format
params.db.trainvalSet = [params.db.imagesetDir '/trainval.txt']; % pascal format
% feature
params.feat.HoG.SqCellSize = 4;
% params.feat.HoG.mdlWH = [64 128]; % [0 0]; 
% params.feat.HoG.maxMdlImgArea = 64*128; % 100*50;
params.feat.HoG.mdlWH = [0 0]; 
params.feat.HoG.maxMdlImgArea = 100*50;
params.feat.HOG.type = 1; % 1: HOG_UOCTTI, 2: 1 wo CN, 3: wo AN, 4: HObC, 5: HObC_N0 ?: HOG_UOCTTI_vlfeat, 6: img
params.feat.partResRatio = 2;
params.feat.cachingDir = [resultDir expID '/' objCls '/cacheDir'];
% training
% params.training.C = 1e3; % 1e1: 0.563, 1e2: .5629, 1e3: .541(?), 1e6: .512
params.training.C = 1e12; % SVM: 1e0: .630, 1e3: .667, 1e6: .667 
params.training.tol = 0.001;
params.training.nMaxNegPerImg = 5;
params.training.hardNegMining = 0;
params.training.activelearning = 0;
params.training.reflect = 0;
params.training.exampleDuplication = 0;
params.training.padSize = 0; %%%% for INRIA, -16 or 0 with imgPadSz=16
% test
params.test.interval = 8;
% params.test.scaleSearch = 0.2:0.1:0.4;
params.test.bgContextSz = 0; % learningWH = realWH*(1+bgContextSz)
params.test.scoreThres = -1000;
params.test.topN = 50;
params.test.nms = 1;
params.test.nmsOverlap = 0.5;
params.test.searchType = 1; % 1: exhuastive, 2: selective
% evalutate
params.eval.minOverlap = 0.5;
% debugging options
params.debug.verbose = 2;
% results
params.results.cachingDir = [resultDir expID '/' objCls '/cacheDir']; 
params.results.detFigureDir = [resultDir expID '/' objCls '/detFigures']; 
params.results.intResDir = [resultDir expID '/' objCls '/intResDir']; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initialize
close all;

rng(1);

assert(~(train_inria == 1 && params.training.reflect == 1))

if clearCache
    rmdir(params.results.cachingDir, 's');
    rmdir(params.results.detFigureDir, 's');
    rmdir(params.results.intResDir, 's');
    return;
end

if matlabpool('size') == 0
    matlabpool open;
end

if params.debug.verbose < 0 || params.debug.verbose > 2
    warning('exceed the normal range of verbose');
    return;
end

if ~exist(params.results.cachingDir, 'dir')
    mkdir(params.results.cachingDir);
end
if ~exist(params.results.detFigureDir, 'dir')
    mkdir(params.results.detFigureDir);
end
if ~exist(params.results.intResDir, 'dir')
    mkdir(params.results.intResDir);
end


cellfun(@addpath, params.lib.libDir);


%% train
if train_inria == 1
    objMdl = train(params, objCls, addObjPading( loadPascalDB(params, [params.db.imagesetDir '/train_inria.txt']), objCls, params.training.padSize));
else
    objMdl = train(params, objCls, []);
end

%% test
[pasDB_gt, pasDB_det] = test(params, objMdl, []);

% %% evaluate
% stats_NPM = evaluate(params, objCls, pasDB_gt, pasDB_det);

