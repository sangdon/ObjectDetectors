%% user settings

expID = 'exp_DPM_tmp';

% object names
% objCls = 'inriaperson';
objCls = 'Motorbikes';
% objCls = 'car_side';
% objCls = 'window';


clearCache = false;
% clearCache = true;

% parameter settings
params = [];
% general
params.general.objCls = objCls;
params.general.enableCaching = true;
params.general.mdlType = 2; % 1: NPM, 2: DPM
% libs
params.lib.libDir = './libs';
% db
params.db.rootDir = '/data/v50/sangdonp/objectDetection/DB/VOC_Caltech101'; % pascal format
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
params.feat.HoG.SqCellSize = 8;
params.feat.HoG.mdlWH = [0 0]; 
params.feat.HoG.maxMdlImgArea = 100*50;
params.feat.HOG.type = 1; % 1: HOG_UOCTTI, 2: 1 wo CN, 3: HOG_UOCTTI_vlfeat
params.feat.partResRatio = 2;
% training
params.training.C = 1e2; % 1e2: .5629, 1e3: .541(?), 1e6: .512
params.training.tol = 0.001;
params.training.nMaxNegPerImg = 5;
params.training.hardNegMining = 0;
params.training.activelearning = 1;
params.training.reflect = 0;
params.training.padSize = 0; %%%% for INRIA
% test
params.test.interval = 8;
% params.test.scaleSearch = 0.2:0.1:0.4;
params.test.bgContextSz = 0; % in pixels of the model space
params.test.scoreThres = -1000;
params.test.topN = 10;
params.test.nms = 1;
params.test.nmsOverlap = 0.5;
% evalutate
params.eval.minOverlap = 0.5;
% debugging options
params.debug.verbose = 1;
% results
params.results.cachingDir = ['/data/v50/sangdonp/objectDetection/' expID '/' objCls '/cacheDir']; 
params.results.detFigureDir = ['/data/v50/sangdonp/objectDetection/' expID '/' objCls '/detFigures']; 
params.results.intResDir = ['/data/v50/sangdonp/objectDetection/' expID '/' objCls '/intResDir']; 

% INRIA person
train_inria = 0;

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


addpath(genpath(params.lib.libDir));

%% train
if train_inria == 1
    objMdl = train(params, objCls, addPading( loadPascalDB(params, [params.db.imagesetDir '/train_inria.txt']), objCls, params.training.padSize));
else
    objMdl = train(params, objCls, []);
end

%% test
[pasDB_gt, pasDB_det] = test(params, objMdl, []);

%% evaluate
stats_NPM = evaluate(params, objCls, pasDB_gt, pasDB_det);





% 
% %% user settings
% 
% % expID = 'exp_NPM';
% % expID = 'exp_DPM';
% expID = 'contrast';
% 
% % object names
% objCls = 'Motorbikes';
% % objCls = 'car_side';
% % objCls = 'window';
% 
% % clearCache = true;
% clearCache = false;
% 
% % parameter settings
% params = [];
% % general
% params.general.enableCaching = false;
% params.general.type = 2; % 1: NPM, 2: DPM
% % libs
% params.lib.libDir = './libs';
% % params.lib.vlfeatDir = '~/UPenn/Dropbox/Research/vlfeat/vlfeat-0.9.17';
% % params.lib.vlfeatDir = './libs/libvlfeat/vlfeat-0.9.17';
% 
% % db
% params.db.rootDir = '/data/v50/sangdonp/objectDetection/DB/VOC_Caltech101'; % pascal format
% % params.db.rootDir = '/data/v50/sangdonp/objectDetection/DB/VOC_Paris'; % pascal format
% params.db.annDir = [params.db.rootDir '/Annotations']; % pascal format
% params.db.imgDir = [params.db.rootDir '/JPEGImages']; % pascal format
% params.db.imagesetDir = [params.db.rootDir '/ImageSets/Main']; % pascal format
% params.db.trainingSet = [params.db.imagesetDir '/train.txt']; % pascal format
% params.db.testSet = [params.db.imagesetDir '/test.txt']; % pascal format
% params.db.valSet = [params.db.imagesetDir '/val.txt']; % pascal format
% params.db.trainvalSet = [params.db.imagesetDir '/trainval.txt']; % pascal format
% % feature
% params.feat.HoG.SqCellSize = 8;
% params.feat.HoG.maxMdlImgArea = 100*50;
% params.feat.partResRatio = 2;
% % training
% params.training.C = 1000;
% params.training.nMaxNegPerImg = 10;
% params.training.hardNegMining = 0;
% params.training.activelearning = 0;
% % test
% params.test.scaleSearch = 0.2:0.1:1;
% params.test.scoreThres = -1000;
% params.test.topN = 1;
% params.test.nms = 1;
% params.test.nmsOverlap = 0.5;
% % evalutate
% params.eval.minOverlap = 0.5;
% % debugging options
% params.debug.verbose = 1;
% % results
% params.results.cachingDir = ['/data/v50/sangdonp/objectDetection/' expID '/' objCls '/cacheDir']; 
% params.results.detFigureDir = ['/data/v50/sangdonp/objectDetection/' expID '/' objCls '/detFigures']; 
% params.results.intResDir = ['/data/v50/sangdonp/objectDetection/' expID '/' objCls '/intResDir']; 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% initialize
% close all;
% 
% if clearCache
%     rmdir(params.results.cachingDir, 's');
%     rmdir(params.results.detFigureDir, 's');
%     rmdir(params.results.intResDir, 's');
%     return;
% end
% 
% if matlabpool('size') == 0
%     matlabpool open;
% end
% 
% if params.debug.verbose < 0 || params.debug.verbose > 2
%     warning('exceed the normal range of verbose');
%     return;
% end
% 
% if ~exist(params.results.cachingDir, 'dir')
%     mkdir(params.results.cachingDir);
% end
% if ~exist(params.results.detFigureDir, 'dir')
%     mkdir(params.results.detFigureDir);
% end
% if ~exist(params.results.intResDir, 'dir')
%     mkdir(params.results.intResDir);
% end
% 
% 
% addpath(genpath(params.lib.libDir));
% 
% %% train
% objMdl = train(params, objCls, []);
% 
% %% test
% [pasDB_gt, pasDB_det] = test(params, objMdl, []);
% 
% %% evaluate
% stats_DPMwP = evaluate(params, objCls, pasDB_gt, pasDB_det);
