%% NPM
% coder -build ../feature/getHCFeat.prj
% !cp ./../feature/codegen/mex/getHCFeat/getHCFeat_mex.mexa64 ./../feature/getHCFeat.mexa64
% !rm ./../feature/codegen -Rf

coder -build ../feature/getHCFeat.prj
!cp /data/extern/sangdonp/objectDetection/HSDPM/test/getHCFeat_mex.mexa64 ./../feature/getHCFeat.mexa64