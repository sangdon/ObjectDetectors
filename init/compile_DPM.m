
%% train
setenv('LD_LIBRARY_PATH', ['/usr/lib64:', getenv('LD_LIBRARY_PATH')]);
coder -build train.prj
!rm ./libs/libSSVM/svm-struct-matlab_140120/coder/*
!cp /data/extern/sangdonp/objectDetection/HSDPM/train/*.a /data/extern/sangdonp/objectDetection/HSDPM/train/*.h /data/extern/sangdonp/objectDetection/HSDPM/train/*.c ./libs/libSSVM/svm-struct-matlab_140120/coder/
cd ./libs/libSSVM/svm-struct-matlab_140120
!make clean
!make
cd ../../../

%% test
coder -build test.prj
!cp /data/extern/sangdonp/objectDetection/HSDPM/test/slideWindow_mex.mexa64 ./

