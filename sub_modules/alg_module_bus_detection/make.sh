clear

rm -rf build

cmake -B build -D USE_BM=1

cd build 

make 

# ./alg_module_bus_detection_main