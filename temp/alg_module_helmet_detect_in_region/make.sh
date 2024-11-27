clear

rm -rf build

cmake -B build -D USE_BM=1

cd build 

cmake -D USE_BM=1

make

./alg_module_helmet_detect_in_region