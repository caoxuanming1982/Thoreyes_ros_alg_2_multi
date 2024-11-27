clear

rm -rf build

cmake -B build -D USE_HW=1 -DCMAKE_INSTALL_PREFIX=/data/thoreyes_hw/ros/alg_module_interfaces/

cd build

make
make install

