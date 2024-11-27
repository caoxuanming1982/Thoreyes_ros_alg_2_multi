cd ..
rm -rf build/*
mkdir build
cd build
cmake ../project
make -j10
cd ../project
cp ../build/test/test_alg_module_license_plate_detection_bm_main ../build/alg_module_license_plate_detection_main
