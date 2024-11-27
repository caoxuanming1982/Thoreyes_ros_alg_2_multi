cd ..
rm -rf build/*
mkdir build
cd build
cmake ../project
make -j10
cp ./lib/bm/alg_module_base_bm_test_cv_lib_main ./alg_module_base_bm_test_cv_lib_main
cp ./lib/nv_cvcuda/alg_module_base_nv_cvcuda_test_cv_lib_main ./alg_module_base_nv_cvcuda_test_cv_lib_main
cd ../project
