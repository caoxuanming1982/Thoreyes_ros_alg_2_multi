cd ../build/
sudo mkdir -p /data//thoreyes/ros/alg_module_submodules/lib/
sudo cp ./lib/bm/libalg_module_detect_tracking_bm_share.so /data//thoreyes/ros/alg_module_submodules/lib/libalg_module_detect_tracking_share.so
sudo mkdir -p /data//thoreyes/ros/alg_module_submodules/base/
sudo cp ./lib/bm/libalg_module_detect_tracking_bm_share.so /data//thoreyes/ros/alg_module_submodules/base/libalg_module_detect_tracking_share.so
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_submodules/lib/
sudo cp ./lib/nv_cvcuda/libalg_module_detect_tracking_nv_cvcuda_share.so /data//thoreyes_nv/ros/alg_module_submodules/lib/libalg_module_detect_tracking_share.so
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_submodules/base/
sudo cp ./lib/nv_cvcuda/libalg_module_detect_tracking_nv_cvcuda_share.so /data//thoreyes_nv/ros/alg_module_submodules/base/libalg_module_detect_tracking_share.so
cd ../project