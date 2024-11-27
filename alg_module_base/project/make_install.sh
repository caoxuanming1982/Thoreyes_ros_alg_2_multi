cd ../build/
sudo mkdir -p /data//thoreyes/ros/alg_module_base/lib/
sudo cp ./lib/bm/libalg_module_base_bm_share.so /data//thoreyes/ros/alg_module_base/lib/libalg_module_base_bm_share.so
rm /data//thoreyes/ros/alg_module_base/lib/libalg_module_base_share.so
ln -s /data//thoreyes/ros/alg_module_base/lib/libalg_module_base_bm_share.so /data//thoreyes/ros/alg_module_base/lib/libalg_module_base_share.so
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/lib/
sudo cp ./lib/nv_cvcuda/libalg_module_base_nv_cvcuda_share.so /data//thoreyes_nv/ros/alg_module_base/lib/libalg_module_base_nv_cvcuda_share.so
rm /data//thoreyes_nv/ros/alg_module_base/lib/libalg_module_base_share.so
ln -s /data//thoreyes_nv/ros/alg_module_base/lib/libalg_module_base_nv_cvcuda_share.so /data//thoreyes_nv/ros/alg_module_base/lib/libalg_module_base_share.so
sudo rm -rf /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/error_type.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/error_type.h
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/alg_module_base_private.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/alg_module_base_private.h
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/inout_type.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/inout_type.h
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/network_engine /data//thoreyes/ros/alg_module_base/include/alg_module_base/network_engine
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/alg_module_base.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/alg_module_base.h
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/common.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/common.h
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/cv_lib /data//thoreyes/ros/alg_module_base/include/alg_module_base/cv_lib
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/tr_cfg_type_base.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/tr_cfg_type_base.h
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/publish_cfg_base.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/publish_cfg_base.h
sudo mkdir -p /data//thoreyes/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/post_process_cfg_base.h /data//thoreyes/ros/alg_module_base/include/alg_module_base/post_process_cfg_base.h
sudo rm -rf /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/error_type.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/error_type.h
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/alg_module_base_private.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/alg_module_base_private.h
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/inout_type.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/inout_type.h
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/network_engine /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/network_engine
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/alg_module_base.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/alg_module_base.h
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/common.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/common.h
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/cv_lib /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/cv_lib
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/tr_cfg_type_base.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/tr_cfg_type_base.h
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/publish_cfg_base.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/publish_cfg_base.h
sudo mkdir -p /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/
sudo cp -r .././include/post_process_cfg_base.h /data//thoreyes_nv/ros/alg_module_base/include/alg_module_base/post_process_cfg_base.h
sudo cp -r ../doc/ /data//thoreyes/ros/alg_module_base/doc/
sudo cp -r ../doc/ /data//thoreyes_nv/ros/alg_module_base/doc/
cd ../project
sudo rm  -rf /opt/alg_sub_module_develop/
sudo mkdir -p /opt/alg_sub_module_develop/
sudo cp -r ./sub_module_define /opt/alg_sub_module_develop//base_define
