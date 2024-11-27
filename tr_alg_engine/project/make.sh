cd ..
rm -rf ./build/*
rm -rf ./install/*
colcon build --packages-select tr_alg_engine --merge-install  --install-base /data//thoreyes/ros/tr_alg_engine
rm -rf ./build/*
rm -rf ./install/*
colcon build --packages-select tr_alg_engine_nv_cvcuda --merge-install  --install-base /data//thoreyes_nv/ros/tr_alg_engine
cd project
