#include "network_engine/network_kernel.h"
#include "error_type.h"
#include <stdlib.h>

Network_kernel::Network_kernel(std::shared_ptr<Device_Handle> handle, std::string file_path, std::string model_name, std::vector<Shape_t> &input_shapes, int max_instance)
{
    handle_ = handle;
    this->file_path = file_path;
    this->model_name = model_name;
    if (input_shapes.size() > 0)
        cache_inputs_shapes = input_shapes;
    this->max_instance = max_instance;
};
Network_kernel::~Network_kernel()
{
    for (int i = 0; i < this->instance_mutex.size(); i++)
    {
        delete this->instance_mutex[i];
        this->instance_mutex[i] = nullptr;
    }
    this->instance_mutex.clear();
};
std::vector<Shape_t> Network_kernel::get_input_shapes()
{
    return cache_inputs_shapes;
};

int Network_kernel::check_need_scale_offset(int idx){
    if(idx>=this->input_scale.size()){
        return 0;
    }
    bool need_scale=false;
    if(input_scale[idx].val[0]!=1||input_scale[idx].val[1]!=1||input_scale[idx].val[2]!=1){
        need_scale=true;
    }
    bool need_offset=false;
    if(input_offset[idx].val[0]!=0||input_offset[idx].val[1]!=0||input_offset[idx].val[2]!=0){
        need_offset=true;
    }
    if(need_scale&&need_offset)
        return 3;
    if(need_scale)
        return 1;
    if(need_offset)
        return 2;

    return 0;
};
