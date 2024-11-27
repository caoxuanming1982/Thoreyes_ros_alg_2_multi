#include "network_engine/bm/network_kernel_bm.h"
#include "network_engine/bm/device_handle_bm.h"
#include "cv_lib/bm/type_def_bm.h"

std::string convert_model_path(std::string model_path){
    auto model_path_t= std::filesystem::path(model_path);
    if(model_path_t.has_extension()){
        std::string ext=model_path_t.extension().string();
        if(ext!="bmodel"){
            model_path_t.replace_extension("bmodel");
        }
        model_path=model_path_t.string();
    }
    else{
        model_path=model_path_t.string()+".bmodel";

    }
    return model_path;
}


Network_kernel_bm::Network_kernel_bm(std::shared_ptr<Device_Handle> handle, std::string file_path, std::string model_name, std::vector<Shape_t> &input_shapes, int max_instance) : Network_kernel(handle, file_path, model_name, input_shapes, max_instance)
{
    std::shared_ptr<Device_Handle_bm> t_handle = std::dynamic_pointer_cast<Device_Handle_bm>(handle);
    if (t_handle == nullptr)
    {
        throw std::runtime_error(file_path + "    " + model_name + "   model kernel get invalid device handle");
    }
    this->p_ctx = new Context(t_handle->handle);

    auto ret = this->p_ctx->load_bmodel(this->file_path.c_str());
    if (ret != BM_SUCCESS)
    {
        delete this->p_ctx;
        this->p_ctx = nullptr;
        throw std::runtime_error("model file " + file_path + " not exists");
    }

    for(int i=0;i<max_instance;i++){
        this->net_instances.push_back(new Network(*(this->p_ctx),model_name.c_str()));
        this->instance_mutex.push_back(new std::shared_mutex());
    }

};

int Network_kernel_bm::forward(std::vector<std::shared_ptr<QyImage>> &inputs, std::vector<Output> &outputs)
{
    int instance_id = (this->cnt++ % this->max_instance);
    auto lock = std::unique_lock(*this->instance_mutex[instance_id]);
    auto &instance = this->net_instances[instance_id];
    auto net_info = instance->info();
    auto &model_inputs = instance->Inputs();

    std::vector<bm_device_mem_t> need_free_memory;
    std::shared_ptr<Device_Handle_bm> dst_handle=std::dynamic_pointer_cast<Device_Handle_bm>(this->handle_);
    std::vector<Shape_t> shapes=this->get_input_shapes();
    std::vector<std::shared_ptr<QyImage>> cache_input;

    for (int i = 0; i < inputs.size(); i++)
    {
        model_inputs[i]->Reshape(net_info->stages[0].input_shapes[i]);
        std::shared_ptr<QyImage> temp1=inputs[i]->convertTo(QyImage::Data_type::Float32);
        std::shared_ptr<QyImage_bm> temp = std::dynamic_pointer_cast<QyImage_bm>(temp1);
        
        switch(check_need_scale_offset(i)){
            case 1:
                temp=std::dynamic_pointer_cast<QyImage_bm>(temp->operator*(input_scale[i]));
                break;
            case 2:
                temp=std::dynamic_pointer_cast<QyImage_bm>(temp->operator+(input_offset[i]));
                break;
            case 3:
                temp=std::dynamic_pointer_cast<QyImage_bm>(temp->scale_add(input_scale[i],input_offset[i]));
                break;
            default:
                break;
        }
        temp1=temp->auto_swap_HWC(shapes[i]);
        
        temp= std::dynamic_pointer_cast<QyImage_bm>(temp1);
        if(temp==nullptr){
            throw std::runtime_error(file_path + "    " + model_name + "   model not load or backend missmatch");
        } 
        cache_input.push_back(temp1);
        bm_device_mem_t mem;
        bm_image_get_device_mem(temp->data.image, &mem);

        if(this->handle_->get_device_id()!=inputs[i]->get_handle()->get_device_id()){
            unsigned int n_bytes=bm_mem_get_device_size(mem);

            bm_device_mem_t device_mem;
            bm_malloc_device_byte(dst_handle->handle, &device_mem, n_bytes);
            std::shared_ptr<Device_Handle_bm> src_handle=std::dynamic_pointer_cast<Device_Handle_bm>(inputs[i]->get_handle());

            if(this->handle_->get_card_id()!=inputs[i]->get_handle()->get_card_id()){
                std::vector<uint8_t> temp;
                temp.resize(n_bytes);
                bm_memcpy_d2s(src_handle->handle,(void*)temp.data(),mem);
                bm_memcpy_s2d(dst_handle->handle,device_mem,(void*)temp.data());

            }
            else{
                bm_memcpy_c2c(src_handle->handle, dst_handle->handle, mem, device_mem, true);
            }
            need_free_memory.push_back(device_mem);
            model_inputs[i]->set_device_mem(device_mem);
        }
        else{
            model_inputs[i]->set_device_mem(mem);

        }

    }
    bm_status_t status = instance->Forward(false);
    for(int i=0;i<need_free_memory.size();i++){
        bm_free_device(dst_handle->handle,need_free_memory[i]);
    }
    if (status != BM_SUCCESS)
        return -1;
    int n_out = instance->info()->output_num;
    outputs.resize(n_out);
    auto &model_outputs = instance->Outputs();

    for (int i = 0; i < n_out; i++)
    {
        int n_dim = net_info->stages[0].output_shapes[i].num_dims;
        outputs[i].shape.resize(n_dim);
        for (int j = 0; j < n_dim; j++)
        {
            outputs[i].shape[j] = net_info->stages[0].output_shapes[i].dims[j];
        }
    }
    status = bm_thread_sync(this->p_ctx->handle());
    if (status != BM_SUCCESS)
        return -1;
    for (int i = 0; i < n_out; i++)
    {
        Tensor *model_output = model_outputs[i];
        outputs[i].data.resize(model_output->ByteSize());
        model_output->CopyTo(outputs[i].data.data());
    }
    return 0;
};
std::vector<Shape_t> Network_kernel_bm::get_input_shapes()
{
    if (this->net_instances.size() <= 0)
    {
        return std::vector<Shape_t>();
    }
    auto &instance = this->net_instances[0];
    auto net_info = instance->info();
    std::vector<Shape_t> res;
    for (int i = 0; i < net_info->input_num; i++)
    {
        Shape_t shape;
        shape.num_dims = net_info->stages[0].input_shapes[i].num_dims;
        for (int j = 0; j < shape.num_dims; j++)
        {
            shape.dims[j] = net_info->stages[0].input_shapes[i].dims[j];
        }
        res.push_back(shape);
    }
//    cache_inputs_shapes = res;
    return res;
};

Network_kernel* get_network_kernel(std::shared_ptr<Device_Handle> handle,std::string file_path,std::string model_name,std::vector<Shape_t>& input_shapes,int max_instance){
    Network_kernel* instance=new Network_kernel_bm(handle,file_path,model_name,input_shapes,max_instance);
    return instance;
};
void free_network_kernel(Network_kernel* instance){
    delete instance;
};

bool global_init(){
    return true;
};
