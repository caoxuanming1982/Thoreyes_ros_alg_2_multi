#include "network_engine/torch_pre_abi/network_kernel_torch.h"
#include "network_engine/torch_pre_abi/device_handle_torch.h"
#include "network_engine/torch_pre_abi/torch_pre_abi_interface.h"
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include "cv_lib/type_def.h"
#include "cv_lib/cv/type_def_cv.h"
#include "error_type.h"

//#define USE_CUDA
//#define USE_CVCUDA
//#define OPTIMIZE
#ifdef USE_CUDA
#include "cv_lib/cuda/type_dev_cuda.h"
#endif
#ifdef USE_CVCUDA
#include "cv_lib/cvcuda/type_def_cvcuda.h"
#endif

std::string convert_model_path(std::string model_path){
    auto model_path_t= std::filesystem::path(model_path);
    if(model_path_t.has_extension()){
        std::string ext=model_path_t.extension().string();
        if(ext!="pt"){
            model_path_t.replace_extension("pt");
        }
        model_path=model_path_t.string();
    }
    else{
        model_path=model_path_t.string()+".pt";

    }
    return model_path;
}


Network_kernel_torch::Network_kernel_torch(std::shared_ptr<Device_Handle> handle, std::string file_path_in, std::string model_name, std::vector<Shape_t> &input_shapes, int max_instance) : Network_kernel(handle, file_path_in, model_name, input_shapes, max_instance)
{
    std::shared_ptr<Device_Handle_torch> t_handle = std::dynamic_pointer_cast<Device_Handle_torch>(handle);
    if (t_handle == nullptr)
    {
        throw std::runtime_error(file_path_in + "    " + model_name + "   model kernel get invalid device handle");
    }
#ifdef OPTIMIZE
    if (torch::cuda::cudnn_is_available())
    {
        putenv("CUDNN_BENCHMARK=1");
        std::cout << "cudnn avaliable !" << std::endl;
    }

    //    torch::cuda::cudnn_is_available();
    torch::jit::getExecutorMode() = false;
#endif

    
    for (int i = 0; i < max_instance; i++)
    {
        this->net_instances.push_back(torch_pre_abi::load(file_path_in.c_str(), t_handle->handle));
#ifdef USE_FP16
        this->net_instances[i].to(c10::ScalarType::Half);
#endif

        this->net_instances[i].eval();
#ifdef OPTIMIZE
        this->net_instances[i] = torch::jit::freeze(torch::jit::optimize_for_inference(this->net_instances[i]));
#endif
        this->instance_mutex.push_back(new std::shared_mutex());

    }
    this->cache_inputs_shapes = input_shapes;
};

int Network_kernel_torch::forward(std::vector<std::shared_ptr<QyImage>> &inputs, std::vector<Output> &outputs)
{
    std::shared_ptr<Device_Handle_torch> t_handle = std::dynamic_pointer_cast<Device_Handle_torch>(handle_);
    std::vector<torch::Tensor> inputs_t;
    inputs_t.resize(inputs.size());
    bool have_data = false;
    std::vector<std::shared_ptr<QyImage>> cache_input;

    int current_device=0;
    cudaGetDevice(&current_device);
#ifdef USE_CUDA
    if (have_data == false)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            std::shared_ptr<QyImage_cv_cuda> input = std::dynamic_pointer_cast<QyImage_cv_cuda>(inputs[i]);
        switch(check_need_scale_offset(i)){
            case 1:
                input=std::dynamic_pointer_cast<QyImage_cv_cuda>(input->operator*(input_scale[i]));
                cache_input.push_back(input);
                break;
            case 2:
                input=std::dynamic_pointer_cast<QyImage_cv_cuda>(input->operator+(input_offset[i]));
                cache_input.push_back(input);
                break;
            case 3:
                input=std::dynamic_pointer_cast<QyImage_cv_cuda>(input->scale_add(input_scale[i],input_offset[i]));
                cache_input.push_back(input);
                break;
            default:
                break;
        }
            std::vector<int64_t> shape_t;
            shape_t.push_back(1);
            shape_t.push_back(input->data.image.rows);
            shape_t.push_back(input->data.image.cols);
            shape_t.push_back(input->data.image.channels());
            cv::cuda::GpuMat temp = input->data.image;
            auto opt = c10::TensorOptions();
            opt = opt.device(t_handle->handle);
            int type = temp.type() & CV_MAT_DEPTH_MASK;
            if (type == CV_32F)
            {
                opt = opt.dtype(c10::ScalarType::Float);
            }
#ifndef USE_IX            
            else if (type == CV_16F)
            {
                opt = opt.dtype(c10::ScalarType::Half);
            }
#endif            
            else if (type == CV_8U)
            {
                opt = opt.dtype(c10::ScalarType::Byte);
            }
            else if (type == CV_8S)
            {
                opt = opt.dtype(c10::ScalarType::Char);
            }
            else if (type == CV_16S)
            {
                opt = opt.dtype(c10::ScalarType::Short);
            }
#ifndef USE_IX            
            else if (type == CV_16U)
            {
                opt = opt.dtype(c10::ScalarType::UInt16);
            }
#endif            
            else if (type == CV_32S)
            {
                opt = opt.dtype(c10::ScalarType::Int);
            }
            else if (type == CV_64F)
            {
                opt = opt.dtype(c10::ScalarType::Double);
            }
            else
            {
                throw Alg_Module_Exception("Error:\t input data type unknown cv mat depth " + std::to_string(type), this->model_name, Alg_Module_Exception::Stage::inference); // 获取模型推理实例异常，一般是因为模型实例还未创建
            }

            if (t_handle->handle.is_cuda())
            {
                if (current_device == t_handle->handle.index())
                {
                    inputs_t[i] = torch::zeros(shape_t, opt);
                    cudaMemcpy2D(inputs_t[i].data_ptr(), temp.cols * temp.elemSize(), temp.data, temp.step, temp.cols * temp.elemSize(), temp.rows, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                }
                else
                {
                    opt = opt.device(torch::Device(c10::DeviceType::CUDA, current_device));
                    inputs_t[i] = torch::zeros(shape_t, opt);
                    cudaMemcpy2D(inputs_t[i].data_ptr(), temp.cols * temp.elemSize(), temp.data, temp.step, temp.cols * temp.elemSize(), temp.rows, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                    inputs_t[i] = inputs_t[i].to(t_handle->handle);
                }
            }
            else
            {
                cudaMemcpy2D(inputs_t[i].data_ptr(), temp.cols * temp.elemSize(), temp.data, temp.step, temp.cols * temp.elemSize(), temp.rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);
            }
            have_data = true;
        }
    }

#endif

#ifdef USE_CVCUDA
    if (have_data == false)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            std::shared_ptr<QyImage_cvcuda> input = std::dynamic_pointer_cast<QyImage_cvcuda>(inputs[i]);
        switch(check_need_scale_offset(i)){
            case 1:
                input=std::dynamic_pointer_cast<QyImage_cvcuda>(input->operator*(input_scale[i]));
                cache_input.push_back(input);
                break;
            case 2:
                input=std::dynamic_pointer_cast<QyImage_cvcuda>(input->operator+(input_offset[i]));
                cache_input.push_back(input);
                break;
            case 3:
                input=std::dynamic_pointer_cast<QyImage_cvcuda>(input->scale_add(input_scale[i],input_offset[i]));
                cache_input.push_back(input);
                break;
            default:
                break;
        }

            auto shape=input->data.image->shape();
            std::vector<int64_t> shape_t;
            shape_t.push_back(1);
            shape_t.push_back(shape[1]);
            shape_t.push_back(shape[2]);
            shape_t.push_back(shape[3]);

            nvcv::Tensor::Requirements inReqs=nvcv::Tensor::CalcRequirements(input->data.image->shape(),input->data.image->dtype());

#ifdef CVCUDA_OLD
    auto temp_base=input->data.image->exportData();
    auto temp=dynamic_cast<const nvcv::ITensorDataStrided*>(temp_base);
#else
    
    auto temp=input->data.image->exportData<nvcv::TensorDataStridedCuda>();
#endif
            NVCVDataType type;
            nvcvTensorGetDataType(input->data.image->handle(), &type);
            auto opt = c10::TensorOptions();
            opt = opt.device(t_handle->handle);
            int n_bit=1;
            switch(type){
                case NVCV_DATA_TYPE_U8: 
                    opt = opt.dtype(c10::ScalarType::Byte);
                    n_bit=1;
                    break;
#ifndef CVCUDA_OLD
                case NVCV_DATA_TYPE_F16: 
                    opt = opt.dtype(c10::ScalarType::Half);
                    n_bit=2;
                    break;
#endif                    
                case NVCV_DATA_TYPE_F32: 
                    opt = opt.dtype(c10::ScalarType::Float);
                    n_bit=4;
                    break;
                default:
                    input=std::dynamic_pointer_cast<QyImage_cvcuda>(input->convertTo(QyImage::Data_type::Float32));
                    n_bit=4;
                    break;
            }
            if (t_handle->handle.is_cuda())
            {
                if (current_device == t_handle->handle.index())
                {
                    inputs_t[i] = torch::zeros(shape_t, opt);
                    cudaMemcpy2D(inputs_t[i].data_ptr(), shape[2]*shape[3]*n_bit, (uint8_t*)temp->basePtr(), inReqs.strides[1], shape[2]*shape[3]*n_bit, shape[1], cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                }
                else
                {
                    opt = opt.device(torch::Device(c10::DeviceType::CUDA, current_device));
                    inputs_t[i] = torch::zeros(shape_t, opt);
                    cudaMemcpy2D(inputs_t[i].data_ptr(), shape[2]*shape[3]*n_bit, (uint8_t*)temp->basePtr(), inReqs.strides[1], shape[2]*shape[3]*n_bit, shape[1], cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                    inputs_t[i] = inputs_t[i].to(t_handle->handle);
                }
            }
            else
            {
                cudaMemcpy2D(inputs_t[i].data_ptr(), shape[2]*shape[3]*n_bit, (uint8_t*)temp->basePtr(), inReqs.strides[1], shape[2]*shape[3]*n_bit, shape[1], cudaMemcpyKind::cudaMemcpyDeviceToHost);
            }
            have_data = true;
        }

    }

#endif

#ifndef USE_CUDA
#ifndef USE_CVCUDA
    if (have_data == false)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            std::shared_ptr<QyImage_cv> input = std::dynamic_pointer_cast<QyImage_cv>(inputs[i]);
        switch(check_need_scale_offset(i)){
            case 1:
                input=std::dynamic_pointer_cast<QyImage_cv>(input->operator*(input_scale[i]));
                cache_input.push_back(input);
                break;
            case 2:
                input=std::dynamic_pointer_cast<QyImage_cv>(input->operator+(input_offset[i]));
                cache_input.push_back(input);
                break;
            case 3:
                input=std::dynamic_pointer_cast<QyImage_cv>(input->scale_add(input_scale[i],input_offset[i]));
                cache_input.push_back(input);
                break;
            default:
                break;
        }
            std::vector<int64_t> shape_t;
            shape_t.push_back(1);
            shape_t.push_back(input->data.image.rows);
            shape_t.push_back(input->data.image.cols);
            shape_t.push_back(input->data.image.channels());
            cv::Mat temp = input->data.image;
            auto opt = c10::TensorOptions();
            opt = opt.device(t_handle->handle);
            int type = temp.type() & CV_MAT_DEPTH_MASK;
            if (type == CV_32F)
            {
                opt = opt.dtype(c10::ScalarType::Float);
            }
#ifndef USE_IX            
            else if (type == CV_16F)
            {
                opt = opt.dtype(c10::ScalarType::Half);
            }
#endif            
            else if (type == CV_8U)
            {
                opt = opt.dtype(c10::ScalarType::Byte);
            }
            else if (type == CV_8S)
            {
                opt = opt.dtype(c10::ScalarType::Char);
            }
            else if (type == CV_16S)
            {
                opt = opt.dtype(c10::ScalarType::Short);
            }
#ifndef USE_IX            
            else if (type == CV_16U)
            {
                opt = opt.dtype(c10::ScalarType::UInt16);
            }
#endif            
            else if (type == CV_32S)
            {
                opt = opt.dtype(c10::ScalarType::Int);
            }
            else if (type == CV_64F)
            {
                opt = opt.dtype(c10::ScalarType::Double);
            }
            else
            {
                throw Alg_Module_Exception("Error:\t input data type unknown cv mat depth " + std::to_string(type), this->model_name, Alg_Module_Exception::Stage::inference); // 获取模型推理实例异常，一般是因为模型实例还未创建
            }
            inputs_t[i] = torch::zeros(shape_t, opt);
            if (t_handle->handle.is_cuda())
            {
                cudaMemcpy2D(inputs_t[i].data_ptr(), temp.cols * temp.elemSize(), temp.data, temp.step, temp.cols * temp.elemSize(), temp.rows, cudaMemcpyKind::cudaMemcpyHostToDevice);
            }
            else
            {
                cudaMemcpy2D(inputs_t[i].data_ptr(), temp.cols * temp.elemSize(), temp.data, temp.step, temp.cols * temp.elemSize(), temp.rows, cudaMemcpyKind::cudaMemcpyHostToHost);
            }
        }
    }
#endif
#endif

    int instance_id = (this->cnt++ % this->max_instance);
    auto lock = std::unique_lock(*this->instance_mutex[instance_id]);
    auto &net_instance = this->net_instances[instance_id];

#ifdef OPTIMIZE
    c10::InferenceMode guard;
    torch::NoGradGuard no_grad;

#endif

    std::vector<c10::IValue> instance_inputs;
    std::vector<Shape_t> input_shapes = get_input_shapes();
    for (int i = 0; i < inputs_t.size(); i++)
    {
        auto input = inputs_t[i];
        if (inputs_t[i].get_device() != t_handle->handle.index())
        {
            input = inputs_t[i].to(t_handle->handle);
        }

#ifdef USE_FP16
        input = input.to(c10::ScalarType::Half);
#else
        input = input.to(c10::ScalarType::Float);

#endif

        auto current_shape = input.sizes();
        if (input_shapes[i].num_dims != current_shape.size())
        {
            if (input_shapes[i].num_dims - 1 == current_shape.size())
            {

                input = input.unsqueeze(0);
            }
            else
            {
                throw Alg_Module_Exception("Error:\t input data shape mismatch", this->model_name, Alg_Module_Exception::Stage::inference); // 获取模型推理实例异常，一般是因为模型实例还未创建
                return -1;
            }
        }
        current_shape = input.sizes();
        if (current_shape.at(current_shape.size() - 1) != input_shapes[i].dims[input_shapes[i].num_dims - 1])
        {
            if (current_shape.at(current_shape.size() - 1) == input_shapes[i].dims[input_shapes[i].num_dims - 3])
            {
                std::vector<int64_t> permute;
                for (int j = 0; j < input_shapes[i].num_dims - 3; j++)
                {
                    permute.push_back(j);
                }
                permute.push_back(input_shapes[i].num_dims - 1);
                permute.push_back(input_shapes[i].num_dims - 3);
                permute.push_back(input_shapes[i].num_dims - 2);
                input = input.permute(permute);
            }
            else
            {
                throw Alg_Module_Exception("Error:\t input data shape mismatch", this->model_name, Alg_Module_Exception::Stage::inference); // 获取模型推理实例异常，一般是因为模型实例还未创建
                return -1;
            }
        }
        instance_inputs.push_back(input);
    }

    auto instance_outputs = torch_pre_abi::forward(net_instance,instance_inputs);
    if (instance_outputs.isTuple())
    {
        auto datas = instance_outputs.toTupleRef().elements();
        int n_output = 0;
        for (int i = 0; i < datas.size(); i++)
        {
            if (datas[i].isTensor())
            {
                n_output += 1;
            }
        }
        outputs.resize(datas.size());
        int idx = 0;
        for (int i = 0; i < datas.size(); i++)
        {
            if (datas[i].isTensor())
            {
                auto data = datas[i].toTensor();

                auto out_shape = data.sizes();
                for (int j = 0; j < out_shape.size(); j++)
                {
                    outputs[idx].shape.push_back(out_shape.at(j));
                }
                torch::Device cpu_device(c10::DeviceType::CPU);
                data = data.reshape({-1});
                data = data.to(cpu_device);
#ifdef USE_FP16
                data = data.to(c10::ScalarType::Float);
#endif
                int len_data = data.element_size() * data.numel();
                outputs[idx].data.resize(len_data);
                memcpy(outputs[idx].data.data(), data.data_ptr(), len_data);
                idx += 1;
            }
        }
    }
    else if (instance_outputs.isTensorList())
    {
        auto datas = instance_outputs.toTensorList();
        outputs.resize(datas.size());
        int n_output = outputs.size();
        for (int i = 0; i < datas.size(); i++)
        {
            auto data = datas.get(i);

            auto out_shape = data.sizes();
            for (int j = 0; j < out_shape.size(); j++)
            {
                outputs[i].shape.push_back(out_shape.at(j));
            }
            torch::Device cpu_device(c10::DeviceType::CPU);
            data = data.reshape({-1});
            data = data.to(cpu_device);
#ifdef USE_FP16
            data = data.to(c10::ScalarType::Float);
#endif
            int len_data = data.element_size() * data.numel();
            outputs[i].data.resize(len_data);
            memcpy(outputs[i].data.data(), data.data_ptr(), len_data);
        }
    }
    else if (instance_outputs.isTensor())
    {
        auto data = instance_outputs.toTensor();

        outputs.resize(1);
        auto out_shape = data.sizes();
        for (int j = 0; j < out_shape.size(); j++)
        {
            outputs[0].shape.push_back(out_shape.at(j));
        }
        torch::Device cpu_device(c10::DeviceType::CPU);
        data = data.reshape({-1});
        data = data.to(cpu_device);
#ifdef USE_FP16
        data = data.to(c10::ScalarType::Float);
#endif
        int len_data = data.element_size() * data.numel();
        outputs[0].data.resize(len_data);
        memcpy(outputs[0].data.data(), data.data_ptr(), len_data);
    }
    else
    {
        return -1;
    }

    return 0;
};
std::vector<Shape_t> Network_kernel_torch::get_input_shapes()
{
    if (cache_inputs_shapes.size() > 0)
    {
        return cache_inputs_shapes;
    }
    throw Alg_Module_Exception("Error:\t input shape not set in libtorch mode", this->model_name, Alg_Module_Exception::Stage::load_model); // 获取模型推理实例异常，一般是因为模型实例还未创建
    return cache_inputs_shapes;
};

Network_kernel* get_network_kernel(std::shared_ptr<Device_Handle> handle,std::string file_path,std::string model_name,std::vector<Shape_t>& input_shapes,int max_instance){
    Network_kernel* instance=new Network_kernel_torch(handle,file_path,model_name,input_shapes,max_instance);
    return instance;
};

void free_network_kernel(Network_kernel* instance){
    delete instance;
};

bool global_init(){
    torch::set_num_interop_threads(4);
    torch::set_num_threads(2);
    return true;

};
