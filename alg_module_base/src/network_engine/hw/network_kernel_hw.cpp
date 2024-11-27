#include "network_engine/hw/network_kernel_hw.h"
#include "network_engine/hw/device_handle_hw.h"
#include "cv_lib/type_def.h"
#include "cv_lib/hw/type_def_hw.h"
#include "error_type.h"

DevicePtr::DevicePtr() {

};
DevicePtr::~DevicePtr()
{
    free();
}

void DevicePtr::resize(int size_in)
{
    if (size_in != size)
    {
        free();
        aclError aclRet = aclrtMalloc(&data, size_in, ACL_MEM_MALLOC_HUGE_FIRST);
        size = size_in;
    }
};

void DevicePtr::from_model_input(aclmdlDesc *modelDesc_, int idx)
{
    free();
    size = aclmdlGetInputSizeByIndex(modelDesc_, idx);
    aclError aclRet = aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST);
};
void DevicePtr::from_model_output(aclmdlDesc *modelDesc_, int idx)
{
    free();
    size = aclmdlGetOutputSizeByIndex(modelDesc_, idx);
    aclError aclRet = aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST);
};
void DevicePtr::free()
{
    if (data != nullptr)
    {
        aclrtFree(data);
        data = nullptr;
        size = 0;
    }
};

aclDataBuffer *DevicePtr::get_buffer()
{
    if (modelInputBuffer != nullptr)
    {
        return aclCreateDataBuffer(data, size);
    }
    return nullptr;
}

Network::Network(std::string path, std::shared_ptr<Device_Handle> handle)
{
    current_handle = handle;
    int length = path.length();
    omModelPath = new char[length + 1];
    strcpy(omModelPath, path.c_str());
    omModelPath[length] = '\0';
};

Network::~Network()
{
    aclError ret = aclrtSetDevice(current_handle->get_device_id());
    free_input_buffer();
    free_output_buffer();
    if (modelId_ != 4294967295)
        ret = aclmdlUnload(modelId_);
    if (modelDesc_ != nullptr)
    {
        aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelWeightPtr_ != nullptr)
    {
        aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
    }
    if (modelMemPtr_ != nullptr)
    {
        aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
    }
    if (omModelPath != nullptr)
    {
        delete[] omModelPath;
        omModelPath = nullptr;
    }
}
bool Network::init()
{
    aclError ret = aclrtSetDevice(current_handle->get_device_id());
    ret = aclmdlQuerySize(omModelPath, &modelMemSize_, &modelWeightSize_);
    ret = aclrtMalloc(&modelMemPtr_, modelMemSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclmdlLoadFromFileWithMem(omModelPath, &modelId_, modelMemPtr_, modelMemSize_, modelWeightPtr_, modelWeightSize_);
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    prepare_input_buffer();
    prepare_output_buffer();
    return true;
};

std::vector<Shape_t> Network::get_input_shapes()
{
    std::vector<Shape_t> result;
    result.resize(input_buffer.size());
    if (modelDesc_ != nullptr)
    {
        for (int i = 0; i < input_buffer.size(); i++)
        {
            aclmdlIODims shape;
            aclmdlInOutputDims(modelDesc_, i, &shape);
            result[i].num_dims = shape.dimCount;
            for (int j = 0; j < shape.dimCount; j++)
            {
                result[i].dims[j] = shape.dims[j];
            }
        }
    }
    return result;
};

bool Network::prepare_input_buffer()
{
    if (modelDesc_ == nullptr)
    {
        return false;
    }

    aclError ret = aclrtSetDevice(current_handle->get_device_id());
    size_t inputNum = aclmdlGetNumInputs(modelDesc_);
    input_buffer.resize(inputNum);
    for (int i = 0; i < input_buffer.size(); i++)
    {
        input_buffer[i].from_model_input(modelDesc_, i);
    }
    inputs = aclmdlCreateDataset();
    for (int i = 0; i < input_buffer.size(); i++)
    {
        ret = aclmdlAddDatasetBuffer(inputs, input_buffer[i].get_buffer());
    }
    return true;
};

bool Network::free_input_buffer()
{
    if (inputs != nullptr)
    {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(inputs); ++i)
        {
            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(inputs, i);
            (void)aclDestroyDataBuffer(dataBuffer);
        }
        (void)aclmdlDestroyDataset(inputs);
        inputs = nullptr;
        input_buffer.clear();
    }
    return true;
};

bool Network::prepare_output_buffer()
{
    if (modelDesc_ == nullptr)
    {
        return false;
    }
    aclError ret = aclrtSetDevice(current_handle->get_device_id());
    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);
    output_buffer.resize(outputNum);
    for (int i = 0; i < output_buffer.size(); i++)
    {
        output_buffer[i].from_model_output(modelDesc_, i);
    }
    outputs = aclmdlCreateDataset();
    for (int i = 0; i < output_buffer.size(); i++)
    {
        ret = aclmdlAddDatasetBuffer(outputs, output_buffer[i].get_buffer());
    }
    return true;
};

bool Network::free_output_buffer()
{
    if (outputs != nullptr)
    {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(outputs); ++i)
        {
            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(outputs, i);
            (void)aclDestroyDataBuffer(dataBuffer);
        }
        (void)aclmdlDestroyDataset(outputs);
        outputs = nullptr;
        output_buffer.clear();
    }
    return true;
};

bool Network::forward(std::vector<std::shared_ptr<QyImage_hw>> &inputs_, std::vector<Output> &outputs_, std::vector<cv::Scalar> &scale, std::vector<cv::Scalar> &offset)
{
    aclError ret = aclrtSetDevice(current_handle->get_device_id());
    aclDataType compute_type = aclDataType::ACL_FLOAT;
    if (use_fp16)
    {
        compute_type = aclDataType::ACL_FLOAT16;
    }

    for (int i = 0; i < input_buffer.size(); i++)
    {
        uint32_t peer_id = inputs_[i]->get_handle()->get_device_id();
        aclmdlIODims shape;
        aclmdlInOutputDims(modelDesc_, i, &shape);
        aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(1);
        ret = aclmdlSetAIPPSrcImageSize(aippDynamicSet, inputs_[i]->get_width(), inputs_[i]->get_height());
        ret = aclmdlSetAIPPInputFormat(aippDynamicSet, ACL_RGB888_U8);
        if (inputs_[i]->get_is_rgb())
        {
            ret = aclmdlSetAIPPRbuvSwapSwitch(aippDynamicSet, 0);
        }
        else
        {
            ret = aclmdlSetAIPPRbuvSwapSwitch(aippDynamicSet, 1);
        }
        ret = aclmdlSetAIPPDtcPixelMean(aippDynamicSet, 0, 0, 0, 0, 0);
        cv::Scalar scale_t(1, 1, 1, 1);
        cv::Scalar offset_t(0, 0, 0, 0);
        if (scale.size() >= i)
        {
            scale_t = scale[i];
        }
        if (offset.size() >= i)
        {
            offset_t = offset[i];
        }
        offset_t = offset_t / scale_t;
        ret = aclmdlSetAIPPDtcPixelMin(aippDynamicSet, offset_t.val[0], offset_t.val[1], offset_t.val[2], offset_t.val[3], 0);
        ret = aclmdlSetAIPPPixelVarReci(aippDynamicSet, scale_t.val[0], scale_t.val[1], scale_t.val[2], scale_t.val[3], 0);
        if (peer_id == current_handle->get_device_id())
        {

            ret = aclrtMemcpy(inputs[i].data, inputs[i].size, inputs_[i]->data.image.picture_address, inputs_[i]->data.image.picture_buffer_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
        }
        else
        {
            int32_t can_access;
            aclrtDeviceCanAccessPeer(&can_access, current_handle->get_device_id(), peer_id);
            if (can_access == 1)
            {
                ret = aclrtDeviceEnablePeerAccess(peer_id, 0);
                ret = aclrtDeviceEnablePeerAccess(current_handle->get_device_id(), 0);
                ret = aclrtMemcpy(inputs[i].data, inputs[i].size, inputs_[i]->data.image.picture_address, inputs_[i]->data.image.picture_buffer_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
                ret = aclrtDeviceDisablePeerAccess(peer_id);
                ret = aclrtDeviceDisablePeerAccess(current_handle->get_device_id());
            }
            else
            {

                std::vector<uint8_t> temp;
                temp.resize(inputs_[i]->data.image.picture_buffer_size);
                ret = aclrtMemcpy(temp.data(), temp.size(), inputs_[i]->data.image.picture_address, inputs_[i]->data.image.picture_buffer_size, ACL_MEMCPY_DEVICE_TO_HOST);
                ret = aclrtMemcpy(inputs[i].data, inputs[i].size, temp.data(), temp.size(), ACL_MEMCPY_HOST_TO_DEVICE);
            }
        }

        ret = aclmdlSetInputAIPP(modelId_, inputs, i, aippDynamicSet);
        ret = aclmdlDestroyAIPP(aippDynamicSet);
    }

    aclError ret = aclmdlExecute(modelId_, inputs, outputs);
    outputs_.resize(output_buffer.size());

    for (int i = 0; i < output_buffer.size(); i++)
    {
        aclmdlIODims shape;
        aclmdlGetOutputDims(modelDesc_, i, &shape);
        outputs_[i].shape.resize(shape.dimCount);
        for (int j = 0; j < shape.dimCount; j++)
        {
            outputs_[i].shape[j] = shape.dims[j];
        }
        outputs_[i].data.resize(output_buffer[i].size);
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(outputs, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
        ret = aclrtMemcpy(outputs_[i].data.data(), output_buffer[i].size, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
    }
}

Network *Network::copy(std::shared_ptr<Device_Handle> handle)
{
    Network *net = nullptr;
    uint32_t peer_id = handle->get_device_id();
    aclError ret;
    ret = aclrtSetDevice(peer_id);
    if (peer_id == current_handle->get_device_id() || aclrtDeviceEnablePeerAccess(peer_id, 0) == 0)
    {

        net = new Network(std::string(omModelPath), handle);
        ret = aclrtDeviceEnablePeerAccess(peer_id, 0);
        ret = aclrtDeviceEnablePeerAccess(current_handle->get_device_id(), 0);

        net->modelMemSize_ = modelMemSize_;
        net->modelWeightSize_ = modelWeightSize_;
        ret = aclrtMalloc(&net->modelMemPtr_, modelMemSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclrtMalloc(&net->modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclmdlLoadFromFileWithMem(omModelPath, &net->modelId_, net->modelMemPtr_, modelMemSize_, net->modelWeightPtr_, modelWeightSize_);

        aclrtMemcpy(net->modelMemPtr_, net->modelMemSize_, modelMemPtr_, modelMemSize_, aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE);
        aclrtMemcpy(net->modelWeightPtr_, net->modelWeightSize_, modelWeightPtr_, modelWeightSize_, aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE);
        ret = aclrtDeviceDisablePeerAccess(peer_id);
        ret = aclrtDeviceDisablePeerAccess(current_handle->get_device_id());
        ret = aclrtSetDevice(current_handle->get_device_id());
    }
    else
    {
        net = new Network(std::string(omModelPath), handle);
        ret = net->init();
    }
    return net;
}
std::string convert_model_path(std::string model_path)
{
    auto model_path_t = std::filesystem::path(model_path);
    if (model_path_t.has_extension())
    {
        std::string ext = model_path_t.extension().string();
        if (ext != "om")
        {
            model_path_t.replace_extension("om");
        }
        model_path = model_path_t.string();
    }
    else
    {
        model_path = model_path_t.string() + ".om";
    }
    return model_path;
}

Network_kernel_hw::Network_kernel_hw(std::shared_ptr<Device_Handle> handle, std::string file_path, std::string model_name, std::vector<Shape_t> &input_shapes, int max_instance) : Network_kernel(handle, file_path, model_name, input_shapes, max_instance)
{
    std::shared_ptr<Device_Handle_hw> t_handle = std::dynamic_pointer_cast<Device_Handle_hw>(handle);
    if (t_handle == nullptr)
    {
        throw std::runtime_error(file_path + "    " + model_name + "   model kernel get invalid device handle");
    }
    for (int i = 0; i < max_instance; i++)
    {
        this->net_instances.push_back(new Network(model_name.c_str(), handle));
        this->net_instances[i]->init();
        this->instance_mutex.push_back(new std::shared_mutex());
    }
};

int Network_kernel_hw::forward(std::vector<std::shared_ptr<QyImage>> &inputs, std::vector<Output> &outputs)
{
    int instance_id = (this->cnt++ % this->max_instance);
    auto lock = std::unique_lock(*this->instance_mutex[instance_id]);
    auto &instance = this->net_instances[instance_id];

    std::vector<std::shared_ptr<QyImage_hw>> inputs_converted;
    for (int i = 0; i < inputs.size(); i++)
    {
        std::shared_ptr<QyImage_hw> temp = std::dynamic_pointer_cast<QyImage_hw>(inputs[i]);
        inputs_converted.push_back(temp);
    }
    instance->forward(inputs_converted, outputs,this->input_scale,this->input_offset);
};
std::vector<Shape_t> Network_kernel_hw::get_input_shapes()
{
    if (this->net_instances.size() <= 0)
    {
        return std::vector<Shape_t>();
    }
    return net_instances[0]->get_input_shapes();
};

Network_kernel *get_network_kernel(std::shared_ptr<Device_Handle> handle, std::string file_path, std::string model_name, std::vector<Shape_t> &input_shapes, int max_instance)
{
    Network_kernel *instance = new Network_kernel_hw(handle, file_path, model_name, input_shapes, max_instance);
    return instance;
};
void free_network_kernel(Network_kernel *instance)
{
    delete instance;
};

bool global_init()
{
    return true;
};
