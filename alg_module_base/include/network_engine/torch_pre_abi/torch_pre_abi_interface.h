#ifndef __TORCH_PRE_ABI_INTERFACE_H__
#define __TORCH_PRE_ABI_INTERFACE_H__

#include <torch/torch.h>
#include <torch/jit.h>

namespace torch_pre_abi{
    torch::jit::script::Module load(const char* file_path,torch::Device& device);
    at::IValue forward(torch::jit::script::Module& module,std::vector<c10::IValue>& inputs);
}

#endif