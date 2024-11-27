#undef _GLIBCXX_USE_CXX11_ABI
#define _GLIBCXX_USE_CXX11_ABI 0
#include "network_engine/torch_pre_abi/torch_pre_abi_interface.h"
#include <iostream>
#include <torch/script.h>
#include <regex>

void dummy() {
    std::regex regstr("Why");
    std::string s = "Why crashed";
    std::regex_search(s, regstr);
};
namespace torch_pre_abi{

torch::jit::script::Module load(const char* file_path,torch::Device& device){
    std::string path(file_path);
    return torch::jit::load(path,device);
};

at::IValue forward(torch::jit::script::Module& module,std::vector<c10::IValue>& inputs){
    return module.forward(inputs);

};
};
