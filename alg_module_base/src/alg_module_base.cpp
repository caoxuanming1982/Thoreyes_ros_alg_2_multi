#include "alg_module_base.h"

std::map<std::string,std::shared_ptr<Alg_Module_Base>> Alg_Module_Base::function_module;

Alg_Module_Base::Alg_Module_Base(){

};
Alg_Module_Base::~Alg_Module_Base(){

};

void Alg_Module_Base::set_function_module(std::string name,std::shared_ptr<Alg_Module_Base> module){
    function_module[name] = module;
}

std::shared_ptr<Alg_Module_Base> Alg_Module_Base::get_function_module(std::string name){
    if(function_module.find(name) != function_module.end()){
        return function_module[name];
    }
    return std::shared_ptr<Alg_Module_Base>();
};
std::vector<std::string> Alg_Module_Base::get_function_module_name(){
    std::vector<std::string> function_module_name;
    for(auto iter=function_module.begin();iter!=function_module.end();iter++){
        function_module_name.push_back(iter->first);
    }
    return function_module_name;
};
