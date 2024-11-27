#include"tr_alg_engine/hw/tr_alg_engine_hw.h"

Tr_Alg_Engine_module_hw::Tr_Alg_Engine_module_hw():Tr_Alg_Engine_module("hw"){
};

Tr_Alg_Engine_module_hw::~Tr_Alg_Engine_module_hw(){

};
bool Tr_Alg_Engine_module_hw::init_(std::string submodule_dir,std::string requirement_dir){
    engine.set_add_reduce_instance_thres(0.5,0.2);
    if (submodule_dir==""){
        submodule_dir="/data/thoreyes_hw/ros/alg_module_submodules/lib/";
    }
    if(requirement_dir==""){
        requirement_dir="/data/thoreyes_hw/ros/requirement/";
    }
    try
    {
        if (load_module_from_dir(submodule_dir,requirement_dir) == false)
        {
            return false;
        }
        if (check_loaded_modules() == false)
        {
            return false;
        }
    }
    catch (Alg_Module_Exception &exception)
    {
        std::cerr << exception.module_name << " in " << exception.Stage2string(exception.stage) << " Error:" << exception.what()<<std::endl;
        return false;
    }
    return true;


};
std::shared_ptr<Tr_Alg_Engine_module> get_engine_node(){
    return std::make_shared<Tr_Alg_Engine_module_hw>();
};
