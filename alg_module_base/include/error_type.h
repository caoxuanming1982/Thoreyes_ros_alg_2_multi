#ifndef __ERROR_TYPE_H__
#define __ERROR_TYPE_H__
#include <iostream>
#include <stdexcept>
#include <string>
#include <boost/stacktrace.hpp>

class Alg_Module_Exception:public std::runtime_error{
public:
    enum Stage{load_module,load_model,load_channel,init_channel_data,check,inference,filter,display};    
    Stage stage;
    std::string module_name;
    std::string channel_name;
    boost::stacktrace::stacktrace st;
    std::string stack_string;
    std::string exception_string;

    static std::string Stage2string(Stage stage){
        switch(stage){
            case   Stage::load_module: return "load_module";
            case   Stage::load_model: return "load_model";
            case   Stage::load_channel: return "load_channel";
            case   Stage::init_channel_data: return "init_channel_data";
            case   Stage::check: return "check";
            case   Stage::inference: return "inference";
            case   Stage::filter: return "filter";
            case   Stage::display: return "display";
        }
        return "unknow";
    };

public:
    Alg_Module_Exception(std::string msg,std::string module_name,Stage stage,std::string channel_name=""):std::runtime_error(msg){
        this->module_name=module_name;
        this->stage=stage;
        this->channel_name=channel_name;
        std::ostringstream stream;
        stream<<"########################################################\n";
        stream<<this->module_name<<"\t"<<Stage2string(stage)<<"\t"<<channel_name<<"\n";
        stream<<"Exception:\n";
        stream<<this->what()<<"\n";
        exception_string=stream.str();
    };
    virtual ~Alg_Module_Exception(){};

    template<class Allocator>
    void set_trace(boost::stacktrace::basic_stacktrace<Allocator>& bt){
        std::ostringstream stream;
        stream<<"Exception Stack:\n";
        stream<<bt<<"\n";
        stack_string=stream.str();
    };
};
#endif