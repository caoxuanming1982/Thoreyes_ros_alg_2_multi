#ifndef __TR_CFG_TYPE_BASE_H__
#define __TR_CFG_TYPE_BASE_H__

#include<vector>
#include<chrono>
#include <shared_mutex>
#include<fstream>
#include<sstream>
#include "common.h"
#include <map>
#include "tinyxml2.h"
#include <shared_mutex>
#include<fstream>
#include<sstream>
#include<unistd.h>
#include "inout_type.h"
#include "publish_cfg_base.h"
#include "post_process_cfg_base.h"

typedef enum Ext_Type{EXTEND,OVERWRITE};

class Channel_data_base{
public:
    std::string channel_name;

public:
    Channel_data_base(std::string channel_name);
    virtual ~Channel_data_base();

};

class Channel_cfg_base{
protected:
    std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> boundary;
public:
    std::string channel_name;
    Ext_Type ext_type=Ext_Type::EXTEND;

    Channel_cfg_base(std::string channel_name);
    virtual ~Channel_cfg_base();
    int from_file(std::string cfg_path);
    virtual int from_string_(std::string cfg_str);
    int from_string(std::string cfg_str);

    std::vector<std::pair<std::string,std::vector<std::pair<float,float>>>> get_boundary(std::string name);
    virtual std::string to_string();


};

class Model_cfg_base{
public:
    int mem_require_mbyte=50;
    int tpu_mem_require_mbyte=100;
    int cpu_util_require=5;
    int tpu_util_require=5;

    Ext_Type ext_type=Ext_Type::EXTEND;
    std::vector<Shape_t> input_shapes;

    Model_cfg_base();
    virtual ~Model_cfg_base();
    virtual int from_string_(std::string cfg_str);
    int from_string(std::string cfg_str);
    int from_file(std::string cfg_path);
    virtual std::string to_string();

};


class Module_cfg_base{
protected:
    std::string module_name;
    std::map<std::string,float> cfg_float;
    std::map<std::string,int>  cfg_int;
    std::map<std::string,std::string> cfg_string;

    std::map<std::string,std::vector<float>> cfg_float_vector;
    std::map<std::string,std::vector<int>>  cfg_int_vector;

    std::shared_ptr<InputOutput_cfg> input_output_cfg;
    std::shared_ptr<Publish_cfg> publish_cfg;
    std::shared_ptr<Post_process_cfg_base> post_process_cfg;

public:
    Ext_Type ext_type=Ext_Type::EXTEND;

    Module_cfg_base(std::string module_name);
    virtual ~Module_cfg_base();
    virtual int from_string_(std::string cfg_str);
    int from_string(std::string cfg_str);
    int from_file(std::string cfg_path);

    virtual bool get_float(std::string element_name,float& res);
    virtual bool get_int(std::string element_name,int& res);
    virtual bool get_string(std::string element_name,std::string& res);

    virtual bool get_float_vector(std::string element_name,std::vector<float>& res);
    virtual bool get_int_vector(std::string element_name,std::vector<int>& res);

    std::shared_ptr<InputOutput_cfg> get_input_output_cfg();
    std::shared_ptr<Publish_cfg> get_publish_cfg();
    std::shared_ptr<Post_process_cfg_base> get_post_process_cfg();

    virtual std::string to_string();
    std::string get_module_name();

};
#endif