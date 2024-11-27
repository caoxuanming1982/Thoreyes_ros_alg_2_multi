#ifndef __ALG_MODULE_BUS_DETECTION_H__
#define __ALG_MODULE_BUS_DETECTION_H__
#include "alg_module_base_private.h"
#include "error_type.h"
#include <omp.h>

using namespace std;

class Result_item_Bus_Detection {             //模块内使用的YOLO的结果的结构
public:
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
    std::string tag="";
    std::vector<float> feature;

    std::string str(){
        std::string res;
        res+="class:"+Num2string<int>(this->class_id);
        res+="\tscore:"+Num2string<float>(this->score);
        res+="\tx1:"+Num2string<float>(this->x1);
        res+="\ty1:"+Num2string<float>(this->y1);
        res+="\tx2:"+Num2string<float>(this->x2);
        res+="\ty2:"+Num2string<float>(this->y2);
        res+="\ttag:"+this->tag;
        return res;
    };
};

class Model_cfg_Bus_Detection:public Model_cfg_base{  //派生的模型配置文件类
public:
    // bmcv_copy_to_atrr_t copy_to_attr_;
public:
    Model_cfg_Bus_Detection(){
        // copy_to_attr_.start_x = 0;
        // copy_to_attr_.start_y = 0;
        // copy_to_attr_.padding_r = 114;
        // copy_to_attr_.padding_g = 114;
        // copy_to_attr_.padding_b = 114;
        // copy_to_attr_.if_padding = 1;
    };
    virtual ~Model_cfg_Bus_Detection(){

    };
};

class Module_cfg_Bus_Detection:public Module_cfg_base{    //派生的模块配置文件类
public:
    Module_cfg_Bus_Detection(std::string module_name):Module_cfg_base(module_name){

    };
    virtual int from_string_(std::string cfg_str) {
        //目标框相关
        if (this->cfg_int.find("box_color_blue")==this->cfg_int.end()) {
            this->cfg_int["box_color_blue"] = 255;
        }
        if (this->cfg_int.find("box_color_green")==this->cfg_int.end()) {
            this->cfg_int["box_color_green"] = 0;
        }
        if (this->cfg_int.find("box_color_red")==this->cfg_int.end()) {
            this->cfg_int["box_color_red"] = 0;
        }
        if (this->cfg_int.find("box_thickness")==this->cfg_int.end()) {
            this->cfg_int["box_thickness"] = 3;
        }

        //模型推理相关
        if (this->cfg_float.find("thresh_iou")==this->cfg_float.end()) {
            this->cfg_float["thresh_iou"]=0.4;
        }
        if (this->cfg_float.find("thresh_score")==this->cfg_float.end()) {
            this->cfg_float["thresh_score"]=0.45;
        }
        if (this->cfg_int_vector.find("classes")==this->cfg_int_vector.end()) {
            std::vector<int> classes;
            this->cfg_int_vector["classes"]=classes;
        }
        return true;
    };  
};

class Alg_Module_Bus_Detection:public Alg_Module_Base_private{ //派生的算法模块
public:
    Alg_Module_Bus_Detection();
    virtual ~Alg_Module_Bus_Detection();

    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path);    //加载模块配置文件
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);      //加载模型配置文件

    virtual bool init_from_root_dir(std::string root_dir);                              //读取相关配置并进行初始化
    virtual bool forward(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>>& input, std::map<std::string,std::shared_ptr<InputOutput>>& output);    //进行模型推理
    virtual bool filter(std::string channel_name,  std::map<std::string,std::shared_ptr<InputOutput>>& input, std::map<std::string,std::shared_ptr<InputOutput>>& output);     //进行结果过滤
    virtual bool display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>>& input, std::map<std::string,std::shared_ptr<InputOutput>>& filter_output);    //可视化

    std::string model_name;
    std::string model_path;
    std::string model_cfg_path;

    int display_count;
};

extern "C" Alg_Module_Base* create();               //外部使用的构造函数的接口声明
extern "C" void destory(Alg_Module_Base* p);        //外部使用的构造函数的接口声明

#endif