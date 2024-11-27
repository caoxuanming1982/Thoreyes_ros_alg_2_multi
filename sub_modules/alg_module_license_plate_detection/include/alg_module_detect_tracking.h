#ifndef __ALG_MODULE_DETECT_TRACKING_H__
#define __ALG_MODULE_DETECT_TRACKING_H__
#include "alg_module_base_private.h"
#include "error_type.h"

#include "tracker.h"
#include <time.h>

class Result_item_Detect_Tracking { 
public:
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
    std::string tag="";
    std::vector<float> feature;

    int temp_idx;

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

class Model_cfg_Detect_Tracking:public Model_cfg_base {
public:
    Model_cfg_Detect_Tracking(){
    };
    virtual ~Model_cfg_Detect_Tracking(){

    };
};

class Module_cfg_Detect_Tracking:public Module_cfg_base { 
public:
    Module_cfg_Detect_Tracking(std::string module_name):Module_cfg_base(module_name){

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
            this->cfg_float["thresh_iou"] = 0.4;
        }
        if (this->cfg_float.find("thresh_score")==this->cfg_float.end()) {
            this->cfg_float["thresh_score"] = 0.45;
        }
        if (this->cfg_int_vector.find("classes")==this->cfg_int_vector.end()) {
            std::vector<int> classes;
            this->cfg_int_vector["classes"] = classes;
        }
        return true;
    };  
};

class Channel_cfg_Detect_Tracking:public Channel_cfg_base {
public:
    Channel_cfg_Detect_Tracking(std::string channel_name);
    virtual ~Channel_cfg_Detect_Tracking();
};

class Channel_data_Detect_Trakcing:public Channel_data_base {
public:
    Channel_data_Detect_Trakcing(std::string channel_name);
    virtual ~Channel_data_Detect_Trakcing();

    int frame_width = 0;
    int frame_height = 0;

    float max_cosine_distance = 0.2;
    int nn_budget = 100;
    float max_iou_distance = 0.7;
    int max_age = 30;
    int n_init = 5;

    tracker *mytracker;
    
    void test_deepsort(std::vector<Result_item_Detect_Tracking> &results);
};

class Alg_Module_Detect_Tracking:public Alg_Module_Base_private { 
public:
    Alg_Module_Detect_Tracking();
    virtual ~Alg_Module_Detect_Tracking();

    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path);
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);
    virtual std::shared_ptr<Channel_cfg_base> load_channel_cfg_(std::string channel_name,std::string cfg_path); 
    virtual std::shared_ptr<Channel_data_base> init_channal_data_(std::string channel_name);

    virtual bool init_from_root_dir(std::string root_dir); 
    virtual bool forward(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output); 
    virtual bool filter(std::string channel_name,  std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output);    
    virtual bool display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &filter_output);   

    bool detect_vehicle_object(std::map<std::string,std::shared_ptr<InputOutput>> &input, std::vector<Result_item_Detect_Tracking> &result);

    std::string model_name;
    std::string model_path;
    std::string model_cfg_path;
};

extern "C" Alg_Module_Base* create();
extern "C" void destory(Alg_Module_Base* p);  

#endif