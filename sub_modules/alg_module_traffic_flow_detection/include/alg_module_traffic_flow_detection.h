#ifndef __ALG_MODULE_TRAFFIC_FLOW_DETECTION_H__
#define __ALG_MODULE_TRAFFIC_FLOW_DETECTION_H__
#include "alg_module_base_private.h"
#include "error_type.h"

#include "trajectory_tracker.h"
#include "duplicate_remover.h"
#include <time.h>

class Result_item_Traffic_Flow_Detection {
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

    //道路编号, 对应双向车道中的检测
    int region_id;

    float dy; //车辆行驶方向

    int color = 19; // 0,1,2,3,4,5,6,7,8
    int face_direction = 19; // 9: "面向镜头", 10: "背向镜头"
    int type = 19;

    float color_score=0;
    float face_direction_score=0;
    float type_score=0;

    int lane_id;

    std::string license = "";
    float license_score = 0.0;

    std::string str() {
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

class Model_cfg_Traffic_Flow_Detection:public Model_cfg_base {
public:
    Model_cfg_Traffic_Flow_Detection(){
    };
    virtual ~Model_cfg_Traffic_Flow_Detection(){

    };
};

class Module_cfg_Traffic_Flow_Detection:public Module_cfg_base {
public:
    Module_cfg_Traffic_Flow_Detection(std::string module_name):Module_cfg_base(module_name){

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

        //轨迹相关
        if (this->cfg_int.find("max_record_time_for_event")==this->cfg_int.end()) {
            this->cfg_int["max_record_time_for_event"] = 200;
        }
        if (this->cfg_int.find("max_record_time_for_trajectory")==this->cfg_int.end()) {
            this->cfg_int["max_record_time_for_trajectory"] = 100;
        }
        if (this->cfg_int.find("max_wait_time")==this->cfg_int.end()) {
            this->cfg_int["max_wait_time"] = 10;
        }
        if (this->cfg_int.find("min_point_for_direction")==this->cfg_int.end()) {
            this->cfg_int["min_point_for_direction"] = 3;
        }
        if (this->cfg_int.find("need_vehicle_attr")==this->cfg_int.end()) {
            this->cfg_int["need_vehicle_attr"] = 1;
        }

        //模型推理相关
        if (this->cfg_float.find("thresh_iou")==this->cfg_float.end()) {
            this->cfg_float["thresh_iou"] = 0.4;
        }
        if (this->cfg_float.find("thresh_score")==this->cfg_float.end()) {
            this->cfg_float["thresh_score"] = 0.45;
        }
        if (this->cfg_int_vector.find("classes")==this->cfg_int_vector.end()) {
            std::vector<int> classes = {2,4,5};
            this->cfg_int_vector["classes"] = classes;
        }
        
        if (this->cfg_float.find("covering_solid_line_triggered_iou")==this->cfg_float.end()) {
            this->cfg_float["covering_solid_line_triggered_iou"] = 0.1;
        }
        
        return true;
    };  
};

class Channel_cfg_Traffic_Flow_Detection:public Channel_cfg_base {
public:
    Channel_cfg_Traffic_Flow_Detection(std::string channel_name);
    virtual ~Channel_cfg_Traffic_Flow_Detection();

    std::vector<std::vector<std::pair<float,float>>> get_boundary_self(std::string boundary_name);
};

class Channel_data_Traffic_Flow_Detection:public Channel_data_base {
public:
    Channel_data_Traffic_Flow_Detection(std::string channel_name);
    virtual ~Channel_data_Traffic_Flow_Detection();

    void filter_invalid_objects(std::vector<Result_item_Traffic_Flow_Detection> &objects);

    //检测车流的检测线
    std::vector<std::vector<cv::Point>> boundary_detection_line;
    std::vector<std::vector<cv::Point>> boundary_lane;

    cv::Mat lane_area;

    int frame_width = 0;
    int frame_height = 0;
    
    Trajectory_Tracker *trajectory_tracker; // 存储运动轨迹, 及根据轨迹推测行为, 报出事件
    Duplicate_Remover *duplicate_remover; // 事件的过滤方法
};

class Alg_Module_Traffic_Flow_Detection:public Alg_Module_Base_private {
public:
    Alg_Module_Traffic_Flow_Detection();
    virtual ~Alg_Module_Traffic_Flow_Detection();

    virtual std::shared_ptr<Module_cfg_base> load_module_cfg_(std::string cfg_path);   
    virtual std::shared_ptr<Model_cfg_base> load_model_cfg_(std::string cfg_path);      
    virtual std::shared_ptr<Channel_cfg_base> load_channel_cfg_(std::string channel_name,std::string cfg_path); 
    virtual std::shared_ptr<Channel_data_base> init_channal_data_(std::string channel_name);    

    virtual bool init_from_root_dir(std::string root_dir);                              
    virtual bool forward(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output);    
    virtual bool filter(std::string channel_name,  std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output);     
    virtual bool display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &filter_output);  
    
    bool detect_vehicle_attr(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::vector<Result_item_Traffic_Flow_Detection> &result);


    std::string ca_model_name;
    std::string ca_model_path;
    std::string ca_model_cfg_path;

    int debug = 0;
};

extern "C" Alg_Module_Base* create();
extern "C" void destory(Alg_Module_Base* p);

#endif