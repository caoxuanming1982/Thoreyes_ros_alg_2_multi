#include "alg_module_bus_detection.h"
#include <iostream>
/*
移植前终版

*/
void roi_pooling(Output& net_output_feature,vector<Result_item_Bus_Detection> &output,int img_h,int img_w)
{
    float* features=(float*)net_output_feature.data.data();
    int f_c=net_output_feature.shape[0];
    int f_h=net_output_feature.shape[1];
    int f_w=net_output_feature.shape[2];
    float factor_h=1.0*f_h/img_h;
    float factor_w=1.0*f_w/img_w;
    int f_size=f_h*f_w;
    for(int i=0;i<output.size();i++){
        output[i].feature.clear();
        int x2=int(output[i].x2*factor_w+1);
        int y2=int(output[i].y2*factor_h+1);
        int x1=int(output[i].x1*factor_w);
        int y1=int(output[i].y1*factor_h);
        float sub=(y2-y1)*(x2-x1);
        output[i].feature.resize(f_c);
        for(int c=0;c<f_c;c++){
            float val=0;
            int offset_c=c*f_size;
            for(int h=y1;h<y2;h++){
                int offset_h=h*f_w;
                for(int w=x1;w<x2;w++){
                    val+=features[w+offset_h+offset_c];
                }
            }
            if(isinf(val))
                val=1.8e17;
            else if(isinf(val)==-1)
                val=-1.8e17;
            else if(isnan(val))
                val=0;
            else{
                val=val/sub;
                if(val>1.8e17)
                    val=1.8e17;
                else if(val<-1.8e17)
                    val=-1.8e17;

            }
            output[i].feature[c]=val;
        }
    }

};
float iou(Result_item_Bus_Detection& box1, Result_item_Bus_Detection& box2)
{
    float x1 = std::max(box1.x1, box2.x1);      //left
    float y1 = std::max(box1.y1, box2.y1);      //top
    float x2 = std::min((box1.x2), (box2.x2));  //right
    float y2 = std::min((box1.y2), (box2.y2));  //bottom
    if(x1>=x2||y1>=y2)
        return 0;
    float over_area = (x2 - x1) * (y2 - y1);
    float box1_w = box1.x2 - box1.x1;
    float box1_h = box1.y2 - box1.y1;
    float box2_w = box2.x2 - box2.x1;
    float box2_h = box2.y2 - box2.y1;
    float iou = over_area / (box1_w * box1_h + box2_w * box2_h - over_area);
    return iou;
};
bool sort_score(Result_item_Bus_Detection& box1, Result_item_Bus_Detection& box2)
{
    return (box1.score > box2.score);


};
inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
};
inline float sigmoid(float x)
{
    if (x >= 87) return 1;
    if (x <= -87) return 0;
    return 1.0f / (1.0f + fast_exp(-x));
    // return 1.0f / (1.0f + exp(-x));
};
void nms_yolo(Output& net_output, vector<Result_item_Bus_Detection>& output, vector<int> class_filter,float threshold_score=0.25,float threshold_iou=0.45)
{
    float* input=(float*)net_output.data.data();
    int dim1=net_output.shape[1];
    int dim2=net_output.shape[2];
    vector<Result_item_Bus_Detection> result;
    float threshold_score_stage_1=threshold_score*0.77;
    for (int k = 0,i=0;k < dim1;k ++,i+=dim2) {
        float obj_conf = input[i + 9];
        obj_conf=sigmoid(obj_conf);
        if (obj_conf > threshold_score_stage_1) {
            Result_item_Bus_Detection item;
            float max_class_conf = input[i + 10];
            int max_class_id = 0;
            for (int j = 1;j < dim2-10;j++) {
                if (input[i + 10 + j] > max_class_conf){
                    max_class_conf = input[i + 10 + j];
                    max_class_id = j;
                }
            }
            max_class_conf = obj_conf*sigmoid(max_class_conf);
            if (max_class_conf > threshold_score_stage_1) {

                float cx=(sigmoid(input[i+5])*2+input[i])*input[i+4];
                float cy=(sigmoid(input[i+6])*2+input[i+1])*input[i+4];
                float w=sigmoid(input[i+7])*2;
                w=w*w*input[i+2];
                float h=sigmoid(input[i+8])*2;
                h=h*h*input[i+3];
                float threshold_score_stage_2=threshold_score*0.77;
                if(abs(cx-320)-160>0)
                    threshold_score_stage_2=threshold_score*(0.67+0.4*w/(abs(cx-320)-160));
                if(abs(cy-176)-88>0)
                    threshold_score_stage_2=threshold_score*(0.67+0.4*h/(abs(cy-176)-88));
                if(threshold_score_stage_2>threshold_score*1.2)
                    threshold_score_stage_2=threshold_score*1.2;

                if(max_class_conf<threshold_score_stage_2)
                    continue;


                item.x1 = cx-w/2;
                item.y1 = cy-h/2;
                item.x2 = cx+w/2;
                item.y2 = cy+h/2;

                item.score = max_class_conf;
                item.class_id = max_class_id;
                item.x1 += item.class_id * 4096;
                item.x2 += item.class_id * 4096;
                if (class_filter.size() > 0) {
                    if (find(class_filter.begin(), class_filter.end(), max_class_id) != class_filter.end()) {
                        result.push_back(item);
                    }
                }
                else{
                    result.push_back(item);
                }
            }
        }
    }
    output.clear();
    if (result.size() <= 0)
        return;

    while (result.size() > 0)
    {
        std::sort(result.begin(), result.end(), sort_score);
        output.push_back(result[0]);
        for (int i = 0;i < result.size() - 1;i++)
        {
            float iou_value = iou(result[0], result[i + 1]);
            if (iou_value > threshold_iou)
            {
                result.erase(result.begin()+i + 1);
                i-=1;
            }
        }
        result.erase(result.begin());
    }
    vector<Result_item_Bus_Detection>::iterator iter=output.begin();
    for(;iter!=output.end();iter++){
        iter->x1-=iter->class_id*4096;
        iter->x2-=iter->class_id*4096;
    }

    return ;
};

Alg_Module_Bus_Detection::Alg_Module_Bus_Detection():Alg_Module_Base_private("bus_detection")
{    //参数是模块名，使用默认模块名初始化

};
Alg_Module_Bus_Detection::~Alg_Module_Bus_Detection()
{

};

/**
 * @brief 从依赖文件根目录读取指定模型文件和配置文件进行初始化
 * @param root_dir
 */
bool Alg_Module_Bus_Detection::init_from_root_dir(std::string root_dir)
{
    bool load_res = true;

    //加载模块配置文件
    load_res = this->load_module_cfg(root_dir + "/cfgs/" + this->node_name + "/module_cfg.xml");
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load module cfg failed",this->node_name,Alg_Module_Exception::Stage::load_module);
        return false;
    }

    std::shared_ptr<Module_cfg_Bus_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Bus_Detection>(this->get_module_cfg());

    //如果文件中有运行频率的字段，则使用文件中设定的频率
    int tick_interval;
    load_res = module_cfg->get_int("tick_interval", tick_interval);
    if (load_res)
        this->tick_interval_ms = tick_interval_ms;
    else
        this->tick_interval_ms = 100;

    //加载模型相关参数
    load_res = module_cfg->get_string("model_path", this->model_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("model_name", this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_name in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("model_cfg_path", this->model_cfg_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no model_cfg_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }

    //加载模型配置文件
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name+"/" + this->model_cfg_path, this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model_cfg_path failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }

    //加载模型
    load_res = this->load_model(root_dir + "/models/" + this->model_path , this->model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }

    return true;
};
/**
 * @brief 推理
 * @param channel_name      通道名称
 * @param input["image"]    输入：需要推理的图片
 * @param output["vehicle"] 输出：检测的省际巴士结果
 */
bool Alg_Module_Bus_Detection::forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::map<std::string, std::shared_ptr<InputOutput>>& output)
{    
    std::shared_ptr<Model_cfg_Bus_Detection> model_cfg = std::dynamic_pointer_cast<Model_cfg_Bus_Detection>(this->get_model_cfg(this->model_name));   //获取模型配置，派生的模型配置文件类指针需要手动转换为子类
    std::shared_ptr<Module_cfg_Bus_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Bus_Detection>(this->get_module_cfg());               //获取模块配置
    
    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        input_image=input["image"]->data.image;
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image type error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        }
    }else{
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;
    }

    //检查参数设置
    std::vector<int> classes;
    float thresh_iou;
    float thresh_score;
    bool load_res = true;
    // load_res &= module_cfg->get_int_vector("classes", classes);
    load_res &= module_cfg->get_float("thresh_score", thresh_score);
    load_res &= module_cfg->get_float("thresh_iou", thresh_iou);
    if (load_res == false)
    {   //找不到必要的配置参数，检查配置文件是否有对应的字段，检查类型，检测名称
        throw Alg_Module_Exception("Error:\t load module params failed",this->node_name,Alg_Module_Exception::Stage::inference);         
        return false;
    }

    //获取指定的模型实例
    auto net = this->get_model_instance(this->model_name);
    if (net == nullptr)
    {   // 模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        throw Alg_Module_Exception("model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);  
        return false;
    }

    auto input_shapes = net->get_input_shapes();
    if (input_shapes.size() <= 0) {
        throw Alg_Module_Exception("Warning:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference); 
        return false;   
    }

    // 获取图像尺寸
    int width=input_image->get_width();
    int height=input_image->get_height();
    // cout<<"width: "<<width<<", height: "<<height<<endl;
    

    /////////////////以下为原版本的yolov5模块的代码//////////////////////
    auto input_shape_ = input_shapes[0];
    float factor1 = input_shape_.dims[3] * 1.0 / width;
    float factor2 = input_shape_.dims[2] * 1.0 / height;

    float factor = factor1 > factor2 ? factor2 : factor1;
    int target_width = width * factor;
    int target_height = height * factor;

    std::vector<Output> net_output;
    std::vector<std::shared_ptr<QyImage>> net_input;

    cv::Mat input_mat = input_image->get_image();
    cout<<"/data/storage/results/diaplay/" + std::to_string(display_count++) + "_ori.jpg"<<endl;
    // cv::imwrite("/data/storage/results/diaplay/" + std::to_string(display_count++) + "_ori.jpg", input_mat);
    cv::imwrite("/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement/results/display/" + std::to_string(display_count) + "_ori.jpg", input_mat);

    // std::shared_ptr<QyImage> sub_image=input_image->resize_keep_ratio(input_shape_.dims[3], input_shape_.dims[2], 0);
    std::shared_ptr<QyImage> sub_image=input_image->resize(input_shape_.dims[3], input_shape_.dims[2]);

    cv::Mat display_mat = sub_image->get_image();
    // cv::imwrite("/data/storage/results/diaplay/" + std::to_string(display_count++) + "_test.jpg", display_mat);
    cv::imwrite("/home/ubuntu/workspace_nv_yan/alg_module_bus_detection/requirement/results/display/" + std::to_string(display_count++) + "_test.jpg", display_mat);
    std::cout<<"保存成功"<<std::endl;

    sub_image=sub_image->cvtcolor(true);
    net_input.push_back(sub_image);
    net->forward(net_input, net_output);  

    std::vector<Result_item_Bus_Detection> detections;
    nms_yolo(net_output[0], detections, classes, thresh_score, thresh_iou);
    for (auto iter = detections.begin(); iter != detections.end(); iter++)
    {
        iter->x1 = (iter->x1 + 0.5) / factor;
        iter->y1 = (iter->y1 + 0.5) / factor;
        iter->x2 = (iter->x2 + 0.5) / factor;
        iter->y2 = (iter->y2 + 0.5) / factor;
    }
    roi_pooling(net_output[1], detections, input_shape_.dims[2] / factor, input_shape_.dims[3] / factor);
    /////////////////以上为原版本的yolov5模块的代码//////////////////////

    //整理数据
    for (int i = 0; i < detections.size(); i++)
    {   
        // cout<<"检出省际客运车辆"<<endl;
        if (detections[i].x1 < 0) detections[i].x1 = 0;
        if (detections[i].y1 < 0) detections[i].y1 = 0;

        if (detections[i].x2 < 0) detections[i].x2 = 0;
        if (detections[i].y2 < 0) detections[i].y2 = 0;

        if (detections[i].x1 > input_image->get_width()) detections[i].x1 = input_image->get_width();
        if (detections[i].x2 > input_image->get_width()) detections[i].x2 = input_image->get_width();
        if (detections[i].y1 > input_image->get_height()) detections[i].y1 = input_image->get_height();
        if (detections[i].y2 > input_image->get_height()) detections[i].y2 = input_image->get_height();
    }

    
    //整理输出数据
    auto forward_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    auto& forward_results = forward_output->data.detect;
    forward_results.resize(detections.size());
    for (int i = 0; i < detections.size(); i++)
    {
        forward_results[i].x1        = detections[i].x1;
        forward_results[i].y1        = detections[i].y1;
        forward_results[i].x2        = detections[i].x2;
        forward_results[i].y2        = detections[i].y2;
        forward_results[i].score     = detections[i].score;
        forward_results[i].class_id  = detections[i].class_id;
        forward_results[i].temp_idx  = i;
        forward_results[i].tag       = "省级客运车辆监管";
    }

    output.clear();
    output["vehicle"] = forward_output;

    return true;
};
/**
 * @brief 整合车辆数据和车牌数据
 * @param channel_name
 * @param input["vehicle"]  输入：省际巴士结果
 * @param input["license"]  输入：车牌结果
 * @param output["result"]  输出：省际巴士和对应车牌的整合结果
 */
bool Alg_Module_Bus_Detection::filter(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::map<std::string, std::shared_ptr<InputOutput>>& output)
{
    //检查是否包含需要的数据
    if (input.find("vehicle") == input.end()) {
        throw Alg_Module_Exception("Error:\t no vehicle in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }
    if (input.find("license") == input.end()) {
        throw Alg_Module_Exception("Error:\t no license in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }

    //未过滤的检测结果
    auto& forward_results_vehicle = input["vehicle"]->data.detect;
    auto& forward_results_license = input["license"]->data.detect_license;

    //整理检测数据
    auto filter_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_license_t); 
    auto& filter_results = filter_output->data.detect_license;
    for (int i = 0; i < forward_results_vehicle.size(); i++)
    {
        Result_Detect_license temp_result;
        temp_result.x1       = forward_results_vehicle[i].x1;
        temp_result.y1       = forward_results_vehicle[i].y1;
        temp_result.x2       = forward_results_vehicle[i].x2;
        temp_result.y2       = forward_results_vehicle[i].y2;
        temp_result.tag      = forward_results_vehicle[i].tag;
        temp_result.car_idx  = forward_results_vehicle[i].temp_idx;
        temp_result.score    = forward_results_vehicle[i].score;
        temp_result.class_id = forward_results_vehicle[i].class_id;
        temp_result.license  = "";

        for (int j = 0; j < forward_results_license.size(); j++)
        {
            if (forward_results_vehicle[i].temp_idx == forward_results_license[j].car_idx)
            {
                temp_result.landms_x1     = forward_results_license[j].landms_x1+temp_result.x1;
                temp_result.landms_x2     = forward_results_license[j].landms_x2+temp_result.x1;
                temp_result.landms_x3     = forward_results_license[j].landms_x3+temp_result.x1;
                temp_result.landms_x4     = forward_results_license[j].landms_x4+temp_result.x1;
                temp_result.landms_y1     = forward_results_license[j].landms_y1+temp_result.y1;
                temp_result.landms_y2     = forward_results_license[j].landms_y2+temp_result.y1;
                temp_result.landms_y3     = forward_results_license[j].landms_y3+temp_result.y1;
                temp_result.landms_y4     = forward_results_license[j].landms_y4+temp_result.y1;
                temp_result.license_type  = forward_results_license[j].license_type;
                temp_result.license_color = forward_results_license[j].license_color;
                temp_result.license       = forward_results_license[j].license;
                // temp_result.license       = "test";
                break;
            }
        }
        
        if (temp_result.license != "") {
            filter_results.push_back(temp_result);
            // std::cout<<"debug"<<std::endl;
            // std::cout<<temp_result.x1<<" "<<temp_result.y1<<" "<<temp_result.x2<<" "<<temp_result.y2<<std::endl;
            // std::cout<<temp_result.landms_x1<<" "<<temp_result.landms_x2<<" "<<temp_result.landms_x3<<" "<<temp_result.landms_x4<<std::endl;
            // std::cout<<temp_result.landms_y1<<" "<<temp_result.landms_y2<<" "<<temp_result.landms_y3<<" "<<temp_result.landms_y4<<std::endl;
        }
    }

    output.clear();
    output["result"] = filter_output;

    return true;
};
/**
 * @brief 可视化：在原图中画出特殊车辆
 * @param channel_name
 * @param input["image"]
 * @param filter_output["result"].data.detect_license.res_image["event_image"]
 */
bool Alg_Module_Bus_Detection::display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>>& input, std::map<std::string,std::shared_ptr<InputOutput>>& filter_output)
{   
    //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }
    if (filter_output.find("result") == filter_output.end()) {
        throw Alg_Module_Exception("Error:\t no result in filter_output",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

    std::shared_ptr<Module_cfg_Bus_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Bus_Detection>(this->get_module_cfg());

    //加载模块需要的参数
    bool load_res = true;
    int box_color_blue;
    int box_color_green;
    int box_color_red;
    int box_thickness;
    load_res &= module_cfg->get_int("box_color_blue", box_color_blue);
    load_res &= module_cfg->get_int("box_color_green", box_color_green);
    load_res &= module_cfg->get_int("box_color_red", box_color_red);
    load_res &= module_cfg->get_int("box_thickness", box_thickness);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t somethine wrong when load param in display",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

    //获取图片
    cv::Mat image;
    if (input["image"]->data_type == InputOutput::Type::Image_t) {
        image = input["image"]->data.image->get_image();
    }
    else {
        //暂时不支持其他类型的图像
        throw Alg_Module_Exception("Error:\t image input type error",this->node_name,Alg_Module_Exception::Stage::display);
        return false;
    }

    //特殊车辆事件
    auto& results = filter_output["result"]->data.detect_license;
    for (int i = 0; i < results.size(); i++)
    {
        cv::Mat image_copy = image.clone();

        int x = results[i].x1;
        int y = results[i].y1;
        int w = results[i].x2 - results[i].x1;
        int h = results[i].y2 - results[i].y1;

        cv::Rect box(x, y, w, h);
        cv::rectangle(image_copy, box, cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness);

        results[i].res_images.insert({"event_image", image_copy});
    }
    
    return true;
};

std::shared_ptr<Module_cfg_base> Alg_Module_Bus_Detection::load_module_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模块配置文件的加载
    auto res = std::make_shared<Module_cfg_Bus_Detection>(this->node_name);
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Model_cfg_base> Alg_Module_Bus_Detection::load_model_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模型配置文件的加载
    auto res = std::make_shared<Model_cfg_Bus_Detection>();
    res->from_file(cfg_path);
    return res;
};

extern "C" Alg_Module_Base *create()        //外部调用的构造函数
{
    return new Alg_Module_Bus_Detection();                     //返回当前算法模块子类的指针
};
extern "C" void destory(Alg_Module_Base *p) //外部调用的析构函数
{
    delete p;
};
