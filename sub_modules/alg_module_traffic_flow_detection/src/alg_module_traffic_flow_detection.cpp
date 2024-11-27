#include "alg_module_traffic_flow_detection.h"
#include <iostream>
float iou(Result_item_Traffic_Flow_Detection &box1, Result_item_Traffic_Flow_Detection &box2)
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

std::string attr_label_to_str(int label)
{
    switch (label)
    {
    case 0:
        return "black";
    case 1:
        return "blue";
    case 2:
        return "brown";
    case 3:
        return "gray";
    case 4:
        return "green";
    case 5:
        return "pink";
    case 6:
        return "red";
    case 7:
        return "white";
    case 8:
        return "yellow";
    case 9:
        return "face_camera";
    case 10:
        return "back_camera";
    case 11:
        return "bus";
    case 12:
        return "car";
    case 13:
        return "engineering";
    case 14:
        return "suv";
    case 15:
        return "trailer";
    case 16:
        return "lorry";
    case 17:
        return "van";
    case 18:
        return "truck";
    default:
        return "unknown";
    }
};
void decode_attr(Output &output, Result_item_Traffic_Flow_Detection &result) 
{
    float* input = (float*)output.data.data();
    int dim0 = output.shape[0];
    int dim1 = output.shape[1];

    // std::vector<string> attr_classes = {
    //     "黑色","蓝色","棕色","灰色","绿色","粉色","红色","白色","黄色",
    //     "面向镜头","背向镜头",
    //     "公交车","轿车","工程车","SUV","拖车","卡车","面包车","货车"
    // };
    
    int max_label = 0;
    float max_score = input[max_label];
    for (int i = 1; i < 9; i++)
    {
        if (input[i] > max_score)
        {
            max_label = i;
            max_score = input[i];
        }
    }
    result.color = max_label;
    result.color_score = max_score;

    max_label = 9;
    max_score = input[max_label];
    for (int i = 10; i < 11; i++)
    {
        if (input[i] > max_score)
        {
            max_label = i;
            max_score = input[i];
        }
    }
    result.face_direction = max_label;
    result.face_direction_score = max_score;

    max_label = 11;
    max_score = input[max_label];
    for (int i = 12; i < 19; i++)
    {
        if (input[i] > max_score)
        {
            max_label = i;
            max_score = input[i];
        }
    }
    result.type = max_label;
    result.type_score = max_score;
};
void decode_attr(std::vector<float> &input, Result_item_Traffic_Flow_Detection &result) 
{
    // std::vector<string> attr_classes = {
    //     "黑色","蓝色","棕色","灰色","绿色","粉色","红色","白色","黄色",
    //     "面向镜头","背向镜头",
    //     "公交车","轿车","工程车","SUV","拖车","卡车","面包车","货车"
    // };
    
    int max_label = 0;
    float max_score = input[max_label];
    for (int i = 1; i < 9; i++)
    {
        if (input[i] > max_score)
        {
            max_label = i;
            max_score = input[i];
        }
    }
    result.color = max_label;
    result.color_score = max_score;
    if(result.color_score<1e-7)
        result.color=19;

    max_label = 9;
    max_score = input[max_label];
    for (int i = 10; i < 11; i++)
    {
        if (input[i] > max_score)
        {
            max_label = i;
            max_score = input[i];
        }
    }
    result.face_direction = max_label;
    result.face_direction_score = max_score;
    if(result.face_direction_score<1e-7)
        result.face_direction=19;

    max_label = 11;
    max_score = input[max_label];
    for (int i = 12; i < 19; i++)
    {
        if (input[i] > max_score)
        {
            max_label = i;
            max_score = input[i];
        }
    }
    result.type = max_label;
    result.type_score = max_score;
    if(result.type_score<1e-7)
        result.type=19;
};

Channel_cfg_Traffic_Flow_Detection::Channel_cfg_Traffic_Flow_Detection(std::string channel_name):Channel_cfg_base(channel_name)
{
    this->channel_name = channel_name;
};
Channel_cfg_Traffic_Flow_Detection::~Channel_cfg_Traffic_Flow_Detection() 
{

};
std::vector<std::vector<std::pair<float,float>>> Channel_cfg_Traffic_Flow_Detection::get_boundary_self(std::string boundary_name)
{
    std::vector<std::vector<std::pair<float,float>>> boundary_;
    for (int i = 0; i < this->boundary.size(); ++i)
    {
        if (this->boundary[i].first != boundary_name) continue;

        boundary_.push_back(this->boundary[i].second);
    }
    return boundary_;
};

Channel_data_Traffic_Flow_Detection::Channel_data_Traffic_Flow_Detection(std::string channel_name):Channel_data_base(channel_name)
{
    this->channel_name = channel_name;
};
Channel_data_Traffic_Flow_Detection::~Channel_data_Traffic_Flow_Detection() 
{

};

/// @brief 通过区域过滤无效的目标
/// @param objects 需要被过滤的目标
void Channel_data_Traffic_Flow_Detection::filter_invalid_objects(std::vector<Result_item_Traffic_Flow_Detection> &objects) 
{
    if (this->boundary_lane.size() == 0) { // 不存在边界就移除所有
        objects.clear();
        return;
    }

    // 对检测结果进行移除
    std::vector<Result_item_Traffic_Flow_Detection>::iterator object = objects.begin();
    for (; object != objects.end(); )
    {
        object->region_id = int(this->lane_area.at<u_char>((object->y1+object->y2)/2, (object->x1+object->x2)/2)) - 1;
        if (object->region_id == -1) {
            object = objects.erase(object);
        } else {
            object++;
        }
    }
    return;
};

Alg_Module_Traffic_Flow_Detection::Alg_Module_Traffic_Flow_Detection():Alg_Module_Base_private("traffic_flow_detection")
{    //参数是模块名，使用默认模块名初始化

};
Alg_Module_Traffic_Flow_Detection::~Alg_Module_Traffic_Flow_Detection()
{

};

/// @brief 车辆信息识别
/// @param channel_name 通道名称
/// @param input_image 输入图片
/// @param handle 设备句柄
/// @param result 检测结果
bool Alg_Module_Traffic_Flow_Detection::detect_vehicle_attr(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>>& input, std::vector<Result_item_Traffic_Flow_Detection> &result)
{
    if (result.size() == 0) return true;
    
    //获取指定的模型实例
    auto net = this->get_model_instance(this->ca_model_name);
    if (net == nullptr)
    {   //模型找不到，要么是模型文件不存在，要么是模型文件中的模型名字不一致
        throw Alg_Module_Exception("Error:\t model instance get fail",this->node_name,Alg_Module_Exception::Stage::inference);  
        return false;
    }
    
    //判断模型是否已经加载
    auto input_shapes = net->get_input_shapes();
    if (input_shapes.size() <= 0)
    {   //获取模型推理实例异常，一般是因为模型实例还未创建
        throw Alg_Module_Exception("Error:\t model not loaded",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }

    auto input_shape_ = input_shapes[0];    //模型输入尺寸: [channel, height, width]
    int target_width = input_shape_.dims[3];
    int target_height = input_shape_.dims[2];

    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        input_image=input["image"]->data.image;
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image type error",this->node_name,Alg_Module_Exception::Stage::inference);
        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;

    }
    
        

#ifdef _OPENMP
#pragma omp parallel for
#endif   
    for (int i = 0; i < result.size(); i++)
    {
        // 获取车辆部分的图片
        cv::Rect crop_rect;
        crop_rect.x = result[i].x1;
        crop_rect.y = result[i].y1;
        crop_rect.width = result[i].x2 - result[i].x1;
        crop_rect.height= result[i].y2 - result[i].y1;


        float image_width = result[i].x2 - result[i].x1;
        float image_height = result[i].y2 - result[i].y1;

        if((image_width < 8) || (image_width > 8192)){
            std::cout << "ch " << channel_name << " model " <<  this->ca_model_name << " image_width invalid " << image_width<< std::endl;
            return false;
        }
        if((image_height < 8) || (image_height > 8192)){
            std::cout << "ch " << channel_name << " model " <<  this->ca_model_name << " image_height invalid " << image_height<< std::endl;
            return false;
        }
        if((target_width < 8) || (target_width > 8192)){
            std::cout << "ch " << channel_name << " model " <<  this->ca_model_name << " target_width invalid " << target_width << std::endl;
            return false;
        }
        if((target_height < 8) || (target_height > 8192)){
            std::cout << "ch " << channel_name << " model " <<  this->ca_model_name << " target_height invalid " << target_height << std::endl;
            return false;
        }
        std::vector<Output> net_output;
        std::shared_ptr<QyImage> sub_image=input_image->crop_resize(crop_rect,input_shape_.dims[3],input_shape_.dims[2]);
        sub_image=sub_image->cvtcolor(true);

        std::vector<std::shared_ptr<QyImage>> net_input;         //这一部分与原版本不一样
        net_input.push_back(sub_image);
        net->forward(net_input, net_output);  

        // 结果解码
        decode_attr(net_output[0], result[i]);

    }

    return true;
};

/// @brief 初始化
/// @param root_dir 
bool Alg_Module_Traffic_Flow_Detection::init_from_root_dir(std::string root_dir)
{
    bool load_res = true;

    //加载模块配置文件
    load_res = this->load_module_cfg(root_dir + "/cfgs/" + this->node_name + "/module_cfg.xml");
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load module cfg failed",this->node_name,Alg_Module_Exception::Stage::load_module);
        return false;
    }

    std::shared_ptr<Module_cfg_Traffic_Flow_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Traffic_Flow_Detection>(this->get_module_cfg());

    //如果文件中有运行频率的字段，则使用文件中设定的频率
    int tick_interval;
    load_res = module_cfg->get_int("tick_interval", tick_interval);
    if (load_res)
        this->tick_interval_ms = tick_interval_ms;
    else
        this->tick_interval_ms = 100;

    //加载模型相关参数
    load_res = module_cfg->get_string("ca_model_name", this->ca_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no ca_model_name in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("ca_model_path", this->ca_model_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no ca_model_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }
    load_res = module_cfg->get_string("ca_model_cfg_path", this->ca_model_cfg_path);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t no ca_model_cfg_path in cfgs",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }


    load_res = module_cfg->get_int("debug", this->debug);
    if (!load_res) {
        this->debug = 0;
    } 
    if (this->debug > 0&&this->debug!=4) {
        std::string debug_path = "/data/storage/" + this->get_module_name();
        std::string debug_path_input_image = "/data/storage/" + this->get_module_name() + "/input_image";
        std::string debug_path_event_image = "/data/storage/" + this->get_module_name() + "/event_image";

        //创建保存事件发生时截图的图片
        std::string cmd1 = "mkdir " + debug_path;
        std::string cmd2 = "mkdir " + debug_path_input_image;
        std::string cmd3 = "mkdir " + debug_path_event_image;
        system(cmd1.c_str());
        system(cmd2.c_str());
        system(cmd3.c_str());
    }

    //加载模型配置文件
    load_res = this->load_model_cfg(root_dir + "/cfgs/" + this->node_name+"/" + this->ca_model_cfg_path, this->ca_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load ca_model_cfg_path failed",this->node_name,Alg_Module_Exception::Stage::check);
        return false;
    }

    //加载模型
    load_res = this->load_model(root_dir + "/models/" + this->ca_model_path , this->ca_model_name);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load model failed",this->node_name,Alg_Module_Exception::Stage::load_model);
        return false;
    }
    
    return true;
};

/// @brief 推理
/// @param channel_name 通道名称
/// @param input['image'] 输入图片
/// @param input['vehicle'] 车辆目标
/// @param input['license'] 车牌目标
/// @param output['vehicle'] 车辆目标（带有车辆信息）
/// @param output['license'] 车牌目标
/// @return 
bool Alg_Module_Traffic_Flow_Detection::forward(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output)
{    
    if (this->debug == 1 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Forward Begin: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
        
    }

    //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }
    if (input.find("vehicle") == input.end()) {
        throw Alg_Module_Exception("Error:\t no vehicle in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }
    if (input.find("license") == input.end()) {
        throw Alg_Module_Exception("Error:\t no license in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }
    if (this->debug == 1 || this->debug == 4) {
        std::cout << this->get_module_name() << " " << input["vehicle"]->data.detect.size()<<"\t"<< input["license"]->data.detect_license.size()<< std::endl;
        
    }

    
    std::shared_ptr<Module_cfg_Traffic_Flow_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Traffic_Flow_Detection>(this->get_module_cfg());   
    std::shared_ptr<Channel_cfg_Traffic_Flow_Detection> channel_cfg = std::dynamic_pointer_cast<Channel_cfg_Traffic_Flow_Detection>(this->get_channel_cfg(channel_name));
    std::shared_ptr<Channel_data_Traffic_Flow_Detection> channel_data = std::dynamic_pointer_cast<Channel_data_Traffic_Flow_Detection>(this->get_channal_data(channel_name));


    int width=0;
    int height=0;
    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        input_image=input["image"]->data.image;
    }
    if(input_image!=nullptr){
        width=input_image->get_width();
        height=input_image->get_height();
    }
    else{
            throw Alg_Module_Exception("Error:\t image type error",this->node_name,Alg_Module_Exception::Stage::inference);

    }

    if (this->debug == 1 || this->debug == 2) {
        time_t tt = time(NULL);
        tm* t = localtime(&tt);

        clock_t now_time = clock();

        std::cout << this->get_module_name() << " " << channel_name << " debug " << std::endl;
        // std::cout << "/data/storage/" + this->get_module_name() + "/" + channel_name[channel_name.size()-1] + "_" + std::to_string(double(now_time)/CLOCKS_PER_SEC) + ".png" << std::endl;
        std::string image_path = 
            "/data/storage/" + this->get_module_name() + "/input_image/" +
            channel_name[channel_name.size()-1] + "_" + 
            std::to_string(t->tm_year+1900) + "_" + 
            std::to_string(t->tm_mon+1) + "_" + 
            std::to_string(t->tm_mday) + "_" + 
            std::to_string(t->tm_hour) + "_" + 
            std::to_string(t->tm_min) + "_" + 
            std::to_string(t->tm_sec) + "_" + 
            std::to_string(double(now_time)/CLOCKS_PER_SEC) + ".png";

        cv::Mat img = cv::Mat(height, width, CV_8UC3);
        if (input["image"]->data_type==InputOutput::Type::Image_t){
            img=input["image"]->data.image->get_image();
        }
        else{   
            throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);
        }
        cv::imwrite(image_path, img);
        
        // cv::imwrite("requirement/results/event/" + channel_name + "_" + std::to_string(double(now_time)/CLOCKS_PER_SEC) + ".png", img);
        
        //创建保存事件发生时截图的图片
        // std::string cmd = "mkdir " + debug_path + "/" + channel_name + "_" + std::to_string(double(now_time)/CLOCKS_PER_SEC);
        // system(cmd.c_str());
    }

    // 更新通道数据
    if (channel_data->frame_width == 0 || channel_data->frame_height == 0) {
 
        //获取通道的宽度和高度, 便于后续进一步映射
        channel_data->frame_width = width;
        channel_data->frame_height = height;

        //检测线, 用于检测压线事件, 转到 cv::Point 方便调用
        std::vector<std::vector<std::pair<float,float>>> boundary = channel_cfg->get_boundary_self("车流检测_车道横截面线");
        channel_data->boundary_detection_line.resize(boundary.size());
        for (int i = 0; i < boundary.size(); ++i) {
            channel_data->boundary_detection_line[i].resize(boundary[i].size());
            for (int j = 0; j < boundary[i].size(); ++j) {
                channel_data->boundary_detection_line[i][j].x = int(boundary[i][j].first * channel_data->frame_width);
                channel_data->boundary_detection_line[i][j].y = int(boundary[i][j].second * channel_data->frame_height);
            }
        }
        channel_data->trajectory_tracker->init_area(channel_data->frame_height, channel_data->frame_width, channel_data->boundary_detection_line);

        //检测车道
        std::vector<std::vector<std::pair<float,float>>> boundary_lane = channel_cfg->get_boundary_self("车流检测_车道");
        channel_data->boundary_lane.resize(boundary_lane.size());
        for (int i = 0; i < boundary_lane.size(); ++i) {
            channel_data->boundary_lane[i].resize(boundary_lane[i].size());
            for (int j = 0; j < boundary_lane[i].size(); ++j) {
                channel_data->boundary_lane[i][j].x = int(boundary_lane[i][j].first * channel_data->frame_width);
                channel_data->boundary_lane[i][j].y = int(boundary_lane[i][j].second * channel_data->frame_height);
            }
        }
        channel_data->lane_area = cv::Mat::zeros(channel_data->frame_height, channel_data->frame_width, CV_8UC1);
        for (int i = 0; i < channel_data->boundary_lane.size(); ++i) {
            std::vector<std::vector<cv::Point>> tmp;
            tmp.push_back(channel_data->boundary_lane[i]);
            cv::fillPoly(channel_data->lane_area, tmp, cv::Scalar(i+1));
        }
    }
    
    //获取跟踪的结果
    auto &vehicles = input["vehicle"]->data.detect;
    std::vector<Result_item_Traffic_Flow_Detection> detections;
    for (int i = 0; i < vehicles.size(); ++i) {
        if (std::isnan(vehicles[i].x1) || vehicles[i].x1 < 0 || vehicles[i].x1 > width) {
            continue;
        }
        if (std::isnan(vehicles[i].y1) || vehicles[i].y1 < 0 || vehicles[i].y1 > height) {
            continue;
        }
        if (std::isnan(vehicles[i].x2) || vehicles[i].x2 < 0 || vehicles[i].x2 > width) {
            continue;
        }
        if (std::isnan(vehicles[i].y2) || vehicles[i].y2 < 0 || vehicles[i].y2 > height) {
            continue;
        }

        Result_item_Traffic_Flow_Detection tmp;
        tmp.x1 = vehicles[i].x1;
        tmp.y1 = vehicles[i].y1;
        tmp.x2 = vehicles[i].x2;
        tmp.y2 = vehicles[i].y2;
        tmp.temp_idx = vehicles[i].temp_idx;
        detections.push_back(tmp);
    }

    //过滤掉非必要车辆
    channel_data->filter_invalid_objects(detections);

    int need_vehicle_attr=1;
    module_cfg->get_int("need_vehicle_attr",need_vehicle_attr);
    // 检测车辆的信息
    if(need_vehicle_attr>0&& detections.size()>0){
        if (this->debug == 1 || this->debug == 4) {
            std::cout << this->get_module_name() << " " << "need predict car attribute"<< std::endl;
        
        }
        this->detect_vehicle_attr(channel_name, input, detections);


    }

    //推理完成后，自己创建的bm_image需要自己销毁

    //整理输出数据
    auto forward_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    auto &forward_results = forward_output->data.detect;
    forward_results.resize(detections.size());
    for (int i = 0; i < detections.size(); i++) {
        forward_results[i].x1 = detections[i].x1;
        forward_results[i].y1 = detections[i].y1;
        forward_results[i].x2 = detections[i].x2;
        forward_results[i].y2 = detections[i].y2;
        forward_results[i].temp_idx = detections[i].temp_idx;
        //车辆颜色
        Ext_Result ext_result_color;
        ext_result_color.score = detections[i].color_score;
        ext_result_color.class_id = detections[i].color;
        ext_result_color.tag = attr_label_to_str(detections[i].color);
        forward_results[i].ext_result["color"] = ext_result_color;
        //车辆朝向
        Ext_Result ext_result_face_direction;
        ext_result_face_direction.score = detections[i].face_direction_score;
        ext_result_face_direction.class_id = detections[i].face_direction;
        ext_result_face_direction.tag = attr_label_to_str(detections[i].face_direction);
        forward_results[i].ext_result["face_direction"] = ext_result_face_direction;
        //车辆类型
        Ext_Result ext_result_type;
        ext_result_type.score = detections[i].type_score;
        ext_result_type.class_id = detections[i].type;
        ext_result_type.tag = attr_label_to_str(detections[i].type);
        forward_results[i].ext_result["type"] = ext_result_type;
    }
    output.clear();
    output["license"] = input["license"];
    output["vehicle"] = forward_output;

    if (this->debug == 1 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Forward Finish: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    return true;
};

/// @brief 过滤
/// @param channel_name 通道名称
/// @param input['vehicle'] 车辆目标
/// @param input['license'] 车牌目标
/// @param output['result'] 车辆事件结果
/// @return 
bool Alg_Module_Traffic_Flow_Detection::filter(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &output)
{
    if (this->debug == 1 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Filter Begin: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    // std::cout << "filter" << std::endl;
    //检查是否包含需要的数据
    if (input.find("vehicle") == input.end()) {
        throw Alg_Module_Exception("Error:\t no vehicle in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }
    if (input.find("license") == input.end()) {
        throw Alg_Module_Exception("Error:\t no license in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }

    std::shared_ptr<Module_cfg_Traffic_Flow_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Traffic_Flow_Detection>(this->get_module_cfg());
    std::shared_ptr<Channel_cfg_Traffic_Flow_Detection> channel_cfg = std::dynamic_pointer_cast<Channel_cfg_Traffic_Flow_Detection>(this->get_channel_cfg(channel_name));
    std::shared_ptr<Channel_data_Traffic_Flow_Detection> channel_data = std::dynamic_pointer_cast<Channel_data_Traffic_Flow_Detection>(this->get_channal_data(channel_name));

    // 未过滤的检测结果
    auto &vehicles = input["vehicle"]->data.detect;
    auto &licenses = input["license"]->data.detect_license;

    // 转到内部处理方法
    std::vector<Result_item_Traffic_Flow_Detection> detections;
    detections.resize(vehicles.size());
    for (int i = 0; i < vehicles.size(); ++i) {
        detections[i].temp_idx = vehicles[i].temp_idx;
        detections[i].x1       = vehicles[i].x1;
        detections[i].y1       = vehicles[i].y1;
        detections[i].x2       = vehicles[i].x2;
        detections[i].y2       = vehicles[i].y2;
        detections[i].color = vehicles[i].ext_result["color"].class_id;
        detections[i].color_score = vehicles[i].ext_result["color"].score;
        detections[i].face_direction = vehicles[i].ext_result["face_direction"].class_id;
        detections[i].face_direction_score = vehicles[i].ext_result["face_direction"].score;
        detections[i].type = vehicles[i].ext_result["type"].class_id;
        detections[i].type_score = vehicles[i].ext_result["type"].score;
        for (auto &license : licenses) {
            if (license.car_idx == vehicles[i].temp_idx) {
                detections[i].license = license.license;
                detections[i].license_score = license.score;
                break;
            }
        }
    }

    std::vector<Result_item_Traffic_Flow_Detection> events;
    std::vector<Centroid_Entity> entitys;
    entitys.resize(detections.size());
    for (int i = 0; i < detections.size(); ++i) {                               // 转到轨迹跟踪时的处理方法
        entitys[i].track_id = detections[i].temp_idx;
        entitys[i].point = cv::Point((detections[i].x1+detections[i].x2)/2, (detections[i].y1+detections[i].y2)/2);
        entitys[i].color = detections[i].color;
        entitys[i].face_direction = detections[i].face_direction;
        entitys[i].type = detections[i].type;
        entitys[i].color_score = detections[i].color_score;
        entitys[i].face_direction_score = detections[i].face_direction_score;
        entitys[i].type_score = detections[i].type_score;
        entitys[i].license = detections[i].license;
        entitys[i].license_score = detections[i].license_score;
    }

    channel_data->trajectory_tracker->update_trajectory(entitys);               // 轨迹跟踪
    for (int i = 0; i < detections.size(); ++i) {                               // 通过跟踪目标的所处车道变化来判断是否发生变道
        Trajectory_Result event_result = channel_data->trajectory_tracker->get_result(
            detections[i].temp_idx, 
            detections[i].x1, 
            detections[i].y1, 
            detections[i].x2, 
            detections[i].y2);
            
        if (event_result.score > 0.5) {  
            detections[i].tag = "车流";
            detections[i].score = event_result.score;
            detections[i].region_id = event_result.region_id;
            detections[i].dy = event_result.dy; 
            detections[i].lane_id = int(channel_data->lane_area.at<u_char>((detections[i].y1+detections[i].y2)/2, (detections[i].x1+detections[i].x2)/2)) - 1;

            if (detections[i].lane_id < 0) continue;
            decode_attr(event_result.labels_socre, detections[i]);
            detections[i].license = event_result.license;
            detections[i].license_score = event_result.license_score;
            events.push_back(detections[i]);
            continue;;
        }
    }
    
    std::vector<Result_item_Traffic_Flow_Detection>::iterator event = events.begin();
    // channel_data->duplicate_remover->process(events);                           // 过滤掉重复事件
    for (; event != events.end(); )
    {
        if (channel_data->duplicate_remover->process(event->temp_idx)) {
            event++;
        } else {
            event = events.erase(event);
        }
    }
    channel_data->duplicate_remover->update();

    // 整理检测数据
    auto filter_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_license_t); 
    auto &filter_results = filter_output->data.detect_license;
    for (auto &event : events)
    {
        Result_Detect_license temp_result;
        temp_result.x1 = event.x1;
        temp_result.y1 = event.y1;
        temp_result.x2 = event.x2;
        temp_result.y2 = event.y2;
        temp_result.tag = event.tag;
        temp_result.car_idx = event.temp_idx;
        temp_result.temp_idx = event.temp_idx;
        temp_result.region_idx = event.region_id;

        Ext_Result ext_result_color;//车辆颜色
        Ext_Result ext_result_face_direction;//车辆朝向
        Ext_Result ext_result_type;//车辆类型

        ext_result_color.score = event.color_score;
        ext_result_face_direction.score = event.face_direction_score;
        ext_result_type.score = event.type_score;

        ext_result_color.class_id = event.color;
        ext_result_face_direction.class_id = event.face_direction;
        ext_result_type.class_id = event.type;

        ext_result_color.tag = attr_label_to_str(event.color);
        ext_result_face_direction.tag = attr_label_to_str(event.face_direction);
        ext_result_type.tag = attr_label_to_str(event.type);
        if (ext_result_face_direction.tag == "unknown") {

        }
        else if (ext_result_face_direction.tag == "back_camera") {
            ext_result_face_direction.tag = "far";
        } else {
            ext_result_face_direction.tag = "near";
        }

        temp_result.ext_result["vehicle_color"] = ext_result_color;
        temp_result.ext_result["vehicle_head_dir"] = ext_result_face_direction;
        temp_result.ext_result["vehicle_type"] = ext_result_type;

        Ext_Result ext_result_travel_dir;
        if (event.dy > 0) {
            ext_result_travel_dir.class_id = 0;
            ext_result_travel_dir.score = 1;
            ext_result_travel_dir.tag = "near";
        } else {
            ext_result_travel_dir.class_id = 1;
            ext_result_travel_dir.score = 1;
            ext_result_travel_dir.tag = "far";
        }
        temp_result.ext_result["travel_dir"] = ext_result_travel_dir;

        Ext_Result ext_result_lane_no;
        ext_result_lane_no.class_id = event.lane_id;
        ext_result_lane_no.score = 1;
        ext_result_lane_no.tag = std::to_string(event.lane_id);
        temp_result.ext_result["lane_no"] = ext_result_lane_no;
        
        temp_result.landms_x1 = temp_result.x1;
        temp_result.landms_x2 = temp_result.x1 + 10;
        temp_result.landms_x3 = temp_result.x1 + 10;
        temp_result.landms_x4 = temp_result.x1;
        temp_result.landms_y1 = temp_result.y1;
        temp_result.landms_y2 = temp_result.y1;
        temp_result.landms_y3 = temp_result.y1 + 10;
        temp_result.landms_y4 = temp_result.y1 + 10;
        temp_result.license = temp_result.license;
        temp_result.state = Result_Detect_license::Detect_state::SMALL_REGION;

        temp_result.score = event.face_direction_score;

        // 根据车辆id 获取车牌
        for (auto &license : licenses) {
            if (license.car_idx == event.temp_idx) {
                temp_result.landms_x1     = license.landms_x1+temp_result.x1;
                temp_result.landms_x2     = license.landms_x2+temp_result.x1;
                temp_result.landms_x3     = license.landms_x3+temp_result.x1;
                temp_result.landms_x4     = license.landms_x4+temp_result.x1;
                temp_result.landms_y1     = license.landms_y1+temp_result.y1;
                temp_result.landms_y2     = license.landms_y2+temp_result.y1;
                temp_result.landms_y3     = license.landms_y3+temp_result.y1;
                temp_result.landms_y4     = license.landms_y4+temp_result.y1;
                temp_result.license_type  = license.license_type;
                temp_result.license_color = license.license_color;
                temp_result.license       = license.license;
                temp_result.state         = license.state;
                break;
            }
        }

        filter_results.push_back(temp_result);
    }

    output.clear();
    output["result"] = filter_output;

    if (this->debug == 1 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Filter Finish: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    return true;
};

/// @brief 可视化
/// @param channel_name 通道名称
/// @param input['image'] 输入图片
/// @param filter_output['result'] 车辆事件结果
/// @return 
bool Alg_Module_Traffic_Flow_Detection::display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &filter_output)
{   
    if (this->debug == 1 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Display Begin: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    if (this->debug == 1 || this->debug == 3) {
        //检查是否包含需要的数据
        if (input.find("image") == input.end()) {
            throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::display);
            return false;
        }
        if (filter_output.find("result") == filter_output.end()) {
            throw Alg_Module_Exception("Error:\t no result in filter_output",this->node_name,Alg_Module_Exception::Stage::display);
            return false;
        }

        std::shared_ptr<Module_cfg_Traffic_Flow_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Traffic_Flow_Detection>(this->get_module_cfg());    
        std::shared_ptr<Channel_cfg_Traffic_Flow_Detection> channel_cfg = std::dynamic_pointer_cast<Channel_cfg_Traffic_Flow_Detection>(this->get_channel_cfg(channel_name));
        std::shared_ptr<Channel_data_Traffic_Flow_Detection> channel_data = std::dynamic_pointer_cast<Channel_data_Traffic_Flow_Detection>(this->get_channal_data(channel_name));

        //加载模块需要的参数
        bool load_res = true;
        int box_color_blue, box_color_green, box_color_red, box_thickness;
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
            throw Alg_Module_Exception("Error:\t input type error in display",this->node_name,Alg_Module_Exception::Stage::display);
            return false;
        }

        //特殊车辆事件
        auto &results = filter_output["result"]->data.detect_license;
        for (auto &result : results)
        {
            cv::Mat image_copy = image.clone();
            int x = result.x1;
            int y = result.y1;
            int w = result.x2 - result.x1;
            int h = result.y2 - result.y1;

            //画出目标框
            cv::rectangle(image_copy, cv::Rect(x, y, w, h), cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness);

            //画出检测线
            cv::line(image_copy, channel_data->boundary_detection_line[result.region_idx][0], channel_data->boundary_detection_line[result.region_idx][1], cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness);

            //画出轨迹线
            channel_data->trajectory_tracker->display_trajectory(result.temp_idx, image_copy);

            //画出车牌
            std::vector<cv::Point> pts;
            pts.push_back(cv::Point(result.landms_x1, result.landms_y1));
            pts.push_back(cv::Point(result.landms_x2, result.landms_y2));
            pts.push_back(cv::Point(result.landms_x3, result.landms_y3));
            pts.push_back(cv::Point(result.landms_x4, result.landms_y4));
            cv::polylines(image_copy, pts, true, cv::Scalar(box_color_blue, box_color_green, box_color_red), 1);

            //画出所在车道
            if (result.ext_result["lane_no"].class_id >= 0) {
                cv::polylines(image_copy, channel_data->boundary_lane[result.ext_result["lane_no"].class_id], true, cv::Scalar(box_color_blue, box_color_green, box_color_red), box_thickness, 8, 0);
            }
            
            std::string text = std::to_string(result.temp_idx) + " ";
            if (result.tag == "车流") text += "count ";

            //车辆车牌
            text += result.license + " ";

            cv::putText(image_copy, text, cv::Point(x, y), 1, 1.5, cv::Scalar(0, 0, 0), 2, cv::LINE_8);

            result.res_images.insert({"event_image", image_copy});

            if (this->debug == 1 || this->debug == 3) {
                time_t tt = time(NULL);
                tm* t = localtime(&tt);

                clock_t now_time = clock();

                std::string image_path = 
                    "/data/storage/" + this->get_module_name() + "/event_image/" + 
                    channel_name[channel_name.size()-1] + "_" + 
                    std::to_string(t->tm_year+1900) + "_" + 
                    std::to_string(t->tm_mon+1) + "_" + 
                    std::to_string(t->tm_mday) + "_" + 
                    std::to_string(t->tm_hour) + "_" + 
                    std::to_string(t->tm_min) + "_" + 
                    std::to_string(t->tm_sec) + "_" + 
                    std::to_string(result.car_idx) + "_" + 
                    std::to_string(double(now_time)/CLOCKS_PER_SEC) + 
                    ".png";

                cv::imwrite(image_path, image_copy);
                
                // cv::imwrite("requirement/results/event/" + channel_name + "_" + std::to_string(double(now_time)/CLOCKS_PER_SEC) + ".png", img);
                
                //创建保存事件发生时截图的图片
                // std::string cmd = "mkdir " + debug_path + "/" + channel_name + "_" + std::to_string(double(now_time)/CLOCKS_PER_SEC);
                // system(cmd.c_str());
            }
        }
    }

    if (this->debug == 1 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Display Finish: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }
    
    return true;
};

std::shared_ptr<Module_cfg_base> Alg_Module_Traffic_Flow_Detection::load_module_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模块配置文件的加载
    auto res = std::make_shared<Module_cfg_Traffic_Flow_Detection>(this->node_name);
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Model_cfg_base> Alg_Module_Traffic_Flow_Detection::load_model_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模型配置文件的加载
    auto res = std::make_shared<Model_cfg_Traffic_Flow_Detection>();
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Channel_cfg_base> Alg_Module_Traffic_Flow_Detection::load_channel_cfg_(std::string channel_name, std::string cfg_path)
{
    auto res = std::make_shared<Channel_cfg_Traffic_Flow_Detection>(channel_name);

    // 通道中增加实线的配置方法 
    if (access(cfg_path.c_str(), F_OK) != 0) {
        //文件不存在
        throw Alg_Module_Exception("Error:\t channel cfg " + cfg_path + " is not exist", this->node_name, Alg_Module_Exception::Stage::load_channel);
    }
    res->from_file(cfg_path);
    
    return res;
};
std::shared_ptr<Channel_data_base> Alg_Module_Traffic_Flow_Detection::init_channal_data_(std::string channel_name)
{
    auto res = std::make_shared<Channel_data_Traffic_Flow_Detection>(channel_name);

    std::shared_ptr<Channel_cfg_Traffic_Flow_Detection> channel_cfg = std::dynamic_pointer_cast<Channel_cfg_Traffic_Flow_Detection>(this->get_channel_cfg(channel_name));
    std::shared_ptr<Module_cfg_Traffic_Flow_Detection> module_cfg = std::dynamic_pointer_cast<Module_cfg_Traffic_Flow_Detection>(this->get_module_cfg());
    
    bool load_res = true;
    int max_record_time_for_trajectory;
    int max_wait_time;
    int min_point_for_direction;
    int max_record_time_for_event;

    load_res &= module_cfg->get_int("max_record_time_for_trajectory", max_record_time_for_trajectory);
    load_res &= module_cfg->get_int("max_wait_time", max_wait_time);
    load_res &= module_cfg->get_int("min_point_for_direction", min_point_for_direction);
    load_res &= module_cfg->get_int("max_record_time_for_event", max_record_time_for_event);

    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load channel param failed",this->node_name,Alg_Module_Exception::Stage::load_channel);
    }

    res->trajectory_tracker = new Trajectory_Tracker();
    res->trajectory_tracker->set_max_record_time(max_record_time_for_trajectory);
    res->trajectory_tracker->set_max_wait_time(max_wait_time);
    res->trajectory_tracker->set_min_point_for_direction(min_point_for_direction);

    res->duplicate_remover = new Duplicate_Remover();
    res->duplicate_remover->set_max_record_time(max_record_time_for_event);
    
    return res;
};

extern "C" Alg_Module_Base *create()
{
    return new Alg_Module_Traffic_Flow_Detection();                    
};
extern "C" void destory(Alg_Module_Base *p)
{
    delete p;
};
