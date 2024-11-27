#include "alg_module_detect_tracking.h"
#include <iostream>

void roi_pooling(Output &net_output_feature,std::vector<Result_item_Detect_Tracking> &output,int img_h,int img_w)
{
    float *features=(float*)net_output_feature.data.data();
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
float iou(Result_item_Detect_Tracking &box1, Result_item_Detect_Tracking &box2)
{
    float x1 = std::max(box1.x1, box2.x1);  //left
    float y1 = std::max(box1.y1, box2.y1);  //top
    float x2 = std::min(box1.x2, box2.x2);  //right
    float y2 = std::min(box1.y2, box2.y2);  //bottom

    if (x1 >= x2 || y1 >= y2) return 0;

    float over_area = (x2 - x1) * (y2 - y1); //计算重复区域

    float box1_w = box1.x2 - box1.x1;
    float box1_h = box1.y2 - box1.y1;
    float box2_w = box2.x2 - box2.x1;
    float box2_h = box2.y2 - box2.y1;

    float iou = over_area / (box1_w * box1_h + box2_w * box2_h - over_area);

    return iou;
};
float intersection_over_one_box(Result_item_Detect_Tracking &box1, Result_item_Detect_Tracking &box2)
{   // 计算box1和box2相交面积在box1面积中的比值
    float x1 = std::max(box1.x1, box2.x1);  //left
    float y1 = std::max(box1.y1, box2.y1);  //top
    float x2 = std::min(box1.x2, box2.x2);  //right
    float y2 = std::min(box1.y2, box2.y2);  //bottom

    if (x1 >= x2 || y1 >= y2) return 0;

    float over_area = (x2 - x1) * (y2 - y1); //计算重复区域

    float box1_w = box1.x2 - box1.x1;
    float box1_h = box1.y2 - box1.y1;

    float over_rat = over_area / (box1_w * box1_h);

    return over_rat;
};
bool sort_score(Result_item_Detect_Tracking &box1, Result_item_Detect_Tracking &box2)
{
    return (box1.score > box2.score);


};
inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23)  *(1.4426950409  *x + 126.93490512f);
    return v.f;
};
inline float sigmoid(float x)
{
    if (x >= 87) return 1;
    if (x <= -87) return 0;
    return 1.0f / (1.0f + fast_exp(-x));
    // return 1.0f / (1.0f + exp(-x));
};
void nms_yolo(Output &net_output, std::vector<Result_item_Detect_Tracking> &output, std::vector<int> class_filter, float threshold_score=0.25, float threshold_iou=0.45)
{
    float *input=(float*)net_output.data.data();
    int dim1=net_output.shape[1];
    int dim2=net_output.shape[2];
    std::vector<Result_item_Detect_Tracking> result;
    float threshold_score_stage_1=threshold_score*0.77;
    for (int k = 0,i=0;k < dim1;k ++,i+=dim2) {
        float obj_conf = input[i + 9];
        obj_conf=sigmoid(obj_conf);
        if (obj_conf > threshold_score_stage_1) {
            Result_item_Detect_Tracking item;
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
                // item.x1 += item.class_id  *4096;
                // item.x2 += item.class_id  *4096;
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
        
        for (int i = 1; i < result.size(); ++i)
        {
            float iou_value = iou(result[0], result[i]);

            if (iou_value > threshold_iou)
            {
                result.erase(result.begin() + i);
                i--;
            }
        }
        result.erase(result.begin());
    }

    std::vector<Result_item_Detect_Tracking>::iterator iter=output.begin();
    // for(;iter!=output.end();iter++){
    //     iter->x1 -= iter->class_id*4096;
    //     iter->x2 -= iter->class_id*4096;
    // }

    return ;
};

Channel_cfg_Detect_Tracking::Channel_cfg_Detect_Tracking(std::string channel_name):Channel_cfg_base(channel_name)
{
    this->channel_name = channel_name;
};
Channel_cfg_Detect_Tracking::~Channel_cfg_Detect_Tracking() 
{

};

Channel_data_Detect_Trakcing::Channel_data_Detect_Trakcing(std::string channel_name):Channel_data_base(channel_name)
{
    this->channel_name = channel_name;
};
Channel_data_Detect_Trakcing::~Channel_data_Detect_Trakcing() 
{
    delete this->mytracker;
};

/// @brief deepsort 目标追踪
/// @param results 需要进行过滤的结果, 过滤后的结果也存在里面
void Channel_data_Detect_Trakcing::test_deepsort(std::vector<Result_item_Detect_Tracking> &results)
{   
    // 将原先的检测结果转到目标类型
    DETECTIONS detections;
    for (int i = 0; i < results.size(); i++)
    {
        DETECTION_ROW tmpRow;
        // tmpRow.tlwh = DETECTBOX(results[i].x1, results[i].y1, results[i].x2 - results[i].x1, results[i].y2 - results[i].y1);
        tmpRow.tlwh = DETECTBOX(results[i].x1/this->frame_width, results[i].y1/this->frame_height, (results[i].x2 - results[i].x1)/this->frame_width, (results[i].y2 - results[i].y1)/this->frame_height);
        tmpRow.confidence = results[i].score;
        detections.push_back(tmpRow);
    }

    // DEEPSORT 目标跟踪
    this->mytracker->predict();

    this->mytracker->update(detections);

    // 结果过滤
    std::vector<RESULT_DATA> filter_result;
    for (Track &track : this->mytracker->tracks) 
    {
        if (!track.is_confirmed() || track.time_since_update > 1) continue; // 未确认或者自上次更新以来的
        filter_result.push_back(std::make_pair(track.track_id, track.to_tlwh()));  // 时间超过 1 个时间步长,则跳过
    }

    if (results.size() == 0) return;

    // 结果整理
    results.clear();
    results.resize(filter_result.size());
    for (int i = 0; i < filter_result.size(); i++)
    {
        results[i].temp_idx = filter_result[i].first;
        results[i].x1       = filter_result[i].second(0) * this->frame_width;
        results[i].y1       = filter_result[i].second(1) * this->frame_height; 
        results[i].x2       = (filter_result[i].second(0) + filter_result[i].second(2)) * this->frame_width;
        results[i].y2       = (filter_result[i].second(1) + filter_result[i].second(3)) * this->frame_height;
    }
};

Alg_Module_Detect_Tracking::Alg_Module_Detect_Tracking():Alg_Module_Base_private("detect_tracking")
{    //参数是模块名，使用默认模块名初始化

};
Alg_Module_Detect_Tracking::~Alg_Module_Detect_Tracking()
{

};

/// @brief 车辆目标检测
/// @param input_image 输入图像
/// @param handle 设备句柄
/// @param result 检测结果
bool Alg_Module_Detect_Tracking::detect_vehicle_object(std::map<std::string,std::shared_ptr<InputOutput>> &input, std::vector<Result_item_Detect_Tracking> &result)
{

    std::shared_ptr<Device_Handle> handle;
    std::shared_ptr<QyImage> input_image;
    if(input["image"]->data_type==InputOutput::Type::Image_t){
        handle=input["image"]->data.image->get_handle();
        input_image=input["image"]->data.image;
        if(input_image==nullptr){
            throw Alg_Module_Exception("Error:\t image type error",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        }
    }
    else
    {
        throw Alg_Module_Exception("Error:\t image type not supported",this->node_name,Alg_Module_Exception::Stage::inference);              //获取模型推理实例异常，一般是因为模型实例还未创建
        return false;

    }
    
    //获取指定的模型实例
    auto net = this->get_model_instance(this->model_name);
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

    std::shared_ptr<Model_cfg_Detect_Tracking> model_cfg = std::dynamic_pointer_cast<Model_cfg_Detect_Tracking>(this->get_model_cfg(this->model_name)); 
    std::shared_ptr<Module_cfg_Detect_Tracking> module_cfg = std::dynamic_pointer_cast<Module_cfg_Detect_Tracking>(this->get_module_cfg());

    //加载参数
    std::vector<int> classes;
    float thresh_iou;
    float thresh_score;
    bool load_res = true;
    load_res &= module_cfg->get_int_vector("classes", classes);
    load_res &= module_cfg->get_float("thresh_score", thresh_score);
    load_res &= module_cfg->get_float("thresh_iou", thresh_iou);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load param failed",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }
    auto input_shape_ = input_shapes[0];

    int width=input_image->get_width();
    int height=input_image->get_height();
    float factor1 = input_shape_.dims[3] * 1.0 / width;
    float factor2 = input_shape_.dims[2] * 1.0 / height;
    float factor = factor1 > factor2 ? factor2 : factor1;   //选择较小的比例

    std::shared_ptr<QyImage> sub_image=input_image->resize_keep_ratio(input_shape_.dims[3],input_shape_.dims[2],0);
    std::vector<std::shared_ptr<QyImage>> net_input;
    net_input.push_back(sub_image);

    std::vector<Output> net_output;                                       
    net->forward(net_input, net_output);                                    //这一部分与原版本不一样


    std::vector<Result_item_Detect_Tracking> detections;
    nms_yolo(net_output[0], detections, classes, thresh_score, thresh_iou);
    for (auto iter = detections.begin(); iter != detections.end(); iter++)
    {
        iter->x1 = (iter->x1 + 0.5) / factor;
        iter->y1 = (iter->y1 + 0.5) / factor;
        iter->x2 = (iter->x2 + 0.5) / factor;
        iter->y2 = (iter->y2 + 0.5) / factor;
    }
    roi_pooling(net_output[1], detections, input_shape_.dims[2] / factor, input_shape_.dims[3] / factor);

    //整理数据
    for (int i = 0; i < detections.size(); i++)
    {   
        if (detections[i].x1 < 0) detections[i].x1 = 0;
        if (detections[i].y1 < 0) detections[i].y1 = 0;

        if (detections[i].x2 < 0) detections[i].x2 = 0;
        if (detections[i].y2 < 0) detections[i].y2 = 0;

        if (detections[i].x1 > width) detections[i].x1 = width;
        if (detections[i].x2 > width) detections[i].x2 = width;
        if (detections[i].y1 > height) detections[i].y1 = height;
        if (detections[i].y2 > height) detections[i].y2 = height;

        Result_item_Detect_Tracking res;
        res.x1 = detections[i].x1 / width;
        res.y1 = detections[i].y1 / height;
        res.x2 = detections[i].x2 / width;
        res.y2 = detections[i].y2 / height;
        result.push_back(res);
    }

    return true;
};

/// @brief 过滤车辆目标
/// @param channel_name 通道名称
/// @param detections 检测结果
bool Alg_Module_Detect_Tracking::filter_vehicle_object(std::string channel_name, std::vector<Result_item_Detect_Tracking> &detections)
{
    std::shared_ptr<Module_cfg_Detect_Tracking> module_cfg = std::dynamic_pointer_cast<Module_cfg_Detect_Tracking>(this->get_module_cfg());
    std::shared_ptr<Channel_data_Detect_Trakcing> channel_data = std::dynamic_pointer_cast<Channel_data_Detect_Trakcing>(this->get_channal_data(channel_name));

    //加载参数
    float thresh_min_scale_for_box;
    float thresh_max_scale_for_box;
    bool load_res = true;
    load_res &= module_cfg->get_float("thresh_min_scale_for_box", thresh_min_scale_for_box);
    load_res &= module_cfg->get_float("thresh_max_scale_for_box", thresh_max_scale_for_box);
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load param failed",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }

    //对检测结果进行过滤
    float frame_w = channel_data->frame_width;
    float frame_h = channel_data->frame_height;
    std::sort(detections.begin(), detections.end(), sort_score);
    for (int i = 0; i < detections.size(); ++i) {
        // 将超出画幅的结果剪切到当前画幅内
        if (detections[i].x1 < 0) detections[i].x1 = 0;
        if (detections[i].y1 < 0) detections[i].y1 = 0;
        if (detections[i].x2 >= 1) detections[i].x2 = 1;
        if (detections[i].y2 >= 1) detections[i].y2 = 1;

        // 过滤目标: 目标框的宽度或高度超过或小于画面一定比例
        float w = detections[i].x2 - detections[i].x1;
        float h = detections[i].y2 - detections[i].y1;
        if (w < thresh_min_scale_for_box || w > thresh_max_scale_for_box) detections[i].need_remove = true;
        if (h < thresh_min_scale_for_box || h > thresh_max_scale_for_box) detections[i].need_remove = true;

        // 过滤 nan 结果
        if (std::isnan(detections[i].x1)) detections[i].need_remove = true;
        if (std::isnan(detections[i].y1)) detections[i].need_remove = true;
        if (std::isnan(detections[i].x2)) detections[i].need_remove = true;
        if (std::isnan(detections[i].y2)) detections[i].need_remove = true;
    }
    for (auto iter = detections.begin(); iter != detections.end(); ) {
        if (iter->need_remove == true) {
            iter = detections.erase(iter);
        } else {
            iter++;
        }
    }
    return true;
};

/// @brief 初始化
/// @param root_dir 
bool Alg_Module_Detect_Tracking::init_from_root_dir(std::string root_dir)
{
    bool load_res = true;

    //加载模块配置文件
    load_res = this->load_module_cfg(root_dir + "/cfgs/" + this->node_name + "/module_cfg.xml");
    if (!load_res) {
        throw Alg_Module_Exception("Error:\t load module cfg failed",this->node_name,Alg_Module_Exception::Stage::load_module);
        return false;
    }

    std::shared_ptr<Module_cfg_Detect_Tracking> module_cfg = std::dynamic_pointer_cast<Module_cfg_Detect_Tracking>(this->get_module_cfg());

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

    load_res = module_cfg->get_int("debug", this->debug);
    if (!load_res) {
        this->debug = 0;
    } 

    return true;
};

/// @brief 推理
/// @param channel_name         通道名称
/// @param input["image"]       输入图像
/// @param output["vehicle"]    车辆目标
bool Alg_Module_Detect_Tracking::forward(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output)
{   
    if (this->debug > 0) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Forward Begin: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    //检查是否包含需要的数据
    if (input.find("image") == input.end()) {
        throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::inference);
        return false;
    }
    
    std::shared_ptr<Module_cfg_Detect_Tracking> module_cfg = std::dynamic_pointer_cast<Module_cfg_Detect_Tracking>(this->get_module_cfg());
    std::shared_ptr<Channel_data_Detect_Trakcing> channel_data = std::dynamic_pointer_cast<Channel_data_Detect_Trakcing>(this->get_channal_data(channel_name));

    //获取计算卡推理核心的handle
    if (channel_data->frame_width==0 || channel_data->frame_height==0) {

        if(input["image"]->data_type==InputOutput::Type::Image_t){
            std::shared_ptr<QyImage>input_image= input["image"]->data.image;
            int width=input_image->get_width();
            int height=input_image->get_height();
            channel_data->frame_width = width;
            channel_data->frame_height = height;
        }
        else{
            throw Alg_Module_Exception("Warning:\t image input type error",this->node_name,Alg_Module_Exception::Stage::inference);    //当前值支持bm_image和opencv的Mat两种图像数据
            return false;
        }
    }

    std::vector<Result_item_Detect_Tracking> detections;
    this->detect_vehicle_object(input, detections);   //检测车辆结果

    // auto forward_output_1 = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    // auto &forward_results_1 = forward_output_1->data.detect;
    // forward_results_1.resize(detections.size());
    // for (int i = 0; i < detections.size(); i++)
    // {
    //     forward_results_1[i].temp_idx = detections[i].temp_idx;
    //     forward_results_1[i].x1       = detections[i].x1 * input_image.width;
    //     forward_results_1[i].y1       = detections[i].y1 * input_image.height;
    //     forward_results_1[i].x2       = detections[i].x2 * input_image.width;
    //     forward_results_1[i].y2       = detections[i].y2 * input_image.height;
    //     forward_results_1[i].score    = detections[i].score;
    // }
    
    this->filter_vehicle_object(channel_name, detections);          //过滤无效结果

    // auto forward_output_2 = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    // auto &forward_results_2 = forward_output_2->data.detect;
    // forward_results_2.resize(detections.size());
    // for (int i = 0; i < detections.size(); i++)
    // {
    //     forward_results_2[i].temp_idx = detections[i].temp_idx;
    //     forward_results_2[i].x1       = detections[i].x1 * input_image.width;
    //     forward_results_2[i].y1       = detections[i].y1 * input_image.height;
    //     forward_results_2[i].x2       = detections[i].x2 * input_image.width;
    //     forward_results_2[i].y2       = detections[i].y2 * input_image.height;
    //     forward_results_2[i].score    = detections[i].score;
    // }
    
    channel_data->test_deepsort(detections);                //目标跟踪
    
    this->filter_vehicle_object(channel_name, detections);  //过滤无效结果
    

    //整理输出数据
    auto forward_output = std::make_shared<InputOutput>(InputOutput::Type::Result_Detect_t);
    auto &forward_results = forward_output->data.detect;
    forward_results.resize(detections.size());
    for (int i = 0; i < detections.size(); i++) {
        forward_results[i].temp_idx = detections[i].temp_idx;
        forward_results[i].x1       = detections[i].x1 * channel_data->frame_width;
        forward_results[i].y1       = detections[i].y1 * channel_data->frame_height;
        forward_results[i].x2       = detections[i].x2 * channel_data->frame_width;
        forward_results[i].y2       = detections[i].y2 * channel_data->frame_height;
        forward_results[i].score    = detections[i].score;
    }
    
    output.clear();
    // output["origin1"] = forward_output_1;
    // output["origin2"] = forward_output_2;
    output["vehicle"] = forward_output;

    if (this->debug > 0 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Forward Finish: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }
    return true;
};

/// @brief 过滤
bool Alg_Module_Detect_Tracking::filter(std::string channel_name, std::map<std::string, std::shared_ptr<InputOutput>> &input, std::map<std::string, std::shared_ptr<InputOutput>> &output)
{
    if (this->debug > 0 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Filter Begin: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    //检查是否包含需要的数据
    if (input.find("vehicle") == input.end()) {
        throw Alg_Module_Exception("Error:\t no vehicle in input",this->node_name,Alg_Module_Exception::Stage::filter);
        return false;
    }

    output.clear();
    output["result"] = input["vehicle"];

    if (this->debug > 0 || this->debug == 4) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Filter Finish: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    return true;
};

/// @brief 可视化
bool Alg_Module_Detect_Tracking::display(std::string channel_name, std::map<std::string,std::shared_ptr<InputOutput>> &input, std::map<std::string,std::shared_ptr<InputOutput>> &filter_output)
{   
    if (this->debug > 0) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Display Begin: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }

    if (this->debug == 2) {
        //检查是否包含需要的数据
        if (input.find("image") == input.end()) {
            throw Alg_Module_Exception("Error:\t no image in input",this->node_name,Alg_Module_Exception::Stage::display);
            return false;
        }
        if (filter_output.find("result") == filter_output.end()) {
            throw Alg_Module_Exception("Error:\t no result in filter_output",this->node_name,Alg_Module_Exception::Stage::display);
            return false;
        }

        std::shared_ptr<Module_cfg_Detect_Tracking> module_cfg = std::dynamic_pointer_cast<Module_cfg_Detect_Tracking>(this->get_module_cfg());

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
            throw Alg_Module_Exception("Error:\t input type error in display",this->node_name,Alg_Module_Exception::Stage::display);
            return false;
        }

        //特殊车辆事件
        auto &results = filter_output["result"]->data.detect;
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
    }

    if (this->debug > 0) {
        clock_t now_time = clock();
        std::cout << this->get_module_name() << " " << channel_name << " Stage Display Finish: " << std::to_string(double(now_time)/CLOCKS_PER_SEC) << std::endl;
    }
    
    return true;
};

std::shared_ptr<Module_cfg_base> Alg_Module_Detect_Tracking::load_module_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模块配置文件的加载
    auto res = std::make_shared<Module_cfg_Detect_Tracking>(this->node_name);
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Model_cfg_base> Alg_Module_Detect_Tracking::load_model_cfg_(std::string cfg_path)
{
    //可以通过虚函数重载，实现派生的模型配置文件的加载
    auto res = std::make_shared<Model_cfg_Detect_Tracking>();
    res->from_file(cfg_path);
    return res;
};
std::shared_ptr<Channel_cfg_base> Alg_Module_Detect_Tracking::load_channel_cfg_(std::string channel_name, std::string cfg_path)
{
    auto res = std::make_shared<Channel_cfg_Detect_Tracking>(channel_name);

    if (access(cfg_path.c_str(), F_OK) != 0) {
        //文件不存在
        throw Alg_Module_Exception("Error:\t channel cfg " + cfg_path + " is not exist", this->node_name, Alg_Module_Exception::Stage::load_channel);
    }
    res->from_file(cfg_path);
    
    return res;
};
std::shared_ptr<Channel_data_base> Alg_Module_Detect_Tracking::init_channal_data_(std::string channel_name)
{
    auto res = std::make_shared<Channel_data_Detect_Trakcing>(channel_name);

    std::shared_ptr<Module_cfg_Detect_Tracking> module_cfg = std::dynamic_pointer_cast<Module_cfg_Detect_Tracking>(this->get_module_cfg());
    
    module_cfg->get_float("max_cosine_distance", res->max_cosine_distance);
    module_cfg->get_int("nn_budget", res->nn_budget);
    module_cfg->get_float("max_iou_distance", res->max_iou_distance);
    module_cfg->get_int("max_age", res->max_age);
    module_cfg->get_int("n_init", res->n_init);

    res->mytracker = new tracker(res->max_cosine_distance, res->nn_budget, res->max_iou_distance, res->max_age, res->n_init);

    return res;
};

extern "C" Alg_Module_Base *create() 
{
    return new Alg_Module_Detect_Tracking();
};
extern "C" void destory(Alg_Module_Base *p)
{
    delete p;
};
