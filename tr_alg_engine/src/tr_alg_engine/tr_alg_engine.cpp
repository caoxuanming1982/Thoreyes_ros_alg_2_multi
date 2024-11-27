#include "tr_alg_engine/tr_alg_engine.h"
#include <iostream>
#include <fstream>
#include <inout_type.h>
#define __DEBUG__

#define __WITH_TRY_CATCH__

Tr_Alg_Engine_module::Tr_Alg_Engine_module(std::string dev_platform_name) : Alg_Node_Base("alg_engine",dev_platform_name)
{
    publisher_primary = this->create_publisher<tr_interfaces::msg::TrImageAlgoResult>("alg_event_result", rclcpp::SystemDefaultsQoS());
    callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    recv_image_callback_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    jpeg_decoder=get_jpeg_decoder();    
    cv::setNumThreads(2);
    global_init();
};

bool Tr_Alg_Engine_module::init(std::string submodule_dir,std::string requirement_dir)
{
    bool res=init_(submodule_dir,requirement_dir);
    if(res==false)
        return false;

    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&Tr_Alg_Engine_module::tick, this));

    channel_list_subscription = this->create_subscription<tr_alg_interfaces::msg::ChannelList>("/channel_list_"+dev_platform_name, rclcpp::ClockQoS(), std::bind(&Tr_Alg_Engine_module::updata_channel, this, _1));
    device_usage_subscription = this->create_subscription<tr_alg_interfaces::msg::DeviceUsage>("/device_usage_"+dev_platform_name, rclcpp::ClockQoS(), std::bind(&Tr_Alg_Engine_module::update_device_handles, this, _1));
    this->set_module_enable_service=this->create_service<tr_alg_interfaces::srv::SetModuleEnable>("/alg_engine/set_module_enable",std::bind(&Tr_Alg_Engine_module::set_module_enable_callback, this, _1, _2));
    this->get_module_enable_service=this->create_service<tr_alg_interfaces::srv::GetModuleEnable>("/alg_engine/get_module_enable",std::bind(&Tr_Alg_Engine_module::get_module_enable_callback, this, _1, _2));
    this->set_channel_config_service=this->create_service<tr_alg_interfaces::srv::SetChannelConfig>("/alg_engine/set_channel_config",std::bind(&Tr_Alg_Engine_module::set_channel_config_callback, this, _1, _2));
    this->get_channel_status_service=this->create_service<tr_alg_interfaces::srv::GetChannelStatus>("/alg_engine/get_channel_status",std::bind(&Tr_Alg_Engine_module::get_channel_status, this, _1, _2));

    return true;
};

bool Tr_Alg_Engine_module::init_(std::string submodule_dir,std::string requirement_dir){
    engine.set_add_reduce_instance_thres(0.7,0.3);
    if (submodule_dir==""){
        submodule_dir="/data_temp/thoreyes/ros/alg_module_submodules/lib/";
    }
    if(requirement_dir==""){
        requirement_dir="/data_temp/thoreyes/ros/requirement/";
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


void Tr_Alg_Engine_module::update_device_handles(const tr_alg_interfaces::msg::DeviceUsage::ConstPtr &msg)
{
    std::vector<int> device_id(msg->tpu_id.begin(), msg->tpu_id.end());
    this->set_device_handles(device_id);
    jpeg_decoder->init(device_handles_inuse,4);
};

void Tr_Alg_Engine_module::tick()
{
    if (this->device_handles_all.size() <= 0)
        return;
    auto models_util = engine.update_and_check_model_util(this->show_util);
    std::vector<Request_Model_instance_data> models_util_res;
    for (int i = 0; i < models_util.size(); i++)
    {
        std::cout <<"******"<< models_util[i].module_name << " " << models_util[i].model_name << " need add instance" << std::endl;
        rclcpp::Client<tr_alg_interfaces::srv::GetAvaliableDevice>::SharedPtr get_device_client;

        get_device_client = this->create_client<tr_alg_interfaces::srv::GetAvaliableDevice>("/service/device_"+dev_platform_name+"/get_avaliable_device", rmw_qos_profile_services_default, callback_group);
        if (get_device_client->wait_for_service(std::chrono::milliseconds(100)) == false)
            return;

        auto request = std::make_shared<tr_alg_interfaces::srv::GetAvaliableDevice::Request>();
        request->cpu_load = models_util[i].cfg->cpu_util_require;
        request->gpu_load = models_util[i].cfg->tpu_util_require;
        request->memory = models_util[i].cfg->mem_require_mbyte;
        request->gpu_memory = models_util[i].cfg->tpu_mem_require_mbyte;
        request->force_get=models_util[i].has_ins==false;
        auto future = get_device_client->async_send_request(request);
        auto response = future.get();
//        std::cout<<"n device response :"<<response->device_id.size()<<std::endl;
        for(int j=0;j<response->device_id.size();j++){
            Request_Model_instance_data res_data=models_util[i];
            res_data.result_device_id=response->device_id[j];
//            std::cout<<res_data.model_name<<" "<<res_data.result_device_id<<std::endl;
            models_util_res.push_back(res_data);
        }
        
//        models_util[i].result_device_id = response->device_id;
    }

    engine.update_model_instance_num(models_util_res);
    //        std::cout<<"need load model "<<models_util.size()<<std::endl;
    if (models_util.size() <= 0){
        load_finish = true;
    }
    this->tick_cnt+=1;
    if(this->tick_cnt%this->log_util_interval==0){
        for (auto iter = this->counters.begin(); iter != counters.end(); iter++)
        {
            RCLCPP_INFO(this->get_logger(), "%s fps: %f intervals %s", iter->first.c_str(),iter->second->get_fps(),iter->second->get_intervals().c_str());
//            std::cout << iter->first << " fps: " << iter->second->get_fps() << " intervals " << iter->second->get_intervals() << std::endl;
        }

        for (auto iter = this->counters_real_time.begin(); iter != counters_real_time.end(); iter++)
        {
            RCLCPP_INFO(this->get_logger(), "%s real_time fps: %f intervals %s", iter->first.c_str(),iter->second->get_fps(),iter->second->get_intervals().c_str());
//            std::cout << iter->first << " fps: " << iter->second->get_fps() << " intervals " << iter->second->get_intervals() << std::endl;
        }
        engine.print_module_time_summary();
        std::cout<<engine_time_counter.to_string();
    }
};

void Tr_Alg_Engine_module::updata_channel(const tr_alg_interfaces::msg::ChannelList::ConstPtr &channel_list)
{
    if (load_finish == false)
        return;
    if (channel_list->changed || this->Subscription_.size() != channel_list->channel_list_string.size())
    {
        std::vector<std::string> channel_names = channel_list->channel_list_string;
        std::set<std::string> current_channel_names;

        for (auto iter = channel_names.begin(); iter != channel_names.end(); iter++)
        {
            current_channel_names.insert(*iter);
        }
        std::set<std::string> last_channel_names;
        for (auto iter = this->Subscription_.begin(); iter != this->Subscription_.end(); iter++)
        {
            last_channel_names.insert(iter->first);
        }

        RCLCPP_INFO(this->get_logger(), "load channel %d,%d", last_channel_names.size(), current_channel_names.size());

        std::vector<std::string> need_remove_channels(this->Subscription_.size());
        std::vector<std::string> need_add_channels(current_channel_names.size());

        auto need_remove_channels_iter = set_difference(last_channel_names.begin(), last_channel_names.end(), current_channel_names.begin(), current_channel_names.end(), need_remove_channels.begin());
        need_remove_channels.resize(need_remove_channels_iter - need_remove_channels.begin());

        auto need_add_channels_iter = set_difference(current_channel_names.begin(), current_channel_names.end(), last_channel_names.begin(), last_channel_names.end(), need_add_channels.begin());
        need_add_channels.resize(need_add_channels_iter - need_add_channels.begin());

        if (need_remove_channels.size() > 0)
        {
            for (int i = 0; i < need_remove_channels.size(); i++)
            {
                this->Subscription_.erase(need_add_channels[i]);
            }
        }
        if (need_add_channels.size() > 0)
        {
            for (int i = 0; i < need_add_channels.size(); i++)
            {
                auto channel_name = need_add_channels[i];
                rclcpp::SubscriptionOptions options;
                options.callback_group = recv_image_callback_group;
                this->counters[channel_name] = std::make_shared<Counter>();
                this->counters_real_time[channel_name] = std::make_shared<Counter>();
                this->Subscription_[channel_name] = this->create_subscription<tr_interfaces::msg::RvFrame>(
                    channel_name, rclcpp::SensorDataQoS(), [this, channel_name](const tr_interfaces::msg::RvFrame::ConstPtr &frame_data)
                    {
//                        RCLCPP_INFO(this->get_logger(), "channel tick %s",channel_name.c_str());
                        this->recv_image_callback(frame_data,channel_name); },
                    options);

                RCLCPP_INFO(this->get_logger(), "channel add %s", channel_name.c_str());
            }
        }
    }
};

Tr_Alg_Engine_module::~Tr_Alg_Engine_module(){};

void Tr_Alg_Engine_module::set_channel_cfg(std::string channel_name, std::string cfg_path)
{
    
    try
    {
        this->engine.set_channel_cfg(channel_name, cfg_path);
    }
    catch (Alg_Module_Exception &exception)
    {
        std::cerr << exception.module_name << " in " << exception.Stage2string(exception.stage) << " Error:" << exception.what()<<std::endl;
        return;
    }
};
bool Tr_Alg_Engine_module::load_module_from_dir(std::string lib_dir, std::string requirement_dir)
{
    try
    {
        this->engine.load_module_from_libdir(lib_dir, requirement_dir);
    }
    catch (Alg_Module_Exception &exception)
    {
        std::cerr << exception.module_name << " in " << exception.Stage2string(exception.stage) << " Error:" << exception.what()<<std::endl;
        return false;
    }
    return true;
};
bool Tr_Alg_Engine_module::load_module_from_path(std::string lib_path, std::string requirement_dir)
{
    try
    {
        this->engine.load_module_from_libfile(lib_path, requirement_dir);
    }
    catch (Alg_Module_Exception &exception)
    {
        std::cerr << exception.module_name << " in " << exception.Stage2string(exception.stage) << " Error:" << exception.what()<<std::endl;
        return false;
    }
    return true;
};
void Tr_Alg_Engine_module::set_device_handles(std::vector<int> &device_ids)
{
    
    std::vector<std::shared_ptr<Device_Handle>> device_handles;
    for (int i = 0; i < device_ids.size(); i++)
    {
        if (device_handles_all.find(device_ids[i]) == device_handles_all.end())
        {

            device_handles_all[device_ids[i]] = std::shared_ptr<Device_Handle>(get_device_handle(device_ids[i]));
        }
        device_handles.push_back(device_handles_all[device_ids[i]]);
    }
    device_handles_inuse = device_handles;
    this->engine.set_device_handles(device_handles);
};

bool Tr_Alg_Engine_module::check_loaded_modules()
{

    auto res = engine.check_same_node_name();
    if (res.size() > 0)
    {
        std::string error_string = "mult module have same name\n";
        for (int i = 0; i < res.size(); i++)
        {
            error_string += "\t" + res[i] + "\n";
        }
        std::cout <<error_string<<std::endl;
        return false;
    }
    auto res1 = engine.check_node_require();
    if (res1.size() > 0)
    {
        std::cout << "module require not meet" << std::endl;
        for (int i = 0; i < res1.size(); i++)
        {
            std::cout << "\t" << res1[i].error_msg << "\t" << res1[i].module_name << " : " << res1[i].param_name << std::endl;
        }

        return false;
    }
    auto res2 = engine.check_node_publish();
    if (res2.size() > 0)
    {
        std::cout << "module publish error" << std::endl;
        for (int i = 0; i < res2.size(); i++)
        {
            std::cout << "\t" << res2[i].error_msg << "\t" << res2[i].module_name << " : " << res2[i].param_name << std::endl;
        }
        return false;
    }


    return true;
};
std::shared_ptr<Device_Handle> Tr_Alg_Engine_module::get_random_device_handle()
{
    if (device_handles_inuse.size() <= 0)
        return std::shared_ptr<Device_Handle>();
    std::shared_ptr<Device_Handle> handle = device_handles_inuse[rand() % device_handles_inuse.size()];
    return handle;
};

template<typename T>
T decode_data(uint8_t* data,int data_type){
    if(data_type==sensor_msgs::msg::PointField::FLOAT32){
        return (T)*(float*)data;
    }
    else if(data_type==sensor_msgs::msg::PointField::FLOAT64){
        return (T)*(double*)data;
    }
    else if(data_type==sensor_msgs::msg::PointField::INT32){
        return (T)*(int32_t*)data;
    }
    else if(data_type==sensor_msgs::msg::PointField::INT16){
        return (T)*(int16_t*)data;
    }
    else if(data_type==sensor_msgs::msg::PointField::INT8){
        return (T)*(int8_t*)data;
    }
    else if(data_type==sensor_msgs::msg::PointField::UINT32){
        return (T)*(uint32_t*)data;
    }
    else if(data_type==sensor_msgs::msg::PointField::UINT16){
        return (T)*(uint16_t*)data;
    }
    else if(data_type==sensor_msgs::msg::PointField::UINT8){
        return (T)*(uint8_t*)data;
    }
    return (T)0;

};

bool Tr_Alg_Engine_module::trans_from_rv_frame_cloud(const tr_interfaces::msg::RvFrame::ConstPtr & frame_data,std::map<std::string,std::shared_ptr<InputOutput>>& input_data){
    uint8_t *ptr=(uint8_t *)frame_data->object.data.data();
    std::map<std::string,std::pair<int,int>> fileds;
    for(auto &field:frame_data->object.fields){
        fileds[field.name]={field.offset,field.datatype};
    }
    std::shared_ptr<InputOutput> objects=std::make_shared<InputOutput>(InputOutput::Type::ObjectCloud_t);
//    if(frame_data->object.data.size()>0){
//        std::cout<<"Receive object cloud not empty " <<frame_data->object.width<<std::endl;

//    }

    if(frame_data->object.width>0){
        int step=frame_data->object.data.size()/frame_data->object.width;

    objects->data.objectcloud.data.resize(frame_data->object.width);
    
//    std::cout<<"Receive object cloud3 " <<frame_data->object.width<<"\t"<<step<<std::endl;
    for(int i=0;i<frame_data->object.width;i++){
        ObjectItem object;
        if(fileds.find("pos_x")!=fileds.end()){
            object.x=decode_data<float>(ptr+fileds["pos_x"].first+step*i,fileds["pos_x"].second);
        }
        if(fileds.find("pos_y")!=fileds.end()){
            object.y=decode_data<float>(ptr+fileds["pos_y"].first+step*i,fileds["pos_y"].second);
        }
        if(fileds.find("pos_z")!=fileds.end()){
            object.z=decode_data<float>(ptr+fileds["pos_z"].first+step*i,fileds["pos_z"].second);
        }
        if(fileds.find("vel_x")!=fileds.end()){
            object.vx=decode_data<float>(ptr+fileds["vel_x"].first+step*i,fileds["vel_x"].second);
        }
        if(fileds.find("vel_y")!=fileds.end()){
            object.vy=decode_data<float>(ptr+fileds["vel_y"].first+step*i,fileds["vel_y"].second);
        }
        if(fileds.find("vel_z")!=fileds.end()){
            object.vz=decode_data<float>(ptr+fileds["vel_z"].first+step*i,fileds["vel_z"].second);
        }
        if(fileds.find("track_id")!=fileds.end()){
            object.track_id=decode_data<uint32_t>(ptr+fileds["track_id"].first+step*i,fileds["track_id"].second);
        }
        if(fileds.find("yaw")!=fileds.end()){
            object.yaw=decode_data<float>(ptr+fileds["yaw"].first+step*i,fileds["yaw"].second);
        }
                if(fileds.find("ext_0_0")!=fileds.end()){
                   object.ext_0=decode_data<float>(ptr+fileds["ext_0_0"].first+step*i,fileds["ext_0_0"].second);
                }
                if(fileds.find("ext_0_1")!=fileds.end()){
                   object.ext_1=decode_data<float>(ptr+fileds["ext_0_1"].first+step*i,fileds["ext_0_1"].second);
                }
                if(fileds.find("ext_0_2")!=fileds.end()){
                   object.ext_2=decode_data<float>(ptr+fileds["ext_0_2"].first+step*i,fileds["ext_0_2"].second);
                }

                if(fileds.find("ext_1_0")!=fileds.end()){
                   object.ext_3=decode_data<float>(ptr+fileds["ext_1_0"].first+step*i,fileds["ext_1_0"].second);
                }
                if(fileds.find("ext_1_1")!=fileds.end()){
                   object.ext_4=decode_data<float>(ptr+fileds["ext_1_1"].first+step*i,fileds["ext_1_1"].second);
                }
                if(fileds.find("ext_1_2")!=fileds.end()){
                   object.ext_5=decode_data<float>(ptr+fileds["ext_1_2"].first+step*i,fileds["ext_1_2"].second);
                }
  
                if(fileds.find("ext_2_0")!=fileds.end()){
                   object.ext_6=decode_data<float>(ptr+fileds["ext_2_0"].first+step*i,fileds["ext_2_0"].second);
                }
                if(fileds.find("ext_2_1")!=fileds.end()){
                   object.ext_7=decode_data<float>(ptr+fileds["ext_2_1"].first+step*i,fileds["ext_2_1"].second);
                }
                if(fileds.find("ext_2_2")!=fileds.end()){
                   object.ext_8=decode_data<float>(ptr+fileds["ext_2_2"].first+step*i,fileds["ext_2_2"].second);
                }
        
        objects->data.objectcloud.data[i]=object;
    }
//    std::cout<<objects->data.objectcloud.to_string()<<std::endl;
//        try{
//        throw 1;
//    }
//    catch(...){
        
//    }

    }

    input_data["radar_object"]=objects;

    ptr=(uint8_t *)frame_data->pointcloud.data.data();
    fileds.clear();
    for(auto &field:frame_data->pointcloud.fields){
        fileds[field.name]={field.offset,field.datatype};
    }
    std::shared_ptr<InputOutput> points=std::make_shared<InputOutput>(InputOutput::Type::PointCloud_t);
    if(frame_data->pointcloud.width>0){
        int step=frame_data->pointcloud.data.size()/frame_data->pointcloud.width;

    points->data.pointcloud.data.resize(frame_data->pointcloud.width);
    for(int i=0;i<frame_data->pointcloud.width;i++){
        PointItem item;
        if(fileds.find("x")!=fileds.end()){
            item.x=decode_data<float>(ptr+fileds["x"].first+step*i,fileds["x"].second);
        }
        if(fileds.find("y")!=fileds.end()){
            item.y=decode_data<float>(ptr+fileds["y"].first+step*i,fileds["y"].second);
        }
        if(fileds.find("z")!=fileds.end()){
            item.z=decode_data<float>(ptr+fileds["z"].first+step*i,fileds["z"].second);
        }
        if(fileds.find("dopper")!=fileds.end()){
            item.doppler=decode_data<float>(ptr+fileds["dopper"].first+step*i,fileds["dopper"].second);
        }
        if(fileds.find("intensity")!=fileds.end()){
            item.intensity=decode_data<float>(ptr+fileds["intensity"].first+step*i,fileds["intensity"].second);
        }
        points->data.pointcloud.data[i]=item;
    }
    }
    input_data["pointcloud"]=points;
    return true;
};


bool Tr_Alg_Engine_module::trans_from_rv_frame(const tr_interfaces::msg::RvFrame::ConstPtr &frame_data, std::map<std::string, std::shared_ptr<InputOutput>> &input_data)
{
    std::shared_ptr<QyImage> image;
    std::shared_ptr<Device_Handle> handle=get_random_device_handle();
    if(handle==nullptr){
        return false;
    }

    if (frame_data->image.encoding == "bgr8")
    {
        cv::Mat input_image_orig=cv::Mat(frame_data->image.height,frame_data->image.width,CV_8UC3);
        memcpy(input_image_orig.data,frame_data->image.data.data(),frame_data->image.data.size());
        image=from_mat(input_image_orig,handle,false);      
    }
    else if (frame_data->image.encoding == "jpeg")
    {


        try{
            image=jpeg_decoder->decode(frame_data->image.data);

        }
        catch(...){
                std::cout << "channel "<<" can not unpack frame data "<<std::endl;
                return false;

        }
    }
    else
    {
        std::cout << "image type "<< frame_data->image.encoding <<" not suppoted" << std::endl;
        return false;
    }

    if (image==nullptr||image->is_empty()){
            std::cout << "channel frame shape error"<< std::endl;
            try{
                throw 1;
            }
            catch(...){}
            return false;
    }


    std::shared_ptr<InputOutput> input_image;
    
    input_image =std::make_shared<InputOutput>(InputOutput::Type::Image_t);
    input_image->data.image=image;

    input_data["image"] = input_image;

    return true;
};


std::shared_ptr<QyImage> Tr_Alg_Engine_module::trans_from_rv_frame(const tr_interfaces::msg::RvFrame::ConstPtr &frame_data){
    std::shared_ptr<QyImage> image;
    std::shared_ptr<Device_Handle> handle=get_random_device_handle();
    if(handle==nullptr){
        return std::shared_ptr<QyImage>();
    }

    if (frame_data->image.encoding == "bgr8")
    {
        cv::Mat input_image_orig=cv::Mat(frame_data->image.height,frame_data->image.width,CV_8UC3);
        memcpy(input_image_orig.data,frame_data->image.data.data(),frame_data->image.data.size());
        image=from_mat(input_image_orig,handle,false);      
    }
    else if (frame_data->image.encoding == "jpeg")
    {


        try{
            image=jpeg_decoder->decode(frame_data->image.data);

        }
        catch(...){
                std::cout << "channel "<<" can not unpack frame data "<<std::endl;
                return std::shared_ptr<QyImage>();

        }
    }
    else
    {
        std::cout << "image type "<< frame_data->image.encoding <<" not suppoted" << std::endl;
        return std::shared_ptr<QyImage>();
    }

    if (image==nullptr||image->is_empty()){
            std::cout << "channel frame shape error"<< std::endl;
            try{
                throw 1;
            }
            catch(...){}

            return std::shared_ptr<QyImage>();
    }
    return image;

};  


void Tr_Alg_Engine_module::recv_image_callback(const tr_interfaces::msg::RvFrame::ConstPtr &frame_data, std::string channel_name)
{
    std::map<std::string, std::shared_ptr<InputOutput>> input_data;

    long long timestamp = (long long)frame_data->tr_header.data_stamp.sec * 1000 + frame_data->tr_header.data_stamp.nanosec / 1e6;
    bool need_real_time=this->engine.check_need_forward(channel_name, timestamp,true)>0;
    bool need_forward=this->engine.check_need_forward(channel_name, timestamp)>0;


//    cv::Mat im=image->get_image();
  //  cv::imwrite("/data_temp/result_temp/image"+std::to_string(tick_cnt)+".png",im);
    //std::ofstream out;
//    out.open("/data_temp/result_temp/image_s"+std::to_string(tick_cnt)+".png",std::ios::binary|std::ios::out);
  //  out.write((const char *)frame_data->image.data.data(),frame_data->image.data.size());
    //out.close();


    std::unique_lock<std::shared_mutex> lock_real_time(this->counters_real_time[channel_name]->mutex,std::try_to_lock);
    if(lock_real_time.owns_lock()){
        if(need_real_time==false){
            lock_real_time.unlock();
        }
    }
    else{
        if(need_real_time){
            need_real_time=false;
        }
    }

    std::unique_lock<std::shared_mutex> lock(this->counters[channel_name]->mutex,std::try_to_lock);
    if(need_real_time==false&&(need_forward==false||lock.owns_lock()==false))
        return;
    else if(need_real_time&&lock.owns_lock()==false){
        need_forward=false;
    }

    engine_time_counter.start_decode(channel_name,get_time());
    std::shared_ptr<QyImage> image=trans_from_rv_frame(frame_data);
    if(image==nullptr){
        return;

    }

//    std::shared_ptr<QyImage> image=trans_from_rv_frame(frame_data);
  //  if(image==nullptr)
    //    return;

//    cv::Mat im=image->get_image();
  //  cv::imwrite("/data_temp/result_temp/image.png",im);

    input_data["image"] =std::make_shared<InputOutput>(InputOutput::Type::Image_t);    
    input_data["image"]->data.image=image;

    input_data["timestamp"]=std::make_shared<InputOutput>(InputOutput::Type::Value_t,Value::Type::Long);
    input_data["timestamp"]->data.value.data.long_value=timestamp;
    trans_from_rv_frame_cloud(frame_data,input_data);


    std::vector<std::shared_ptr<Publish_data>> output_data_real_time;
    std::vector<std::shared_ptr<Publish_data>> output_data;
    engine_time_counter.end_decode(channel_name,get_time());
    engine_time_counter.start_inference(channel_name,get_time());

    //        std::cout<<"get image from"<<channel_name<<std::endl;
    if(need_real_time){
        this->counters_real_time[channel_name]->tick(timestamp);
#ifdef __WITH_TRY_CATCH__
        try
        {
#endif            
            output_data_real_time = engine.forward(channel_name, input_data, timestamp,true);
            lock_real_time.unlock();
#ifdef __WITH_TRY_CATCH__
        }
        catch (Alg_Module_Exception &exception)
        {
            std::cerr<<exception.exception_string<<std::endl;
            std::cerr<<exception.stack_string<<std::endl;
            lock_real_time.unlock();
            if(lock.owns_lock())
                lock.unlock();
            engine.channel_status[channel_name].status = ChannelStatus::Status::runtime_error;

            std::cerr << exception.module_name << " in " << exception.Stage2string(exception.stage) << " Error:" << exception.what()<<std::endl;
            return;
        }
        
        catch(...){
            lock_real_time.unlock();
            if(lock.owns_lock())
                lock.unlock();
            engine.channel_status[channel_name].status = ChannelStatus::Status::runtime_error;
            std::cerr << channel_name<<" unknown error"<<std::endl<<std::endl;
            return;

        }
#endif
    }
    if (need_forward&&this->engine.check_need_forward(channel_name, timestamp) <= 0)
        need_forward=false;

    if(need_forward==false && lock.owns_lock())
        lock.unlock();

    if(need_forward){
        this->counters[channel_name]->tick(timestamp);
#ifdef __WITH_TRY_CATCH__
        try
        {   
#endif
            output_data = engine.forward(channel_name, input_data, timestamp);
            lock.unlock();
#ifdef __WITH_TRY_CATCH__

        }
        catch (Alg_Module_Exception &exception)
        {
            std::cerr<<exception.exception_string<<std::endl;
            std::cerr<<exception.stack_string<<std::endl;
            lock.unlock();
            if(need_real_time&&lock_real_time.owns_lock())
                lock_real_time.unlock();
            engine.channel_status[channel_name].status = ChannelStatus::Status::runtime_error;

            std::cerr << exception.module_name << " in " << exception.Stage2string(exception.stage) << " Error:" << exception.what()<<std::endl;
            return;
        }
        
        catch(...){
            lock.unlock();
            if(need_real_time&&lock_real_time.owns_lock())
                lock_real_time.unlock();
            engine.channel_status[channel_name].status = ChannelStatus::Status::runtime_error;

            std::cerr << channel_name<<" unknown error"<<std::endl<<std::endl;
            return;

        }
#endif

    }
    engine_time_counter.end_inference(channel_name,get_time());
    engine_time_counter.start_publish(channel_name,get_time());

#ifdef __WITH_TRY_CATCH__
    try
    {
#endif
        if(need_real_time){
        for (int i = 0; i < output_data_real_time.size(); i++)
        {
            if (output_data_real_time[i]->topic_base == "")
            {
                std::vector<tr_interfaces::msg::TrImageAlgoResult> result;
                trans_to_event_publish_struct(frame_data, output_data_real_time[i], result);
                debug_show_event_publish_data(channel_name,result,timestamp);
                for (int i = 0; i < result.size(); i++)
                {
                    publisher_primary->publish(result[i]);
                }
            }
            else
            {
                tr_alg_interfaces::msg::Results result;
                trans_to_raw_publish_struct(channel_name, timestamp, output_data_real_time[i], result);
                if (this->raw_publisher.find(output_data_real_time[i]->topic_base) == this->raw_publisher.end())
                {
                    this->raw_publisher[output_data_real_time[i]->topic_base] = this->create_publisher<tr_alg_interfaces::msg::Results>(output_data_real_time[i]->topic_base, 10);
                }
                this->raw_publisher[output_data_real_time[i]->topic_base]->publish(result);
            }
        }
        }
        if(need_forward){
        for (int i = 0; i < output_data.size(); i++)
        {
            if (output_data[i]->topic_base == "")
            {
                std::vector<tr_interfaces::msg::TrImageAlgoResult> result;
                trans_to_event_publish_struct(frame_data, output_data[i], result);
                debug_show_event_publish_data(channel_name,result,timestamp);
                for (int i = 0; i < result.size(); i++)
                {
                    publisher_primary->publish(result[i]);
                }
            }
            else
            {
                tr_alg_interfaces::msg::Results result;
                trans_to_raw_publish_struct(channel_name, timestamp, output_data[i], result);
                if (this->raw_publisher.find(output_data[i]->topic_base) == this->raw_publisher.end())
                {
                    this->raw_publisher[output_data[i]->topic_base] = this->create_publisher<tr_alg_interfaces::msg::Results>(output_data[i]->topic_base, 10);
                }
                this->raw_publisher[output_data[i]->topic_base]->publish(result);
            }
        }
        }
#ifdef __WITH_TRY_CATCH__

    }
    
    catch(...){
        engine.channel_status[channel_name].status = ChannelStatus::Status::runtime_error;

        std::cerr << channel_name<<" unknown error1"<<std::endl<<std::endl;
        return;

    }
#endif    
    engine_time_counter.end_publish(channel_name,get_time());

    return;
};

bool Tr_Alg_Engine_module::trans_to_raw_publish_struct(std::string channel_name, long long timestamp, std::shared_ptr<Publish_data> output, tr_alg_interfaces::msg::Results &raw_result)
{
    if (output->data == nullptr)
    {
        return false;
    }
    if (output->data->data_type != InputOutput::Type::Result_Detect_t && output->data->data_type != InputOutput::Type::Result_Detect_license_t)
    {
        return false;
    }
    if (output->data->data_type == InputOutput::Type::Result_Detect_t)
    {
        raw_result.channel_name = channel_name;
        raw_result.module_name = output->module_name;
        raw_result.timestamp = timestamp;
        auto &result = raw_result.items;
        auto &outputs = output->data->data.detect;
        result.resize(outputs.size());
        for (int i = 0; i < outputs.size(); i++)
        {
            result[i].class_id = outputs[i].class_id;
            result[i].x1 = outputs[i].x1;
            result[i].x2 = outputs[i].x2;
            result[i].y1 = outputs[i].y1;
            result[i].y2 = outputs[i].y2;
            result[i].tag = outputs[i].tag;
            result[i].temp_idx = outputs[i].temp_idx;
            result[i].score = outputs[i].score;
            result[i].region_idx = outputs[i].region_idx;
            result[i].new_obj = outputs[i].new_obj;
            result[i].feature = outputs[i].feature;
            for (int j = 0; j < outputs[i].contour.size(); j++)
            {
                result[i].contour.push_back(outputs[i].contour[j].first);
                result[i].contour.push_back(outputs[i].contour[j].second);
            }
            result[i].mask_data = outputs[i].mask_data;
            result[i].mask_shape = outputs[i].mask_shape;
            if(outputs[i].ext_result.size()>0){
                result[i].ext_result.resize(outputs[i].ext_result.size());
                int idx=0;
                for (auto iter=outputs[i].ext_result.begin();iter!=outputs[i].ext_result.end();iter++,idx++){
                    auto & ext_res=result[i].ext_result[idx];
                    ext_res.name=iter->first;
                    ext_res.score=iter->second.score;
                    ext_res.class_id=iter->second.class_id;
                    ext_res.tag=iter->second.tag;

                }


            }

        }
    }
    else if (output->data->data_type == InputOutput::Type::Result_Detect_license_t)
    {
        raw_result.channel_name = channel_name;
        raw_result.module_name = output->module_name;
        raw_result.timestamp = timestamp;
        auto &result = raw_result.items;
        auto &outputs = output->data->data.detect_license;
        result.resize(outputs.size());
        for (int i = 0; i < outputs.size(); i++)
        {
            result[i].class_id = outputs[i].class_id;
            result[i].x1 = outputs[i].x1;
            result[i].x2 = outputs[i].x2;
            result[i].y1 = outputs[i].y1;
            result[i].y2 = outputs[i].y2;
            result[i].tag = outputs[i].tag;
            result[i].temp_idx = outputs[i].temp_idx;
            result[i].score = outputs[i].score;
            result[i].region_idx = outputs[i].region_idx;
            result[i].new_obj = outputs[i].new_obj;
            result[i].feature = outputs[i].feature;
            for (int j = 0; j < outputs[i].contour.size(); j++)
            {
                result[i].contour.push_back(outputs[i].contour[j].first);
                result[i].contour.push_back(outputs[i].contour[j].second);
            }
            result[i].mask_data = outputs[i].mask_data;
            result[i].mask_shape = outputs[i].mask_shape;
            result[i].landms_x1 = outputs[i].landms_x1;
            result[i].landms_x2 = outputs[i].landms_x2;
            result[i].landms_x3 = outputs[i].landms_x3;
            result[i].landms_x4 = outputs[i].landms_x4;
            result[i].landms_y1 = outputs[i].landms_y1;
            result[i].landms_y2 = outputs[i].landms_y2;
            result[i].landms_y3 = outputs[i].landms_y3;
            result[i].landms_y4 = outputs[i].landms_y4;
            result[i].license = outputs[i].license;
            result[i].car_idx = outputs[i].car_idx;
            switch (outputs[i].state)
            {
            case Result_Detect_license::Detect_state::SUCCESS:
                result[i].state = result[i].DETECT_STATE_SUCCESS;
                break;
            case Result_Detect_license::Detect_state::LOW_SCORE:
                result[i].state = result[i].DETECT_STATE_LOW_SCORE;
                break;
            case Result_Detect_license::Detect_state::SMALL_REGION:
                result[i].state = result[i].DETECT_STATE_SMALL_REGION;
                break;
            }

            switch (outputs[i].license_color)
            {
            case Result_Detect_license::License_Color::Blue:
                result[i].license_color = result[i].LICENSE_COLOR_BLUE;
                break;
            case Result_Detect_license::License_Color::Green:
                result[i].license_color = result[i].LICENSE_COLOR_GREEN;
                break;
            case Result_Detect_license::License_Color::Yellow:
                result[i].license_color = result[i].LICENSE_COLOR_YELLOW;
                break;
            case Result_Detect_license::License_Color::Yellow_Green:
                result[i].license_color = result[i].LICENSE_COLOR_YELLOW_GREEN;
                break;
            case Result_Detect_license::License_Color::Black:
                result[i].license_color = result[i].LICENSE_COLOR_BLACK;
                break;
            case Result_Detect_license::License_Color::White:
                result[i].license_color = result[i].LICENSE_COLOR_WHITE;
                break;
            case Result_Detect_license::License_Color::Color_UNKNOWN:
                result[i].license_color = result[i].LICENSE_COLOR_UNKNOWN;
                break;
            }
            switch (outputs[i].license_type)
            {
            case Result_Detect_license::License_Type::Single:
                result[i].license_type = result[i].LICENSE_TYPE_SINGLE;
                break;
            case Result_Detect_license::License_Type::Double:
                result[i].license_type = result[i].LICENSE_TYPE_DOUBLE;
                break;
            case Result_Detect_license::License_Type::Type_UNKNOWN:
                result[i].license_type = result[i].LICENSE_TYPE_UNKNOWN;
                break;
            }
            if(outputs[i].ext_result.size()>0){
                result[i].ext_result.resize(outputs[i].ext_result.size());
                int idx=0;
                for (auto iter=outputs[i].ext_result.begin();iter!=outputs[i].ext_result.end();iter++,idx++){
                    auto & ext_res=result[i].ext_result[idx];
                    ext_res.name=iter->first;
                    ext_res.score=iter->second.score;
                    ext_res.class_id=iter->second.class_id;
                    ext_res.tag=iter->second.tag;

                }


            }
        }
    }
    return true;
};
bool Tr_Alg_Engine_module::debug_show_event_publish_data(std::string channel_name,std::vector<tr_interfaces::msg::TrImageAlgoResult>& result,long long timestamp_in){
    if(result.size()<=0)
        return false;
    
    std::cout<<"------------------------"<<channel_name<<"-------------------------"<<std::endl;
    time_t timestamp=time_t(timestamp_in/1000);
    tm* date= localtime(&timestamp);
    std::cout<< (date->tm_year + 1900)<<"-"<<(date->tm_mon + 1)<<"-"<<date->tm_mday<<" "<<date->tm_hour<<":"<<date->tm_min<<":"<<date->tm_sec<<"."<<timestamp_in%1000<<std::endl;

    std::cout<<result.size()<<" event"<<std::endl;
    for(int i=0;i<result.size();i++){
        auto& item=result[i];
        std::cout<<item.tag<<std::endl;
        std::cout<<item.x1<<" "<<item.y1<<" "<<item.x2<<" "<<item.y2<<std::endl;
        std::cout<<item.class_id<<" "<<item.score<<std::endl;
        if (item.license!=""){
            std::cout<<item.license<<" "<<(int)item.license_color<<" "<<(int)item.license_type<<std::endl;
        }
        std::cout<<item.tag<<std::endl;

    }

    return true;
};


bool Tr_Alg_Engine_module::trans_to_event_publish_struct(const tr_interfaces::msg::RvFrame::ConstPtr &frame_data, std::shared_ptr<Publish_data> output, std::vector<tr_interfaces::msg::TrImageAlgoResult> &result)
{
    if (output->data == nullptr)
    {
        return false;
    }
    if (output->data->data_type != InputOutput::Type::Result_Detect_t && output->data->data_type != InputOutput::Type::Result_Detect_license_t)
    {
        return false;
    }
    if (output->data->data_type == InputOutput::Type::Result_Detect_t)
    {

        auto &outputs = output->data->data.detect;
        result.resize(outputs.size());
        for (int i = 0; i < outputs.size(); i++)
        {
            result[i].rv_frame = *frame_data;
            result[i].tr_header = frame_data->tr_header;
            result[i].module_name = output->module_name;
            result[i].class_id = outputs[i].class_id;
            result[i].x1 = outputs[i].x1;
            result[i].x2 = outputs[i].x2;
            result[i].y1 = outputs[i].y1;
            result[i].y2 = outputs[i].y2;
            result[i].tag = outputs[i].tag;
            result[i].temp_idx = outputs[i].temp_idx;
            result[i].score = outputs[i].score;
            result[i].region_idx = outputs[i].region_idx;
            result[i].new_obj = outputs[i].new_obj;
            result[i].feature = outputs[i].feature;
            for (int j = 0; j < outputs[i].contour.size(); j++)
            {
                result[i].contour.push_back(outputs[i].contour[j].first);
                result[i].contour.push_back(outputs[i].contour[j].second);
            }
            result[i].mask_data = outputs[i].mask_data;
            result[i].mask_shape = outputs[i].mask_shape;
            result[i].res_images.resize(outputs[i].res_images.size());
            int idx = 0;
            for (auto iter = outputs[i].res_images.begin(); iter != outputs[i].res_images.end(); iter++, idx++)
            {
                auto &image_msg = result[i].res_images[idx].image;
                result[i].res_images[idx].name = iter->first;

                image_msg.header.frame_id = this->node_name;
                image_msg.header.stamp = frame_data->image.header.stamp;
                image_msg.height = iter->second.rows;
                image_msg.width = iter->second.cols;
                image_msg.encoding = "bgr8";
                image_msg.is_bigendian = false;
                image_msg.step = iter->second.step;
                size_t size = iter->second.step * iter->second.rows;
                image_msg.data.resize(size);
                memcpy(&image_msg.data[0], iter->second.data, size);
            }
            if(outputs[i].ext_result.size()>0){
                result[i].ext_result.resize(outputs[i].ext_result.size());
                int idx=0;
                for (auto iter=outputs[i].ext_result.begin();iter!=outputs[i].ext_result.end();iter++,idx++){
                    auto & ext_res=result[i].ext_result[idx];
                    ext_res.name=iter->first;
                    ext_res.score=iter->second.score;
                    ext_res.class_id=iter->second.class_id;
                    ext_res.tag=iter->second.tag;

                }

            }

        }
    }
    else if (output->data->data_type == InputOutput::Type::Result_Detect_license_t)
    {
        auto &outputs = output->data->data.detect_license;
        result.resize(outputs.size());
        for (int i = 0; i < outputs.size(); i++)
        {
            result[i].rv_frame = *frame_data;
            result[i].tr_header = frame_data->tr_header;
            result[i].module_name = output->module_name;
            result[i].class_id = outputs[i].class_id;
            result[i].x1 = outputs[i].x1;
            result[i].x2 = outputs[i].x2;
            result[i].y1 = outputs[i].y1;
            result[i].y2 = outputs[i].y2;
            result[i].tag = outputs[i].tag;
            result[i].temp_idx = outputs[i].temp_idx;
            result[i].score = outputs[i].score;
            result[i].region_idx = outputs[i].region_idx;
            result[i].new_obj = outputs[i].new_obj;
            result[i].feature = outputs[i].feature;
            for (int j = 0; j < outputs[i].contour.size(); j++)
            {
                result[i].contour.push_back(outputs[i].contour[j].first);
                result[i].contour.push_back(outputs[i].contour[j].second);
            }
            result[i].mask_data = outputs[i].mask_data;
            result[i].mask_shape = outputs[i].mask_shape;
            result[i].res_images.resize(outputs[i].res_images.size());
            int idx = 0;
            for (auto iter = outputs[i].res_images.begin(); iter != outputs[i].res_images.end(); iter++, idx++)
            {
                auto &image_msg = result[i].res_images[idx].image;
                result[i].res_images[idx].name = iter->first;

                image_msg.header.frame_id = this->node_name;
                image_msg.header.stamp = frame_data->image.header.stamp;
                image_msg.height = iter->second.rows;
                image_msg.width = iter->second.cols;
                image_msg.encoding = "bgr8";
                image_msg.is_bigendian = false;
                image_msg.step = iter->second.step;
                size_t size = iter->second.step * iter->second.rows;
                image_msg.data.resize(size);
                memcpy(&image_msg.data[0], iter->second.data, size);
            }
            result[i].landms_x1 = outputs[i].landms_x1;
            result[i].landms_x2 = outputs[i].landms_x2;
            result[i].landms_x3 = outputs[i].landms_x3;
            result[i].landms_x4 = outputs[i].landms_x4;
            result[i].landms_y1 = outputs[i].landms_y1;
            result[i].landms_y2 = outputs[i].landms_y2;
            result[i].landms_y3 = outputs[i].landms_y3;
            result[i].landms_y4 = outputs[i].landms_y4;
            result[i].license = outputs[i].license;
            result[i].car_idx = outputs[i].car_idx;
            switch (outputs[i].state)
            {
            case Result_Detect_license::Detect_state::SUCCESS:
                result[i].state = result[i].DETECT_STATE_SUCCESS;
                break;
            case Result_Detect_license::Detect_state::LOW_SCORE:
                result[i].state = result[i].DETECT_STATE_LOW_SCORE;
                break;
            case Result_Detect_license::Detect_state::SMALL_REGION:
                result[i].state = result[i].DETECT_STATE_SMALL_REGION;
                break;
            }

            switch (outputs[i].license_color)
            {
            case Result_Detect_license::License_Color::Blue:
                result[i].license_color = result[i].LICENSE_COLOR_BLUE;
                break;
            case Result_Detect_license::License_Color::Green:
                result[i].license_color = result[i].LICENSE_COLOR_GREEN;
                break;
            case Result_Detect_license::License_Color::Yellow:
                result[i].license_color = result[i].LICENSE_COLOR_YELLOW;
                break;
            case Result_Detect_license::License_Color::Yellow_Green:
                result[i].license_color = result[i].LICENSE_COLOR_YELLOW_GREEN;
                break;
            case Result_Detect_license::License_Color::Black:
                result[i].license_color = result[i].LICENSE_COLOR_BLACK;
                break;
            case Result_Detect_license::License_Color::White:
                result[i].license_color = result[i].LICENSE_COLOR_WHITE;
                break;
            case Result_Detect_license::License_Color::Color_UNKNOWN:
                result[i].license_color = result[i].LICENSE_COLOR_UNKNOWN;
                break;
            }
            switch (outputs[i].license_type)
            {
            case Result_Detect_license::License_Type::Single:
                result[i].license_type = result[i].LICENSE_TYPE_SINGLE;
                break;
            case Result_Detect_license::License_Type::Double:
                result[i].license_type = result[i].LICENSE_TYPE_DOUBLE;
                break;
            case Result_Detect_license::License_Type::Type_UNKNOWN:
                result[i].license_type = result[i].LICENSE_TYPE_UNKNOWN;
                break;
            }
            if(outputs[i].ext_result.size()>0){
                result[i].ext_result.resize(outputs[i].ext_result.size());
                int idx=0;
                for (auto iter=outputs[i].ext_result.begin();iter!=outputs[i].ext_result.end();iter++,idx++){
                    auto & ext_res=result[i].ext_result[idx];
                    ext_res.name=iter->first;
                    ext_res.score=iter->second.score;
                    ext_res.class_id=iter->second.class_id;
                    ext_res.tag=iter->second.tag;

                }


            }


        }
    }
    return true;
};

void Tr_Alg_Engine_module::set_channel_config_callback(const tr_alg_interfaces::srv::SetChannelConfig::Request::SharedPtr request,
                                                       const tr_alg_interfaces::srv::SetChannelConfig::Response::SharedPtr response){
    if(request->mode=="set"){
        std::cout<<"set channel cfg "<<request->channel_topic_name<<" "<<request->cfg_file_path<<std::endl;
        this->set_channel_cfg(request->channel_topic_name,request->cfg_file_path);
        response->res=0;
    }
    else{
        response->res=-1;
        response->message="not supported set mode";
    }
};
void Tr_Alg_Engine_module::set_module_enable_callback(const tr_alg_interfaces::srv::SetModuleEnable::Request::SharedPtr request,
                                                      const tr_alg_interfaces::srv::SetModuleEnable::Response::SharedPtr response){
    if(request->mode==request->MODE_ENABLE){
        this->engine.enable_modules(request->module_name,request->channel_name);
    }
    else if(request->mode==request->MODE_DISABLE){
        this->engine.disable_modules(request->module_name,request->channel_name);
    }
    else if(request->mode==request->MODE_RELOAD){
        for(int i=0;i<request->module_name.size();i++)
            this->engine.reload_module(request->module_name[i]);
    }

};
void Tr_Alg_Engine_module::get_module_enable_callback(const tr_alg_interfaces::srv::GetModuleEnable::Request::SharedPtr request,
                                                      const tr_alg_interfaces::srv::GetModuleEnable::Response::SharedPtr response){
    this->engine.get_enable_modules(request->channel_name,response->module_name,response->module_enable);
};
void Tr_Alg_Engine_module::get_channel_status(const tr_alg_interfaces::srv::GetChannelStatus::Request::SharedPtr request,
                                              const tr_alg_interfaces::srv::GetChannelStatus::Response::SharedPtr response){
    this->engine.get_channel_status(request->channel_name,response->channel_name,response->status,response->str);

};
