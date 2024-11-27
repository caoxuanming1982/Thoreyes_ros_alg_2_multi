/*
    项目编译方法
    rm -rf build & cmake -B build & cd build & make & ./alg_module_sample_main
*/

#include "alg_module_burst_into_ban_detection.h"
#include <iostream>
#include <filesystem> 

namespace fs = std::filesystem; 

cv::Mat drawText(cv::Mat image, int x, int y, string text)
{
    cv::Point point(x, y);
    cv::putText(image, text, point, 1, 2, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
    return image;
};

cv::Mat drawBox(cv::Mat image, int x1, int y1, int x2, int y2, cv::Scalar color = cv::Scalar(255, 127, 255))
{
    // cv::Scalar(255, 255, 255
    cv::Rect box(x1, y1, x2 - x1, y2 - y1);
    cv::rectangle(image, box, color, 2);
    return image;
};

cv::Mat drawLandms(cv::Mat image, int landms_x1, int landms_y1, int landms_x2, int landms_y2, int landms_x3, int landms_y3, int landms_x4, int landms_y4) 
{
    cv::Point2d p1(landms_x1, landms_y1);
    cv::Point2d p2(landms_x2, landms_y2);
    cv::Point2d p3(landms_x3, landms_y3);
    cv::Point2d p4(landms_x4, landms_y4);

    cv::circle(image, p1, 5, cv::Scalar(255, 0, 0), -1, cv::LINE_8);    // 蓝 绿 红
    cv::circle(image, p2, 5, cv::Scalar(0, 255, 0), -1, cv::LINE_8);
    cv::circle(image, p3, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_8);
    cv::circle(image, p4, 5, cv::Scalar(255, 255, 0), -1, cv::LINE_8);

    return image;
};

Alg_Module_Burst_Into_Ban_Detection* get_alg_module_burst_into_ban_detection(std::string root_dir)
{
    // 模型初始化
    Alg_Module_Burst_Into_Ban_Detection* module = new Alg_Module_Burst_Into_Ban_Detection();

    module->init_from_root_dir(root_dir);                           //初始化，加载模型和配置文件
    module->get_module_cfg()->get_input_output_cfg();               //可以获取输入输出配置文件
    module->get_module_cfg()->get_publish_cfg();                    //可以获取发布的topic的配置文件

    // 请求设备
    std::vector<std::shared_ptr<Device_Handle>> device_handles;                        //获取tpu的核心的handle
    device_handles.resize(1);
//    for (long unsigned int i=0; i<device_handles.size(); i++) {
        device_handles[0]=(std::shared_ptr<Device_Handle>(get_device_handle(2)));
  //  }

    // 增加模型实例
    module->set_device_handles(device_handles);                     //在模块中增加设备句柄
    auto utils = module->check_model_util();                        //
    for (auto iter = utils.begin(); iter != utils.end(); iter++) {
        if (iter->second<0 || iter->second>0.8) {
            for(long unsigned int i=0; i<device_handles.size(); i++)
                module->increase_model_instane(iter->first, device_handles[i]->get_device_id());
        }
    }
    return module;
}

void test_video(std::string video_path, std::string channel_cfg_path)
{ 
    std::string channel_name = "test";
//    std::string burst_into_ban_detection_root_dir = "../requirement";
    std::string txt="../requirement";
//    std::shared_ptr<char []> burst_into_ban_detection_root_dir=std::shared_ptr<char []>(new char[txt.size()]);
  //  strcpy(burst_into_ban_detection_root_dir.get(),txt.c_str());

    Alg_Module_Burst_Into_Ban_Detection* module_burst_into_ban_detection = nullptr;
    module_burst_into_ban_detection = get_alg_module_burst_into_ban_detection(txt); 
    
    module_burst_into_ban_detection->load_channel_cfg(channel_name, channel_cfg_path);
    module_burst_into_ban_detection->init_channal_data(channel_name);
    
    std::string event_save_path = "../requirement/results/event";
    std::string video_save_path = "../requirement/results/result.mp4";

    //打开视频
    cv::VideoCapture video;
    video.open(video_path);
        try{
            throw 1;
        }
        catch(...){}
    int frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);

    //保存视频
    int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat image(height, width, CV_8UC3);

    cv::VideoWriter writer;
    int coder = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 25.0;
    writer.open(video_save_path, coder, fps, cv::Size(width, height), CV_8UC3);

    //保存事件图片
    int event_count = 0;
    if (access(event_save_path.c_str(), F_OK) == 0) {
        //清空文件夹下的检测结果
        std::string cmd = "rm " + event_save_path + "/*";
        system(cmd.c_str());
    } else {
        //创建保存事件发生时截图的图片
        std::string cmd = "mkdir " + event_save_path;
        system(cmd.c_str());
    } 
    std::shared_ptr<Device_Handle>  handle(get_device_handle(0));
    for (int i = 0; i < frame_count; ++i)
    {
        std::cout << std::to_string(i+1) << "/" << std::to_string(frame_count);

        video.read(image);
        std::shared_ptr<QyImage> input_image=from_mat(image,handle);

        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image = input_image;
        
        std::map<std::string,std::shared_ptr<InputOutput>> input;
        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;
    
        input["image"] = input_data_image;
        module_burst_into_ban_detection->forward(channel_name, input, output);//推理
        module_burst_into_ban_detection->filter(channel_name, output, filter_output);//过滤
        module_burst_into_ban_detection->display(channel_name, input, filter_output);//可视化

        
        std::vector<Result_Detect> results = filter_output["result"]->data.detect;

        std::vector<Result_Detect> results_r = output["result"]->data.detect;
        for (int i = 0; i < results_r.size(); i++)
        {
            drawBox(image,results_r[i].x1,results_r[i].y1,results_r[i].x2,results_r[i].y2);
        }
        //保存事件检测结果
        for (int i = 0; i < results.size(); i++)
        {   
            std::string event_image_save_path = event_save_path + "/" + std::to_string(i) + "_" + std::to_string(event_count) + "_" + results[i].tag + ".png";
            cv::imwrite(event_image_save_path, results[i].res_images["image"]);
            event_count++;
            if (results[i].class_id == 0) {
                std::cout << "行人闯入数量: " << results[i].ext_result["person_num"].class_id << std::endl;
            }

        }
        writer.write(image);

        std::cout << std::endl;
    }
    writer.release();

}

// 使用图片进行测试

void test_memory(std::string images_path){
    std::string channel_name = "test";

    // 检查路径是否存在  
    if (!fs::exists(images_path) || !fs::is_directory(images_path)) {  
        std::cerr << "Path does not exist or is not a directory." << std::endl;  
        return;  
    }
 
    std::string txt="../requirement";
    Alg_Module_Burst_Into_Ban_Detection* module_burst_into_ban_detection = nullptr;
    module_burst_into_ban_detection = get_alg_module_burst_into_ban_detection(txt); 
    
    // 统一检测区域  使用全图检测
    std::string channel_cfg_path = "../requirement/imgs/all.xml";
    module_burst_into_ban_detection->load_channel_cfg(channel_name, channel_cfg_path);
    module_burst_into_ban_detection->init_channal_data(channel_name);
    std::shared_ptr<Device_Handle>  handle(get_device_handle(2));

    for (const auto& entry : fs::directory_iterator(images_path)) {  
        auto path = entry.path();  
  
        if (path.extension() != ".jpg") continue;
        std::cout << "Reading image: " << path << std::endl;  
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);  

        if (image.empty()) {  
            std::cerr << "Failed to load image: " << path << std::endl;  
            continue;  
        }

        std::cout <<entry<< std::endl;
        for(int i=0;i<10000;i++){
        std::shared_ptr<QyImage> input_image=from_mat(image,handle);


        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image= input_image;

        std::map<std::string,std::shared_ptr<InputOutput>> input;
        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;

        input["image"] = input_data_image;
        module_burst_into_ban_detection->forward(channel_name, input, output);//推理

        }
    }

}
void test_image(std::string images_path)
{ 
    std::string channel_name = "test";

    // 检查路径是否存在  
    if (!fs::exists(images_path) || !fs::is_directory(images_path)) {  
        std::cerr << "Path does not exist or is not a directory." << std::endl;  
        return;  
    }
 
//    std::string burst_into_ban_detection_root_dir = "../requirement";
    std::string txt="../requirement";
//    std::shared_ptr<char []> burst_into_ban_detection_root_dir=std::shared_ptr<char []>(new char[txt.size()]);
  //  strcpy(burst_into_ban_detection_root_dir.get(),txt.c_str());
    Alg_Module_Burst_Into_Ban_Detection* module_burst_into_ban_detection = nullptr;
    module_burst_into_ban_detection = get_alg_module_burst_into_ban_detection(txt); 
    
    // 统一检测区域  使用全图检测
    std::string channel_cfg_path = "../requirement/imgs/all.xml";
    module_burst_into_ban_detection->load_channel_cfg(channel_name, channel_cfg_path);
    module_burst_into_ban_detection->init_channal_data(channel_name);
    
    std::string event_save_path = "../requirement/imgs/event";

    //保存事件图片
    int event_count = 0;
    if (access(event_save_path.c_str(), F_OK) == 0) {
        //清空文件夹下的检测结果
        std::string cmd = "rm " + event_save_path + "/*";
        system(cmd.c_str());
    } else {
        //创建保存事件发生时截图的图片
        std::string cmd = "mkdir " + event_save_path;
        system(cmd.c_str());
    } 

    for (const auto& entry : fs::directory_iterator(images_path)) {  
        auto path = entry.path();  
  
        if (path.extension() != ".jpg") continue;
        std::cout << "Reading image: " << path << std::endl;  
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);  

        if (image.empty()) {  
            std::cerr << "Failed to load image: " << path << std::endl;  
            continue;  
        }

        std::shared_ptr<Device_Handle>  handle(get_device_handle(0));
        std::shared_ptr<QyImage> input_image=from_mat(image,handle);


        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image= input_image;

        std::map<std::string,std::shared_ptr<InputOutput>> input;
        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;

        input["image"] = input_data_image;
        module_burst_into_ban_detection->forward(channel_name, input, output);//推理
        module_burst_into_ban_detection->filter(channel_name, output, filter_output);//过滤
        module_burst_into_ban_detection->display(channel_name, input, filter_output);//可视化


        //保存检测视频
        auto& results = filter_output["result"]->data.detect;
        std::cout << "\t 检测结果数量:" << results.size();
        for (int i = 0; i < results.size(); i++) {   
            // 画框
            image = drawBox(image, results[i].x1, results[i].y1, results[i].x2, results[i].y2);
            // 显示置信度
            std::ostringstream oss;  
            oss << std::fixed << std::setprecision(1) <<  results[i].score;  
            std::string confidenceText = oss.str(); 
            int baseLine = 2;  
            cv::Size labelSize = cv::getTextSize(confidenceText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);  
            int x = results[i].x1;  
            int y = results[i].y1 - labelSize.height - baseLine + 5;  
            if (y < 0) {  
                y = 0;  
            }
            cv::putText(image, confidenceText, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3);

            if(results[i].tag == "行人 闯入 "){
                std::string event_image_save_path = event_save_path + "/" + std::to_string(i) + "_" + std::to_string(event_count) + "_行人闯入.png";
                // cout<<"图片已保存: "<<event_image_save_path<<endl;
                cv::imwrite(event_image_save_path, image);
                event_count++;
            }
        }
        std::cout << std::endl;
    }
}


int main(int argc, char* argv[])
{
//    test_video("../test_data/闯入/1.mp4", "../test_data/all.xml");
//    test_image("../test_data/images/");
    test_memory("../test_data/images/");
//    test_video("../test_data/闯入/行人闯入_测试视频.mp4", "../test_data/all.xml");
}
