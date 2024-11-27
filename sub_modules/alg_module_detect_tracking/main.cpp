/*
    项目编译方法
    rm -rf build & cmake -B build & cd build & make & ./alg_module_sample_main
*/

#include "alg_module_detect_tracking.h"
#include <iostream>
#include <filesystem>
#include <time.h>

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

Alg_Module_Detect_Tracking* get_alg_module_detect_tracking(std::string root_dir)
{
    // 模型初始化
    Alg_Module_Detect_Tracking* module_detect_tracking = new Alg_Module_Detect_Tracking();
    module_detect_tracking->init_from_root_dir(root_dir);
    module_detect_tracking->get_module_cfg()->get_input_output_cfg();
    module_detect_tracking->get_module_cfg()->get_publish_cfg();

    // 请求设备
    std::vector<std::shared_ptr<Device_Handle>> device_handles;     
    device_handles.resize(1);
    for (long unsigned int i=0; i<device_handles.size(); i++) 
    {
        device_handles[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
    }

    // 增加模型实例
    module_detect_tracking->set_device_handles(device_handles);
    auto utils = module_detect_tracking->check_model_util();
    for (auto iter = utils.begin(); iter != utils.end(); iter++) 
    {
        if (iter->second<0 || iter->second>0.8) 
        {
            for(long unsigned int i=0; i<device_handles.size(); i++)
                module_detect_tracking->increase_model_instane(iter->first, i);
        }
    }

    return module_detect_tracking;
};

void test_video(std::string video_path, std::string channel_cfg_path="", int max_frame_count=100000)
{ 
    std::string channel_name = "test";

    std::string detect_tracking_root_dir = "../requirement";
    Alg_Module_Detect_Tracking* module_detect_tracking = get_alg_module_detect_tracking(detect_tracking_root_dir);
    module_detect_tracking->init_channal_data(channel_name);

    //视频文件,结果保存
    std::string video_save_path = "../requirement/results/result.mp4";
    std::string event_save_path = "../requirement/results/event";

    //打开视频
    cv::VideoCapture video;
    video.open(video_path);
    int frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
    frame_count = frame_count < max_frame_count ? frame_count : max_frame_count;
    
    //保存视频
    int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat image;
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

    std::shared_ptr<Device_Handle> handle(get_device_handle(0));
    int cnt=0;
    long start=get_time();
    for (int i = 0; i < frame_count; ++i)
    {
        if (i%5 != 0) {
            video.grab();
            continue;
        } 

        video.read(image);
        std::shared_ptr<QyImage> input_image=from_mat(image,handle);
        cnt++;
        std::cout << std::to_string(i+1) << "/" << std::to_string(frame_count);

        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image=input_image;

        std::map<std::string,std::shared_ptr<InputOutput>> input;
        std::map<std::string,std::shared_ptr<InputOutput>> output;
        std::map<std::string,std::shared_ptr<InputOutput>> filter_output;

        input["image"] = input_data_image;
        module_detect_tracking->forward(channel_name, input, output); //检测和追踪
        module_detect_tracking->filter(channel_name, output, filter_output); //检测和追踪
        module_detect_tracking->forward(channel_name, input, filter_output); //检测和追踪


  //      clock_t end = clock();

        std::cout << " 检测结果数量 " << output["vehicle"]->data.detect.size(); 

        //可视化跟踪结果
        if (output.find("origin1") != output.end()) {
            for (auto& result : output["origin1"]->data.detect) {   
                image = drawBox(image, result.x1, result.y1, result.x2, result.y2, cv::Scalar(255, 255, 255));
            }
        }
        if (output.find("origin2") != output.end()) {
            for (auto& result : output["origin2"]->data.detect) {   
                image = drawBox(image, result.x1, result.y1, result.x2, result.y2, cv::Scalar(255, 0, 0));
            }
        }
        for (auto& result : output["vehicle"]->data.detect) {   
            image = drawBox(image, result.x1, result.y1, result.x2, result.y2);
            image = drawText(image, result.x1, result.y1, std::to_string(result.temp_idx));
        }
        writer.write(image);

        std::cout << std::endl;

    }
    long end=get_time();
        std::cout << " time " << double(end-start)/cnt << " ms"<<std::endl; 

    writer.release();
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cout << "参数不足: 视频路径 检测帧数" << std::endl;
        return 0;
    }

    std::string argv1 = argv[1];
    int argv2 = std::stoi(argv[2]);

    std::filesystem::path path(argv1);
    if (!std::filesystem::exists(path)) {  
        std::cout << "该路径不存在" << std::endl;
        return 0;
    } 
    
    test_video(path, "", argv2);

    return 0;
}
