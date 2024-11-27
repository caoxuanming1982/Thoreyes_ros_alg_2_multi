/*
    项目编译方法
    rm -rf build & cmake -B build & cd build & make & ./alg_module_sample_main
*/

#include "alg_module_helmet_detect_in_region.h"
#include <iostream>
using namespace std;


cv::Mat drawText(cv::Mat image, int x, int y, string text)
{
    cv::Point point(x, y);
    cv::putText(image, text, point, 1, 2, cv::Scalar(0, 0, 0), 2, cv::LINE_8);
    return image;
}

cv::Mat drawBox(cv::Mat image, int x1, int y1, int x2, int y2, cv::Scalar color = cv::Scalar(255, 127, 255))
{
    // cv::Scalar(255, 255, 255
    cv::Rect box(x1, y1, x2 - x1, y2 - y1);
    cv::rectangle(image, box, color, 2);
    return image;
}

cv::Mat drawLandms(cv::Mat image, int landms_x1, int landms_y1, int landms_x2, int landms_y2, int landms_x3, int landms_y3, int landms_x4, int landms_y4)
{
    cv::Point2d p1(landms_x1, landms_y1);
    cv::Point2d p2(landms_x2, landms_y2);
    cv::Point2d p3(landms_x3, landms_y3);
    cv::Point2d p4(landms_x4, landms_y4);

    cv::circle(image, p1, 5, cv::Scalar(255, 0, 0), -1, cv::LINE_8); // 蓝 绿 红
    cv::circle(image, p2, 5, cv::Scalar(0, 255, 0), -1, cv::LINE_8);
    cv::circle(image, p3, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_8);
    cv::circle(image, p4, 5, cv::Scalar(255, 255, 0), -1, cv::LINE_8);

    return image;
}

Alg_Module_helmet_detect_in_region *get_alg_module(std::string root_dir)
{

    // 模型初始化
    Alg_Module_helmet_detect_in_region *module = new Alg_Module_helmet_detect_in_region();
    module->init_from_root_dir(root_dir);             // 初始化，加载模型和配置文件
    module->get_module_cfg()->get_input_output_cfg(); // 可以获取输入输出配置文件
    module->get_module_cfg()->get_publish_cfg();      // 可以获取发布的topic的配置文件

    // 请求设备
    std::vector<std::shared_ptr<Device_Handle>> device_handles;                        //获取tpu的核心的handle
    device_handles.resize(1);
    for (long unsigned int i=0; i<device_handles.size(); i++) 
    {
        device_handles[i]=std::shared_ptr<Device_Handle>(get_device_handle(i));
//        bm_dev_request(&device_handles[i], i);
    }

    // 增加模型实例
    module->set_device_handles(device_handles); // 在模块中增加设备句柄
    auto utils = module->check_model_util();    //
    for (auto iter = utils.begin(); iter != utils.end(); iter++)
    {
        if (iter->second < 0 || iter->second > 0.8)
        {
            try{
                throw 1;

            }
            catch(...){}
            for (long unsigned int i = 0; i < device_handles.size(); i++)
                module->increase_model_instane(iter->first, i);
        }
    }

    return module;
}
void print_memory_usage() {
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, shared, text, lib, data, dt;
        statm >> size >> resident >> shared >> text >> lib >> data >> dt;
        statm.close();
        
        std::cout << "Memory Usage (in pages): " << std::endl;
        std::cout << "Size: " << size << " (total) " << std::endl;
        std::cout << "Resident: " << resident << " (in RAM) " << std::endl;
        std::cout << "Shared: " << shared << std::endl;
        std::cout << "Text: " << text << std::endl;
        std::cout << "Data: " << data << std::endl;
    } else {
        std::cerr << "Unable to open /proc/self/statm" << std::endl;
    }
}
void test_channel(Alg_Module_helmet_detect_in_region *module, std::string name, std::string src_path, std::string cfg_path, std::string dest_video_path, std::string dest_image_path)
{
    // 载入通道文件
    if (!module->load_channel_cfg(name, cfg_path))
    {
        return;
    }
    // 初始化通道数据
    auto ch_data = module->init_channal_data(name);
    if (!ch_data)
    {
        return;
    }

    // 加载视频流
    cv::VideoCapture video;
    video.open(src_path);
    if (!video.isOpened()) {
        std::cerr << "无法打开视频文件！" << std::endl;
        return;
    }

    int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Mat image(height, width, CV_8UC3);
    cv::VideoWriter writer;
    int coder = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 25.0;
    writer.open(dest_video_path, coder, fps, cv::Size(width, height), CV_8UC3);
    
    int frame_num = video.get(cv::CAP_PROP_FRAME_COUNT);
    std::shared_ptr<Device_Handle> handle(get_device_handle(0));

    for (int frame_id = 0; frame_id < frame_num; frame_id++)
    {
       

        if(video.read(image) == false){
            break;    
        }
        std::shared_ptr<QyImage> input_image=from_mat(image,handle);
        auto input_data_image = std::make_shared<InputOutput>(InputOutput::Type::Image_t);
        input_data_image->data.image = input_image;

        
        std::map<std::string, std::shared_ptr<InputOutput>> input;
        std::map<std::string, std::shared_ptr<InputOutput>> output;
        std::map<std::string, std::shared_ptr<InputOutput>> filter_output;
        
        input["image"] = input_data_image;
        module->forward(name, input, output); // 推理

        auto &output_results = output["result"]->data.detect;

        module->filter(name, output, filter_output); // 过滤        
        auto &filter_results = filter_output["result"]->data.detect;

        if (filter_output["result"] != nullptr && filter_results.size() != 0)
        {
            std::cout << " filter " << filter_results.size() << std::endl;
        }
        else
        {
            //bm_image_destroy(input_image);
            continue;
        }

        module->display(name, input, filter_output); // 可视化

        auto &last_filter_results = filter_output["result"]->data.detect;
        if (filter_output["result"] != nullptr && last_filter_results.size() != 0)
        {  
            std::cout << " display filter " << last_filter_results.size() << std::endl;
        }
        else
        {
            //bm_image_destroy(input_image);
            continue;
        }


        // Debug: save marked image
        for (const auto &det : filter_output.at("result")->data.detect)
        {
            image = det.res_images.at("image");
            writer.write(image);
        }


        //bm_image_destroy(input_image);
    }
    writer.release();
}

void test()
{

    std::string root_dir = "/home/ubuntu/pen_workspace/alg_module_helmet_detect_in_region/requirement";

    Alg_Module_helmet_detect_in_region *module = get_alg_module(root_dir);
    std::string channel_name2 = "test";
    std::string channel_src2 = root_dir + "/test/" + "3.mkv";
    std::string channel_cfg2 = root_dir + "/test/" + "truck.xml";
    std::string channel_dest_video2 = root_dir + "/results/" + "all.mp4";
    std::string channel_dest_image2 = root_dir + "/results/"  + ".png";


    
    //test_image_detection(module,channel_name1,channel_cfg1,channel_src1, channel_dest_image1);
    
    test_channel(module, channel_name2, channel_src2, channel_cfg2, channel_dest_video2, channel_dest_image2);
}

int main(void)
{
    test();
}