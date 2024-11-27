新版使用流程（2024.9.19）

移植流程：
    修改图像处理的api，以及获取推理硬件handle相关的api，原始图像为输入模块的图像时，使用QyImage中自带的图像处理api进行相关操作,然后放入模型输入的数组std::vector<std::shared_ptr<QyImage>>
    修改测试部分的代码，同样使用QyImage作为模块的图像输入，修改获取和设置handle的代码

编译流程：
    修改project/gen_cmakelist.py中开头的几个配置，具体参数见gen_cmakelist.py中的注释
    cd project
    python gen_cmakelist.py
    chmod 777 make.sh
    chmod 777 make_install.sh

    ./make.sh           可编译指定数个分支，以及指定单个分支的测试程序，make.sh后，build下*_main的程序为测试程序
    sudo ./make_install.sh   (调试完成，模块稳定后)可安装编译时选择的分支到配置中的发布目录

