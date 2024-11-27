import os
import shutil
import json

include_dirs=["./include","./include_private"]              #当前项目的include目录
install_dirs=["./include"]

test_main="../test/test_cv_lib.cpp"                  #调试中测试用main函数所在的文件
project_name="alg_module_base"  #项目名
install_root="/data/"                          #安装根目录

project_branch=["bm","nv_cvcuda"]                     #批量编译的分支 "bm","nv",
build_type="Release"                                #是否编译为调试版本
sub_module_develop_dir="/opt/alg_sub_module_develop/"


sub_module_cfg=json.load(open("./sub_module_define/base.json"))
branch_base_name=sub_module_cfg["install_dir_name"]
#branch_base_name={"bm":"thoreyes","ix":"thoreyes_ix","ix_cvcuda":"thoreyes_ix","nv":"thoreyes_nv","hw":"thoreyes_hw"} #bm：算丰，ix:天数，nv：nvidia，hw：华为

if os.path.exists("./lib"):
    shutil.rmtree("./lib")
if os.path.exists("./test"):
    shutil.rmtree("./test")

lib_cmakelist=""
lib_cmakelist+="cmake_minimum_required(VERSION 3.8)\n"
lib_cmakelist+="if(NOT CMAKE_C_STANDARD)\n"
lib_cmakelist+="  set(CMAKE_C_STANDARD 99)\n"
lib_cmakelist+="endif()\n"
lib_cmakelist+="if(NOT CMAKE_CXX_STANDARD)\n"
lib_cmakelist+="  set(CMAKE_CXX_STANDARD 17)\n"
lib_cmakelist+="endif()\n"

lib_cmakelists={}
for branch_name in project_branch:
    lib_cmakelists[branch_name]=lib_cmakelist+"project("+project_name+"_"+branch_name+")\n"


for branch_name in project_branch:
    lib_cmakelists[branch_name]+="SET(CMAKE_INSTALL_PREFIX "+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base)\n"

if os.path.exists("./base_define/base.json"):
    base_cfg=json.load(open("./base_define/base.json","r"))
else:
    base_cfg={"init_env":[],"link_lib":""}

for branch_name in project_branch:

    if os.path.exists("./base_define/"+branch_name+".json"):
        cfg=json.load(open("./base_define/"+branch_name+".json","r"))
    else:
        cfg={"init_env":[],"link_lib":""}
        continue

    if "init_env" in base_cfg:
        lib_cmakelists[branch_name]+="\n".join(base_cfg["init_env"])+"\n"

    lib_cmakelists[branch_name]+="SET(CMAKE_BUILD_TYPE \""+build_type+"\")\n"
    for name in include_dirs:
        lib_cmakelists[branch_name]+="include_directories("+"../../../"+name+")\n"

    lib_cmakelists[branch_name]+= "\n".join(cfg["init_env"])+"\n"

    lib_cmakelists[branch_name]+="file(GLOB SRC_FILES ../../../src/*.cpp)\n"
    lib_cmakelists[branch_name]+="file(GLOB SRC_FILES_CV_LIB ../../../src/cv_lib/*.cpp)\n"
    lib_cmakelists[branch_name]+="file(GLOB SRC_FILES_CV_LIB_P ../../../src/cv_lib/"+cfg["cv_lib_branch"]+"/*.cpp)\n"
    lib_cmakelists[branch_name]+="file(GLOB SRC_FILES_NETWORK ../../../src/network_engine/*.cpp)\n"
    lib_cmakelists[branch_name]+="file(GLOB SRC_FILES_NETWORK_P ../../../src/network_engine/"+cfg["network_kernel_branch"]+"/*.cpp)\n"


    
    lib_cmakelists[branch_name]+="add_library(${PROJECT_NAME}_share SHARED ${SRC_FILES} ${SRC_FILES_CV_LIB} ${SRC_FILES_CV_LIB_P} ${SRC_FILES_NETWORK} ${SRC_FILES_NETWORK_P})\n"
    lib_cmakelists[branch_name]+="target_link_libraries(${PROJECT_NAME}_share\n"
    if "link_lib" in cfg:
        lib_cmakelists[branch_name]+= cfg["link_lib"]+"\n"
    if "link_lib" in base_cfg:
        lib_cmakelists[branch_name]+= base_cfg["link_lib"]+"\n"

    lib_cmakelists[branch_name]+= ")\n"

    lib_cmakelists[branch_name]+="add_executable(${PROJECT_NAME}_test_cv_lib_main  ../../"+test_main+")\n"
    lib_cmakelists[branch_name]+="target_link_libraries(${PROJECT_NAME}_test_cv_lib_main \n"
    if "link_lib" in cfg:
        lib_cmakelists[branch_name]+= cfg["link_lib"]+"\n"
    if "link_lib" in base_cfg:
        lib_cmakelists[branch_name]+= base_cfg["link_lib"]+"\n"
    lib_cmakelists[branch_name]+= "${PROJECT_NAME}_share\n"
    lib_cmakelists[branch_name]+= ")\n"


lib_cmakelist+="project("+project_name+")\n"

for branch_name in project_branch:
    if os.path.exists("./lib/"+branch_name)==False:
        os.makedirs("./lib/"+branch_name)
    f=open("./lib/"+branch_name+"/CMakeLists.txt","w")
    f.write(lib_cmakelists[branch_name])
    f.close()

for branch_name in project_branch:
    lib_cmakelist+="add_subdirectory(./lib/"+branch_name+")\n"

run_make_txt="cd ..\n"
run_make_txt+="rm -rf build/*\n"
run_make_txt+="mkdir build\n"
run_make_txt+="cd build\n"
run_make_txt+="cmake ../project\n"
run_make_txt+="make -j10\n"



f=open("./CMakeLists.txt","w")
f.write(lib_cmakelist)
f.close()



for branch_name in project_branch:
    src_name="./lib/"+branch_name+"/"+project_name+"_"+branch_name+"_test_cv_lib_main"
    dst_name="./"+project_name+"_"+branch_name+"_test_cv_lib_main"
    run_make_txt+="cp "+src_name+" "+dst_name+"\n"


run_make_txt+="cd ../project\n"
f=open("./make.sh","w")
f.write(run_make_txt)
f.close()

run_install_txt="cd ../build/\n"
for branch_name in project_branch:
    src_name="./lib/"+branch_name+"/lib"+project_name+"_"+branch_name+"_share.so"
    dst_name=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/lib/lib"+project_name+"_"+branch_name+"_share.so"
    re_name=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/lib/lib"+project_name+"_share.so"
    dst_dir=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/lib/"
    run_install_txt+="sudo mkdir -p "+dst_dir+"\n"
    run_install_txt+="sudo cp "+src_name+" "+dst_name+"\n"
    run_install_txt+="rm "+re_name+"\n"
    run_install_txt+="ln -s "+dst_name+" "+re_name+"\n"


for branch_name in project_branch:
    dst_dir=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/include/"+project_name+"/"
    run_install_txt+="sudo rm -rf "+dst_dir+"\n"

    for install_dir in install_dirs:
        names=os.listdir("../"+install_dir)
        for name in names:
            src_name="../"+install_dir+"/"+name
            dst_name=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/include/"+project_name+"/"+name
            dst_dir=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/include/"+project_name+"/"
            run_install_txt+="sudo mkdir -p "+dst_dir+"\n"
            run_install_txt+="sudo cp -r "+src_name+" "+dst_name+"\n"

for branch_name in project_branch:
    if os.path.exists("../doc/"):
        src_name="../doc/"
        dst_name=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/doc/"
        run_install_txt+="sudo cp -r "+src_name+" "+dst_name+"\n"




run_install_txt+="cd ../project\n"
run_install_txt+="sudo rm  -rf "+sub_module_develop_dir+"\n"
run_install_txt+="sudo mkdir -p "+sub_module_develop_dir+"\n"
run_install_txt+="sudo cp -r "+"./sub_module_define"+" "+sub_module_develop_dir+"/base_define"+"\n"


f=open("./make_install.sh","w")
f.write(run_install_txt)
f.close()
