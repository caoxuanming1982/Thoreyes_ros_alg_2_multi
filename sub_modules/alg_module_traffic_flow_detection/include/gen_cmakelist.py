import os
import shutil
import json

include_dirs=["./include","./deepsort/include"]              #当前项目的include目录
test_main="./main.cpp"                  #调试中测试用main函数所在的文件


project_name="alg_module_traffic_flow_detection"  #项目名
install_root="/data_temp/"                          #安装根目录

project_branch=["bm","ix","nv"]                     #批量编译的分支
test_branch="nv"                                    #测试的分支，一次只能编译一个分支的测试，一般情况下，只需要测试其中一个分支就能保证其他分支推理正常
build_type="Release"                                #是否编译为调试版本
#build_type="Debug"

#以下一般不要修改
branch_base_name={"bm":"thoreyes","ix":"thoreyes_ix","nv":"thoreyes_nv","hw":"thoreyes_hw"} #bm：算丰，ix:天数，nv：nvidia，hw：华为

if os.path.exists("./lib"):
    shutil.rmtree("./lib")
if os.path.exists("./test"):
    shutil.rmtree("./test")

if os.path.exists("./base_define/base.json"):
    base_cfg=json.load(open("./base_define/base.json","r"))
else:
    base_cfg={"init_env":[],"link_lib":""}

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
    lib_cmakelists[branch_name]+="SET(CMAKE_INSTALL_PREFIX "+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_submodules)\n"
    lib_cmakelists[branch_name]+="include_directories("+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/include/alg_module_base)\n"  
    lib_cmakelists[branch_name]+="link_directories("+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/lib)\n"


for branch_name in project_branch:
    if os.path.exists("./base_define/"+branch_name+".json"):
        cfg=json.load(open("./base_define/"+branch_name+".json","r"))
    else:
        cfg={"init_env":[],"link_lib":""}
        continue
    if "init_env" in base_cfg:
        lib_cmakelists[branch_name]+="\n".join(base_cfg["init_env"])+"\n"
    lib_cmakelists[branch_name]+="SET(CMAKE_BUILD_TYPE \""+build_type+"\")\n"
    if "init_env_only_link" in cfg:
        lib_cmakelists[branch_name]+="\n".join(cfg["init_env_only_link"])+"\n"

    for name in include_dirs:
        lib_cmakelists[branch_name]+="include_directories("+"../../../"+name+")\n"
    lib_cmakelists[branch_name]+="file(GLOB_RECURSE SRC_FILES ../../../src/*.cpp)\n"
    lib_cmakelists[branch_name]+="add_library(${PROJECT_NAME}_share SHARED ${SRC_FILES})\n"



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
run_make_txt+="cd ../project\n"



run_install_txt="cd ../build/\n"
for branch_name in project_branch:
    src_name="./lib/"+branch_name+"/lib"+project_name+"_"+branch_name+"_share.so"
    dst_name=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_submodules/lib/lib"+project_name+"_share.so"
    dst_dir=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_submodules/lib/"
    run_install_txt+="sudo mkdir -p "+dst_dir+"\n"
    run_install_txt+="sudo cp "+src_name+" "+dst_name+"\n"
    dst_name=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_submodules/base/lib"+project_name+"_share.so"
    dst_dir=install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_submodules/base/"
    run_install_txt+="sudo mkdir -p "+dst_dir+"\n"
    run_install_txt+="sudo cp "+src_name+" "+dst_name+"\n"

run_install_txt+="cd ../project\n"

test_cmakelist=""
test_cmakelist+="cmake_minimum_required(VERSION 3.8)\n"
test_cmakelist+="if(NOT CMAKE_C_STANDARD)\n"
test_cmakelist+="  set(CMAKE_C_STANDARD 99)\n"
test_cmakelist+="endif()\n"
test_cmakelist+="if(NOT CMAKE_CXX_STANDARD)\n"
test_cmakelist+="  set(CMAKE_CXX_STANDARD 17)\n"
test_cmakelist+="endif()\n"

if test_branch in project_branch:
    if os.path.exists("./base_define/"+test_branch+".json"):
        cfg=json.load(open("./base_define/"+test_branch+".json","r"))
    else:
        cfg={"init_env":[],"link_lib":""}
        
    test_cmakelist+="project(test_"+project_name+"_"+test_branch+")\n"

    if "init_env" in base_cfg:
        test_cmakelist+="\n".join(base_cfg["init_env"])+"\n"

    test_cmakelist+="include_directories("+install_root+"/"+branch_base_name[test_branch]+"/ros/alg_module_base/include/alg_module_base)\n"  
    test_cmakelist+="link_directories("+install_root+"/"+branch_base_name[test_branch]+"/ros/alg_module_base/lib)\n"
    test_cmakelist+="SET(CMAKE_BUILD_TYPE \""+build_type+"\")\n"

    test_cmakelist+="\n".join(cfg["init_env"])+"\n"
#    test_cmakelist+="link_directories(../../build/"+test_branch+"/"+")\n"

    for name in include_dirs:
        test_cmakelist+="include_directories("+"../../"+name+")\n"

    test_cmakelist+="add_executable(${PROJECT_NAME}_main  ../../"+test_main+")\n"
    test_cmakelist+="target_link_libraries(${PROJECT_NAME}_main\n"
    test_cmakelist+=cfg["link_lib"]+"\n"
    test_cmakelist+=base_cfg["link_lib"]+"\n"
    test_cmakelist+=project_name+"_"+test_branch+"_share\n"
    test_cmakelist+=")"

    if os.path.exists("./test/")==False:
        os.makedirs("./test/")

    f=open("./test/CMakeLists.txt","w")
    f.write(test_cmakelist)
    f.close()
    lib_cmakelist+="add_subdirectory(./test)\n"

    src_name="../build/test/test_"+project_name+"_"+test_branch+"_main"
    dst_name="../build/"+project_name+"_main"
    run_make_txt+="cp "+src_name+" "+dst_name+"\n"


f=open("./CMakeLists.txt","w")
f.write(lib_cmakelist)
f.close()

f=open("./make.sh","w")
f.write(run_make_txt)
f.close()

f=open("./make_install.sh","w")
f.write(run_install_txt)
f.close()
