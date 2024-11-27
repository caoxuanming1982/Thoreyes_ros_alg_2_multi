import os
import shutil
import json
import argparse


def run(include_dirs,test_main,project_name,install_root,project_branch,test_branch,build_type):
    #以下一般不要修改
    define_root="/opt/alg_sub_module_develop/"

    if os.path.exists("./lib"):
        shutil.rmtree("./lib")
    if os.path.exists("./test"):
        shutil.rmtree("./test")

    if os.path.exists(define_root+"./base_define/base.json"):
        base_cfg=json.load(open(define_root+"./base_define/base.json","r"))
    else:
        base_cfg={"init_env":[],"link_lib":""}

    branch_base_name=base_cfg["install_dir_name"]        #bm：算丰，ix:天数，nv：nvidia，hw：华为

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
        if branch_name not in branch_base_name:
            print(branch_name,"is not supported")
            continue
        lib_cmakelists[branch_name]=lib_cmakelist+"project("+project_name+"_"+branch_name+")\n"
        lib_cmakelists[branch_name]+="SET(CMAKE_INSTALL_PREFIX "+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_submodules)\n"
        lib_cmakelists[branch_name]+="include_directories("+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/include/alg_module_base)\n"  
        lib_cmakelists[branch_name]+="link_directories("+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/lib)\n"


    for branch_name in project_branch:
        if branch_name not in branch_base_name:
            continue
        if os.path.exists(define_root+"./base_define/"+branch_name+".json"):
            cfg=json.load(open(define_root+"./base_define/"+branch_name+".json","r"))
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
        if branch_name not in branch_base_name:
            continue
        if os.path.exists("./lib/"+branch_name)==False:
            os.makedirs("./lib/"+branch_name)
        f=open("./lib/"+branch_name+"/CMakeLists.txt","w")
        f.write(lib_cmakelists[branch_name])
        f.close()

    for branch_name in project_branch:
        if branch_name not in branch_base_name:
            continue
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
        if branch_name not in branch_base_name:
            continue
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

    if test_branch in project_branch and test_branch in branch_base_name:
        if os.path.exists(define_root+"./base_define/"+test_branch+".json"):
            cfg=json.load(open(define_root+"./base_define/"+test_branch+".json","r"))
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

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("-n","--name",type=str,default="alg_module_bus_detection",help="project_name")               #算法模块名

    parser.add_argument("-i","--include",type=str,nargs="+",default=["./include","./deepsort/include"],help="include dirs")     #include的目录
    parser.add_argument("-m","--test",type=str,default="./main.cpp",help="test main function")                                  #调试时main函数所在的cpp文件路径

    parser.add_argument("-b","--branch",type=str,nargs="+",default=["bm","nv"],help="build branch")                        #需要编译的分支
    parser.add_argument("-tb","--test_branch",type=str,default="nv",help="test main function for branch")                       #调试main函数时使用的分支

    parser.add_argument("-B","--build",type=str,default="Release",help="Release or Debug")                                      #

    parser.add_argument("-s","--install",type=str,default="/data_temp/",help="install root dir")                                #安装算法模块的根目录
    args=parser.parse_args()
    run(args.include,args.test,args.name,args.install,args.branch,args.test_branch,args.build)

# /usr/bin/python3 gen_cmakelist.py --branch bm --test_branch bm
# /usr/bin/python3 gen_cmakelist.py --branch nv --test_branch nv --build Debug