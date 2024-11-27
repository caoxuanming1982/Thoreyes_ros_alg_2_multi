import os
import json
import argparse
import shutil


def run(include_dirs,node_main,project_name,install_root,interface_root,project_branch,ext_test,build_type):
    #以下一般不要修改
    define_root="/opt/alg_sub_module_develop/"

    if os.path.exists(define_root+"./base_define/base.json"):
        base_cfg=json.load(open(define_root+"./base_define/base.json","r"))
    else:
        base_cfg={"init_env":[],"link_lib":""}

    for name in base_cfg["install_dir_name"]:
        if os.path.exists("./"+name):
            shutil.rmtree("./"+name)

    if os.path.exists("./base_define/base.json"):
        cur_cfg=json.load(open("./base_define/base.json","r"))
    else:
        cur_cfg={"init_env":[],"link_lib":""}



    branch_base_name=base_cfg["install_dir_name"]        #bm：算丰，ix:天数，nv：nvidia，hw：华为
    interface_base_name=base_cfg["interface_dir_name"]        #bm：算丰，ix:天数，nv：nvidia，hw：华为

    branch_src_remap=cur_cfg["src_remap"]

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
        if branch_name=="bm":            
            lib_cmakelists[branch_name]=lib_cmakelist+"project("+project_name+")\n"

        else:
            lib_cmakelists[branch_name]=lib_cmakelist+"project("+project_name+"_"+branch_name+")\n"

        lib_cmakelists[branch_name]+="include_directories("+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/include/alg_module_base)\n"  
        lib_cmakelists[branch_name]+="link_directories("+install_root+"/"+branch_base_name[branch_name]+"/ros/alg_module_base/lib)\n"
        lib_cmakelists[branch_name]+="set(tr_interfaces_DIR "+interface_root+"/"+interface_base_name[branch_name]+"/ros/thoreyes_base/install/share/tr_interfaces/cmake/)\n"
        lib_cmakelists[branch_name]+="set(tr_alg_interfaces_DIR "+interface_root+"/"+interface_base_name[branch_name]+"/ros/alg_module_interfaces/share/tr_alg_interfaces/cmake/)\n"
        lib_cmakelists[branch_name]+="\n".join(cur_cfg["init_env"])+"\n"

    for branch_name in project_branch:
        if branch_name not in branch_base_name:
            continue
        if os.path.exists(define_root+"./base_define/"+branch_name+".json"):
            cfg=json.load(open(define_root+"./base_define/"+branch_name+".json","r"))
        else:
            cfg={"init_env":[],"link_lib":""}
            continue

        if "init_env" in cfg:
            lib_cmakelists[branch_name]+="\n".join(cfg["init_env"])+"\n"

        lib_cmakelists[branch_name]+="SET(CMAKE_BUILD_TYPE \""+build_type+"\")\n"

        for name in include_dirs:
            lib_cmakelists[branch_name]+="include_directories("+"../../"+name+")\n"

        
        lib_cmakelists[branch_name]+="file(GLOB SRC_FILES ../../src/*.cpp)\n"
        names=os.listdir("../src")
        src_files=["SRC_FILES"]
        for name in names:
            if os.path.isdir("../src/"+name):
                target_branch_name=branch_name
                if branch_name in branch_src_remap:
                    target_branch_name=branch_src_remap[branch_name]

                f_name="SRC_FILES_"+name
                lib_cmakelists[branch_name]+="file(GLOB "+f_name+" ../../src/"+name+"//*.cpp)\n"
                src_files.append(f_name)
                if os.path.exists("../src/"+name+"/"+target_branch_name):
                    f_name_p="SRC_FILES_"+name+"_P"
                    lib_cmakelists[branch_name]+="file(GLOB "+f_name_p+" ../../src/"+name+"/"+target_branch_name+"/*.cpp)\n"
                    src_files.append(f_name_p)


        lib_cmakelists[branch_name]+="add_library(${PROJECT_NAME}_share SHARED "
        for file in src_files:
            lib_cmakelists[branch_name]+="${"+file+"} "
        lib_cmakelists[branch_name]+=")\n"
        lib_cmakelists[branch_name]+="ament_target_dependencies(${PROJECT_NAME}_share\n"
        lib_cmakelists[branch_name]+="\n".join(cur_cfg["ament_target_dependencies"])+"\n"
        lib_cmakelists[branch_name]+=")\n"

        lib_cmakelists[branch_name]+="target_link_libraries(${PROJECT_NAME}_share\n"
        lib_cmakelists[branch_name]+=cfg["link_lib_engine"]+"\n"
        lib_cmakelists[branch_name]+=base_cfg["link_lib"]+"\n"
        lib_cmakelists[branch_name]+=")\n"

        lib_cmakelists[branch_name]+="install(DIRECTORY ../../include/ DESTINATION include/${PROJECT_NAME}/)\n"
        lib_cmakelists[branch_name]+="install(TARGETS ${PROJECT_NAME}_share EXPORT ${PROJECT_NAME}_share DESTINATION lib)\n"

        lib_cmakelists[branch_name]+="ament_export_targets(${PROJECT_NAME}_share HAS_LIBRARY_TARGET)\n"
        lib_cmakelists[branch_name]+="ament_export_dependencies(  \n"
        lib_cmakelists[branch_name]+="\n".join(cur_cfg["ament_export_dependencies"])+"\n"
        lib_cmakelists[branch_name]+=")\n"

        lib_cmakelists[branch_name]+="ament_export_targets(${PROJECT_NAME}_share HAS_LIBRARY_TARGET)\n"
        lib_cmakelists[branch_name]+="ament_export_include_directories(\"include/${PROJECT_NAME}\")\n"

        lib_cmakelists[branch_name]+="ament_export_libraries(${PROJECT_NAME}_share)\n"

        node_files=os.listdir("../"+node_main)

        for name in node_files:
            target_name="_".join(name.split(".")[:-1])
            lib_cmakelists[branch_name]+="add_executable("+target_name+" ../../"+node_main+"/"+name+")\n"
            lib_cmakelists[branch_name]+="target_link_libraries("+target_name+" ${PROJECT_NAME}_share)\n"
            lib_cmakelists[branch_name]+="install(TARGETS "+target_name+" DESTINATION lib/${PROJECT_NAME})\n"

        lib_cmakelists[branch_name]+="ament_package()\n"

        for name in ext_test:
            
            target_name="_".join("_".join(name.split(".")[:-1]).replace("\\","/").split("/"))
            while target_name.startswith("_"):
                target_name=target_name[1:]

            lib_cmakelists[branch_name]+="add_executable("+target_name+" ../../"+name+")\n"
            lib_cmakelists[branch_name]+="target_link_libraries("+target_name+" ${PROJECT_NAME}_share)\n"


    for branch_name in project_branch:
        if branch_name not in branch_base_name:
            continue
        if os.path.exists("./"+branch_name)==False:
            os.makedirs("./"+branch_name)
        f=open("./"+branch_name+"/CMakeLists.txt","w")
        f.write(lib_cmakelists[branch_name])
        f.close()

    for branch_name in project_branch:
        if branch_name not in branch_base_name:
            continue

        package_text="<?xml version=\"1.0\"?>\n"
        package_text+="<?xml-model href=\"http://download.ros.org/schema/package_format3.xsd\" schematypens=\"http://www.w3.org/2001/XMLSchema\"?>\n"
        package_text+="<package format=\"3\">\n"
        if branch_name=="bm":            
            package_text+="  <name>"+project_name+"</name>\n"
        else:
            package_text+="  <name>"+project_name+"_"+branch_name+"</name>\n"
        package_text+="  <version>0.0.0</version>\n"
        package_text+="  <description>TODO: Package description</description>\n"
        package_text+="  <maintainer email=\"vensin@todo.todo\">vensin</maintainer>\n"
        package_text+="  <license>TODO: License declaration</license>\n"
        package_text+="  <buildtool_depend>ament_cmake</buildtool_depend>\n"
        package_text+="  <depend>rclcpp</depend>\n"
        package_text+="  <build_depend>rosidl_default_generators</build_depend>\n"
        package_text+="  <exec_depend>rosidl_default_runtime</exec_depend>\n"
        package_text+="  <member_of_group>rosidl_interface_packages</member_of_group>\n"
        package_text+="  <test_depend>ament_lint_auto</test_depend>\n"
        package_text+="  <test_depend>ament_lint_common</test_depend>\n"
        package_text+="  <export>\n"
        package_text+="    <build_type>ament_cmake</build_type>\n"
        package_text+="  </export>\n"
        package_text+="</package>\n"
        f=open("./"+branch_name+"/package.xml","w")
        f.write(package_text)
        f.close()

    make_txt="cd ..\n"  
    for branch_name in project_branch:
        if branch_name not in branch_base_name:
            continue
        make_txt+="rm -rf ./build/*\n"
        make_txt+="rm -rf ./install/*\n"
        if branch_name=="bm":            
            make_txt+="colcon build --packages-select "+project_name+" --merge-install  --install-base "+install_root+"/"+branch_base_name[branch_name]+"/ros/"+project_name+"\n"  
        else:
            make_txt+="colcon build --packages-select "+project_name+"_"+branch_name+" --merge-install  --install-base "+install_root+"/"+branch_base_name[branch_name]+"/ros/"+project_name+"\n"  
    make_txt+="cd project\n"  
    f=open("./make.sh","w")
    f.write(make_txt)
    f.close()

    








if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-n","--name",type=str,default="tr_alg_engine",help="project_name")               #算法模块名

    parser.add_argument("-i","--include",type=str,nargs="+",default=["./include"],help="include dirs")     #include的目录
    parser.add_argument("-m","--main",type=str, default="./node",help="node main function")                                  #调试时main函数所在的cpp文件路径
    parser.add_argument("-et","--ext_test",type=str,nargs="+", default=["./test_src/test_submodule_reload.cpp"],help="ext test main function")                                  #调试时main函数所在的cpp文件路径

    parser.add_argument("-b","--branch",type=str,nargs="+",default=["bm","nv_cvcuda"],help="build branch")                        #需要编译的分支
    parser.add_argument("-B","--build",type=str,default="Release",help="Release or Debug")                                      #
 
    parser.add_argument("-s","--install",type=str,default="/data/",help="install root dir")                                #安装算法模块的根目录
    parser.add_argument("--interface",type=str,default="/data/",help="install root dir")                                #安装算法模块的根目录
    args=parser.parse_args()
    run(args.include,args.main,args.name,args.install,args.interface,args.branch,args.ext_test,args.build)

