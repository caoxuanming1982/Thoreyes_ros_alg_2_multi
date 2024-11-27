import os
target_branch=["bm","nv_cvcuda"]
build_type="Release"
install_root="/data/"
python_exec="/usr/bin/python3 "
test_branch="bm"
names=os.listdir("./")

build_txt=""
install_txt=""

for name in names:
    if os.path.isdir("./"+name):
        path="./"+name+"/project"
        if os.path.exists(path)==False:
            continue

        build_txt+="cd "+path+"\n"
        build_txt+=python_exec+" gen_cmakelist.py "+" -b "+" ".join(target_branch) +" -B "+build_type+" -s "+install_root+" -tb "+test_branch+ " \n"
        build_txt+="chmod 777 ./make.sh\n"
        build_txt+="chmod 777 ./make_install.sh\n"
        build_txt+="./make.sh\n"
        build_txt+="cd ../../\n"


        install_txt+="cd "+path+"\n"
        install_txt+="sudo ./make_install.sh\n"
        install_txt+="cd ../../\n"

f=open("./make_all.sh","w")
f.write(build_txt)
f.close()

f=open("./install_all.sh","w")
f.write(install_txt)
f.close()
