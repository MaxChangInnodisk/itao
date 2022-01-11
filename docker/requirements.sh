#!/bin/bash
REST='\e[0m';
GREEN='\e[0;32m';
BGREEN='\e[7;32m';
BRED='\e[7;31m';
Cyan='\033[0;36m';
BCyan='\033[7;36m'

printd(){            
    
    if [ -z $2 ];then COLOR=$REST
    elif [ $2 = "G" ];then COLOR=$GREEN
    elif [ $2 = "R" ];then COLOR=$BRED
    elif [ $2 = "Cy" ];then COLOR=$Cyan
    elif [ $2 = "BCy" ];then COLOR=$BCyan
    else COLOR=$REST
    fi

    echo -e "${COLOR}$1${REST}"
}

# if [[ -z ${OPT} ]];then
#     echo "Error input ${OPT}, please enter argument:"
#     printf "%s" "Choose environment's dependencies to install [ tensorrt-dev/deepstream-base/deepstream-dev ]: "
#     read OPT
# fi

echo 
OPT=$1
case ${OPT} in
    "tensorrt-dev")

        printd "Install Develop's Dependencies of TensorRT" BCy

        printd "Update ... \c" Cy
        apt-get update -qq
        apt-get install -qy figlet boxes > /dev/null 2>&1
        echo "Done"

        printd "Install Dependencies of OpenCV ... \c" Cy
        apt-get install -qy ffmpeg libsm6 libxext6 > /dev/null 2>&1
        echo "Done"

        # TLT Converter
        printd "Install Dependencies of TLT Converter ... \c" Cy
        apt-get install -qqy libssl-dev > /dev/null 2>&1
        echo 'export TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu' >> ~/.bashrc 
        echo 'TRT_INC_PATH=/usr/include/x86_64-linux-gnu' >> ~/.bashrc 
        echo "Done"

        # Python Packages
        printd "Install Some Python Packages ... \c" Cy
        pip3 install --upgrade -q pip > /dev/null 2>&1
        pip3 install -q opencv-python tqdm colorlog > /dev/null 2>&1
        echo -e "Done${REST}"

        printd "Install Dependencies of iTAO ... \c" Cy
        apt-get -y update && apt-get install -y libxcb-xinerama0 qt5-default
        if [[ -z $(pip3 list --disable-pip-version-check | grep pyindex) ]];then pip3 install nvidia-pyindex -q --disable-pip-version-check;fi
        if [[ -z $(pip3 list --disable-pip-version-check | grep tao) ]];then pip3 install nvidia-tao==0.1.19 -q --disable-pip-version-check;fi        
        pip3 --disable-pip-version-check install PyQt5 pyqtgraph wget GPUtil
        # 
        echo -e "Done${REST}"
        ;;

    "deepstream-base")    

        printd "Install Basic Dependencies of Deep Stream" BCy
        
        printd "Update ... \c" Cy
        apt-get update -qq > /dev/null 2>&1
        apt-get install -qy wget curl tree > /dev/null 2>&1
        echo "Done"

        printd "Install Gstreamer ... \c" Cy
        # 安裝 gstreamer 會因為互動界面而報錯 
        # 需要在 Dockerfile 中添加 環境變數 ENV DEBIAN_FRONTEND noninteractive
        apt-get install -y -qq libssl1.0.0 \
        libgstreamer1.0-0 gstreamer1.0-tools \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        libgstrtspserver-1.0-0 libjansson4 > /dev/null 2>&1
        echo -e "Done ${REST}"
        ;;

    "deepstream-dev")

        printd "Install Develop's Dependencies of Deep Stream " BCy

        printd "Update ... \c" Cy
        apt-get update -qq > /dev/null 2>&1
        apt-get install -qqy python3-pip figlet boxes lolcat > /dev/null 2>&1
        
        echo 'export PATH="$PATH:/usr/games"' >> ~/.bashrc
        echo 'export LC_ALL=C' >> ~/.bashrc 
        pip3 install --upgrade pip > /dev/null 2>&1
        echo "Done"

        printd "Install Dependencies of OpenCV ... \c" Cy
        apt-get install -qy ffmpeg libsm6 libxext6 > /dev/null 2>&1
        echo "Done"
        
        printd "Install Some Python Packages ... \c" Cy
        pip3 install -q opencv-python argparse colorlog > /dev/null 2>&1
        echo -e "Done ${REST}"
        ;;
esac