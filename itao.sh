# ---------------------------------------------------------
function log() {
    now=$(date +"%T")
    echo -e "[$now] $@"
}
# ---------------------------------------------------------
function printd(){

    REST="\e[0m";
    GREEN="\\e[0;32m";   BGREEN="\\e[7;32m";
    RED="\\e[0;31m";     BRED="\\e[7;31m";
    Cyan="\\e[0;36m";  BCyan="\\e[7;36m";
    YELLOW="\\e[33m";
    BLINK="\e[5m";

    COL=$(stty size | cut -d" " -f2);
    SYM="—"
    # printf "${SYM}%.0s" $(seq 1 $COL); printf "\n"
    if [ -z $2 ];then COLOR=$REST
    elif [ $2 = "G" ];then COLOR=$GREEN
    elif [ $2 = "R" ];then COLOR=$RED
    elif [ $2 = "BG" ];then COLOR=$BGREEN
    elif [ $2 = "BR" ];then COLOR=$BRED
    elif [ $2 = "Cy" ];then COLOR=$Cyan
    elif [ $2 = "BCy" ];then COLOR=$BCyan
    elif [ $2 = "Blink" ];then COLOR=$BLINK
    else COLOR=$REST
    fi
    echo
    echo -e "${COLOR}$1${REST}"
}
# ---------------------------------------------------------
function help(){
    echo "Welcom to TAO-ECO" | figlet -k | lolcat
    echo "---------------------------------------"
    echo ""
    echo "$ ./itao.sh [OPT]"
    echo ""
    echo "[OPT]"
    echo "build     build itao environment."
    echo "run       run itao with QT window."
    echo "debug     run debug mode for target feature."
    echo ""
    echo "----------------------------------------"
    echo ""
    echo "$ ./itao.sh debug [DEBUG_PAGE] [DEBUG_OPT]"
    echo ""
    echo "[DEBUG_PAGE] which page you want to debug (from 1 to 4)"
    echo "[DEBUG_OPT]  which option you want to debug (e.g. train, eval, kmeans, etc.)"
    echo ""
    echo "---------------------------------------"
}
# ---------------------------------------------------------
function counting_time(){

    TITLE=$1
    TIMES=$2
    CNT="${TITLE} ... "

    printf "%s" "${CNT}"
    for i in $(seq ${TIMES} -1 1 );do
        echo -e "\r${dict[$i]} ${TITLE} (${i}) \c${RESET}"
        sleep 1
    done
    echo -e "${RESET}START\n"
}
# ---------------------------------------------------------
function build_image(){
    printd "Bulding Docker Image ..." BG
    IMG=$1
    DOCKER=$2

    docker build -t ${IMG} -f ${DOCKER} .
}
# ---------------------------------------------------------
function check_image(){ 
    echo "$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep ${1} | wc -l )"
}
# ---------------------------------------------------------
function check_container(){ 
    echo "$(docker ps -a --format "{{.Names}}" | grep ${1} | wc -l ) "
}
# ---------------------------------------------------------
function run_container(){
    CNT=$1
    IMG=$2
    CAM=$3
    WORK="/workspace"

    echo -e "Searching Container (${CNT}) ... \c"
    export DISPLAY=:0
    `xhost +` > /dev/null 2>&1

    if [[ $( check_container ${CNT} ) -gt 0 ]]; then
        
        echo -e "PASS \n"

        # counting_time "Start to run the container" 3 
        
        docker start ${CNT} > /dev/null 2>&1
        docker attach ${CNT} 
    else 
        echo -e "Failed \n"

        echo -e "Runing a new one ... \n"
        # clear

        docker run --gpus all --name ${CNT} -it \
        -w ${WORK} \
        -v `pwd`:${WORK} \
        -v `realpath ~/.docker/config.json`:/root/.docker/config.json \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=unix$DISPLAY \
        ${IMG}
    fi
}
# ---------------------------------------------------------
printd "Initialize ..." BG
sudo apt-get install figlet boxes lolcat -qqy


while getopts ":h:i" option; do
   case $option in
        h) # display help
            help
            exit;;
        i)
            exit;;
   esac
done
# ---------------------------------------------------------
MODE=$1
CAM=$2

IMG_NAME="itao"
IMG_VER="v0.3"
IMG="${IMG_NAME}:${IMG_VER}"
DOCKER="./docker/Dockerfile"

CNT="itao"
# ---------------------------------------------------------
if [[ -z ${MODE} ]] || [[ ${MODE} == "help" ]];then
    help

elif [[ ${MODE} = "build" ]];then
    # build
    
    build_image ${IMG} ${DOCKER} 

elif [[ ${MODE} = "run" ]];then
    
    printd "Checking Environment " BG

    if [[ $(check_image ${IMG}) -eq 0 ]];then
        build_image ${IMG} ${DOCKER}
    else
        echo "Environment is exists."
    fi

    printd "Activate Environment" BG
    run_container ${CNT} ${IMG} ${CAM}

    exit 
else
    help
fi
