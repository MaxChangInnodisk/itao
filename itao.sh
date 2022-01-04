#!/bin/bash

MODE=$1
DEBUG_PAGE=$2
DEBUG_OPT=$3
BASHRC=$(realpath ~/.bashrc)
WORKON_HOME=$(realpath ~/Envs)
LS_INFO=("export WORKON_HOME=${WORKON_HOME}","export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3")
TRG_ENV="tao-test"
CREATE_ENV=1

function log() {
    now=$(date +"%T")
    echo -e "[$now] $@"
}

function help(){
    echo "---------------------------------------"
    echo "$ ./itao.sh [OPT]"
    echo ""
    echo "[OPT]"
    echo "build     build itao environment."
    echo "run       run itao with QT window."
    echo "---------------------------------------"
}

if [[ -z ${MODE} ]] || [[ ${MODE} == "help" ]];then
    help
    exit 1
    
elif [[ ${MODE} == "build" ]];then

    log "Start building iTAO ..."

    log "Checking virtualenv ..."
    if [[ -z $(pip3 list | grep virtualenv) ]];then

        log "Installing virtualenv and virtualenvwrapper ..."
        pip3 install virtualenv 
        pip3 install virtualenvwrapper 

        log "Seting up virtual enviroment ..."
        for INFO in LS_INFO;do
            if [[ -z "$(cat ${BASHRC} | grep ${INFO})" ]];then
                echo $INFO >> $BASHRC
            fi
        done
        source $BASHRC

    else
        log "Checking enviroment ..."

        source `which virtualenvwrapper.sh`
        for ENV in $(lsvirtualenv);do
            if [[ ${ENV} == ${TRG_ENV} ]]; then CREATE_ENV=0; fi
        done

        if [[ ${CREATE_ENV} -eq 1 ]];then
            log "Create new enviroment ..."
            mkvirtualenv ${TRG_ENV} -p $(which python3)
        fi

        log "Launch ${TRG_ENV} & checking python's packages ('nvidia-pyindex', 'nvidia-tao') ..."
        workon ${TRG_ENV}
        if [[ -z $(pip3 list --disable-pip-version-check | grep pyindex) ]];then pip3 install nvidia-pyindex -q --disable-pip-version-check;fi
        if [[ -z $(pip3 list --disable-pip-version-check | grep tao) ]];then pip3 install nvidia-tao==0.1.19 -q --disable-pip-version-check;fi
        pip3 install numpy PyQt5 matplotlib pyqtgraph -q --disable-pip-version-check wget
        if [[ -z $(pip3 list --disable-pip-version-check | grep tao) ]];then log "Testing TAO ... Done";fi

        # dependy for PyQt5 on Ubuntu
        # apt-get install -y libxcb-xinerama0

        # TAO_ZIP="cv_samples_v1.2.0.zip"
        # TASK_ROOT="tasks"
        # if [[ ! -d ${TASK_ROOT} ]];then
        #     log "Downing load specifications of tao_v1.2.0 ..."
        #     wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/tao/cv_samples/versions/v1.2.0/zip -O ${TAO_ZIP} > /dev/null 2>&1
        #     unzip -u ${TAO_ZIP} -d ${TASK_ROOT} > /dev/null 2>&1
        #     rm -rf ${TAO_ZIP} 
        # fi

        DATA_ROOT="${TASK_ROOT}/data"
        if [[ ! -d ${DATA_ROOT} ]];then
        log "Create data folder in ${TASK_ROOT}"
            mkdir ${DATA_ROOT}
        fi
        
        log "Done"

        source ./itao.sh run
    fi

elif [[ ${MODE} == 'run' ]];then
    log "Start training AI with iTAO."
    source `which virtualenvwrapper.sh`
    log "Launch virtualenv ${TRG_ENV}"
    workon ${TRG_ENV}
    python3 ./demo
elif [[ ${MODE} == 'debug' ]];then
    log "Start training AI with iTAO ( debug mode )."
    source `which virtualenvwrapper.sh`
    log "Launch virtualenv ${TRG_ENV}"
    workon ${TRG_ENV}
    log "Check arguments ... "
    if [[ -z ${DEBUG_PAGE} || -z ${DEBUG_OPT} ]];then
        log "DEBUG MODE: $./itao.sh debug <DEBUG_PAGE:1, 2, 3, 4> <DEBUG_OPT:train, eval, prune, retrain, export, infer>"
        exit 1
    fi
    python3 ./demo ${MODE} ${DEBUG_PAGE} ${DEBUG_OPT}

fi