# iTAO
This documentation is for devoloper.

## Install virtualenv and virtualenvwrapper
```bash
pip3 install virtualenv
pip3 install virtualenvwrapper
source `which virtualenvwrapper.sh`
```

## Create & Activate virtual environment
```bash
mkvirtualenv tao-test -p $(which python3)   # create a new one 
workon tao-test                             # activate 
```

## Install NVIDIA TAO Toolkit and dependences
```bash
pip3 install nvidia-pyindex     # remenber install pyindex first 
pip3 install nvidia-tao         # because the TAO Toolkit dependence on pyindex    

sudo apt-get install -y libxcb-xinerama0    # for PyQt5 in ubuntu 18
pip3 install numpy PyQt5 matplotlib pyqtgraph wget -q --disable-pip-version-check 
```

## Run iTAO
```bash
# user
./itao.sh run

# developer
workon tao-test
python3 demo

# debug mode
# python3 demo --debug --page <TAB_ID> --opt <TAB_OPT>
```

## Detail

```bash
.
├── LICENSE   
├── README.md
├── README-DEV.md       
├── itao.sh             # build and run itao
├── itao.log            # log file which generated when iTAO
├── ui                  # pyqt ui file
├── configs             # create environ in itao
├── demo
│   ├── __init__.py     # make folder of demo as an API
│   ├── app.py          # entry of pyqt demo
│   ├── configs.py      # organize variable train,retrain,eval,prune,infer,export
│   ├── qt_init.py      # init pyqt and all common function in here
│   ├── qt_tab1.py      # any function in tab1
│   ├── qt_tab2.py      # any function in tab2
│   ├── qt_tab3.py      # any function in tab3
│   ├── qt_tab4.py      # any function in tab4
│   └── qt_tabs.py      # organize module from tab1 to tab4
│
├── itao                        # API of iTAO
│   ├── __init__.py             
│   ├── csv_tools.py            # read .csv for result.csv generated after inference
│   ├── dataset_format.py       # provide each dataset's format
│   ├── environ.py              # tool for setting up environ
│   ├── spec_tools.py           # tool for mapping specification's content
│   ├── qtasks                  
│   │   ├── __init__.py         
│   │   ├── install_ngc.py      # install ngc cli
│   │   ├── download_model.py   # download NGC's pre-trained model
│   │   ├── stop_tao.py         # stop all container of TAO
│   │   ├── eval.py             # do evaluation
│   │   ├── export.py           # do export
│   │   ├── inference.py        # do inference
│   │   ├── prune.py            # do prune
│   │   ├── retrain.py          # do retrain
│   │   └── train.py            # do train
│   └── utils
│       ├── __init__.py
│       └── qt_logger.py        # setup logger with name 'dev'
│
└── tasks               # work folder for iTAO ( NVIDIA sample code in here too )
    ├── classification  # default of classification
    ├── data            # place your dataset in here
    :                   
    └── yolo_v4         # default of objected detection
```

## iTAO API Refernece
🚧🚧🚧 under construction 🚧🚧🚧

## Add New Task
