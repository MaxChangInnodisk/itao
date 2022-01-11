# ----------------------------------------------------------
# 底層環境是吃 tensorrt:20.11-py3
FROM nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.08-py3

# 複製檔案
COPY ["./docker/requirements.sh", "./assets/env_tools", "/"]

# 由於 安裝套件有互動界面會導致報錯 所以先行關閉
ENV DEBIAN_FRONTEND noninteractive
RUN chmod +x /requirements.sh \
&& /requirements.sh tensorrt-dev \
&& rm /requirements.sh \
&& echo "alias check-env=/check_env_tool " >> ~/.bashrc \
&& echo "alias hello=/hello " >> ~/.bashrc

# 環境變數相關
ENV PATH="$PATH:/usr/games" \
    LC_ALL=C 

# 透過 ENTRYPOINT 洗掉之前 IMAGE 定義的指令
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD [ "/hello" ]