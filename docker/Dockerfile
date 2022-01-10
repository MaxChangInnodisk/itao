# ----------------------------------------------------------
# 底層環境是吃 tensorrt:20.11-py3
FROM nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.08-py3

COPY ./docker/requirements.sh ./requirements.sh
COPY ./assets/env_tools /tools
# 由於 安裝套件有互動界面會導致報錯 所以先行關閉
ENV DEBIAN_FRONTEND noninteractive
RUN chmod +x ./requirements.sh \
&& ./requirements.sh tensorrt-dev  \
&& rm ./requirements.sh \
&& echo "alias check-env=/tools/check_env_tool " >> ~/.bashrc
ENV PATH="$PATH:/usr/games"
ENV export LC_ALL=C 
CMD /tools/hello eco && /bin/bash