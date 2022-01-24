# itao
`iTAO` is the GUI version for NVIDIA TAO Toolkit.

## Pre-requisite
1. [Docker](https://max-c.notion.site/Install-Docker-9a0927c9b8aa4455b66548843246152f)
2. [NVIDIA-Container-Toolkit](https://max-c.notion.site/Install-NVIDIA-Container-Toolkit-For-Docker-7db1728db09e4378871303ae6c616401)
3. [NGC API Key](https://max-c.notion.site/Get-NVIDIA-NGC-API-Key-911f9d0a5e1147bf8ad42f3c0c8ca116)
4. base on virtualenv and virtualenvwrapper

## Build
auto build the docker image
```bash
./itao.sh build
```

## Run
activate the container with iTAO.
```bash
./itao.sh run
```

## Debug Mode
You can enable target option for debug. DEBUG_PAGE means the page from 1 to 4 and DEBUG_OPT is the feature in DEBUG_PAGE (e.g. train, kmeans, eval, etc.)
```bash
# Make sure you are in docker container
python3 demo --docker --debug --page <DEBUG_PAGE> --opt <DEBUG_OPT>
```

## Developer Mode
please check `README-DEV.md` ...


## Future Work
* feature
  - [x] Generate log file
  - [x] Document for developer ( README-DEV.md )
  - [x] Full screen mode
  - [ ] Make "evaluation" as optional item.
  - [ ] Checkpoint
  - [ ] Select GPU

* tasks
  - [x] classification:resnet
  - [x] objected detection ( YoloV4 ):resnet
  - [ ] instance segmentation ( UNet ? not sure )
