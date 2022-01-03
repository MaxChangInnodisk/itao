# itao
`iTAO` is the GUI version for NVIDIA TAO Toolkit.

## Pre-requisite
1. [Docker](https://max-c.notion.site/Install-Docker-9a0927c9b8aa4455b66548843246152f)
2. [NVIDIA-Container-Toolkit](https://max-c.notion.site/Install-NVIDIA-Container-Toolkit-For-Docker-7db1728db09e4378871303ae6c616401)
3. [NGC API Key](https://max-c.notion.site/Get-NVIDIA-NGC-API-Key-911f9d0a5e1147bf8ad42f3c0c8ca116)
4. base on virtualenv and virtualenvwrapper

## Build
auto install virtualenv and create a new virtual environment (tao-test).
```bash
./itao.sh build
```

## Run
activate "tao-test" and startup iTAO demo.
```bash
./itao.sh run
```

## Debug Mode
any actions which spending a lot time in the NVIDIA TAO Toolkit won't be enabled.
```bash
./itao.sh run debug
```

## Developer Mode
please check `README-DEV.md` ...


## Future Work
* feature
  - [x] tab1 - init
  - [x] tab2 - train % eval
  - [x] tab3 - prune & retrain
  - [x] tab4 - infer & export
  - [x] generate log file
  - [x] document for developer ( README-DEV.md )
  - [ ] make "evaluation" as optional item.

* tasks
  - [x] classification
  - [ ] objected detection ( YoloV4 )
  - [ ] instance segmentation ( UNet ? not sure )


