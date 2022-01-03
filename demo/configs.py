OPT = {
    "Image Classification":{
        "ResNet":{"ResNet"}, 
        "VGG":{"VGG"}, 
        "GoogleNet":{"GoogleNet"}, 
        "AlexNet":{"AlexNet"}
    },
    "Object Detection":{
        "DetectNet_v2":{"ResNet", "VGG", "GoogleNet", "AlexNet"}, 
        "SSD":{"ResNet", "VGG", "GoogleNet", "AlexNet"}, 
        "YOLO_v3":{"ResNet", "VGG", "EfficientNet", "MobileNet", "DarkNet", "CSPDarkNet"}, 
        "YOLO_v4":{"ResNet", "VGG", "EfficientNet", "MobileNet", "DarkNet", "CSPDarkNet"}, 
    },
    "Semantic Segmentation":{
        "ResNet":{"ResNet"}, 
        "VGG":{"VGG"}, 
        "GoogleNet":{"GoogleNet"}, 
        "AlexNet":{"AlexNet"}
    },
    "Other":{
        None
    }
}

ARCH_LAYER= {
    "ResNet":["10", "18", "50"],
    "VGG":["16", "19"],
    "GoogleNet":["Default"],
    "AlexNet":["Default"],
    "MobileNet":["_v1","_v2"],
    "EfficientNet":["_b1_swish", "b1_relu"],
    "DarkNet":["19", "53"],
    "CSPDarkNet":["_tiny", "53", "101"]
}

TRAIN_CONF = {
    "key":"nvidia_tlt",
    "task":"",
    "model":"",
    "backbone":"",
    "nlayer":"",
    "dataset_path":"",
    "checkpoint":"",
    "output_name":"",
    "input_shape":"",
    "epoch":"",
    "batch_size":"",
    "learning_rate":"",
    "custom":""
}

PRUNE_CONF = {
    "key":"",
    "thres":"",
    "input_model":"",
    "output_model":""
}

INFER_CONF = {
    
}

EXPORT_CONF = {
    
}

RETRAIN_CONF = {
    "key":"",
    "backbone":"",
    "epoch":"",
    "learning_rate":"",
    "pretrain":"",
    "output_model":"",
    "batch_size":"",
    "custom":""
}
