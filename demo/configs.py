OPT = {
    "Image Classification":{
        "ResNet":{"ResNet"}, 
        # "VGG":{"VGG"}, 
        # "GoogleNet":{"GoogleNet"}, 
        # "AlexNet":{"AlexNet"},
        "MobileNet":{"MobileNet"}
    },
    "Object Detection":{
        # "DetectNet_v2":{"ResNet", "VGG", "GoogleNet", "AlexNet"}, 
        # "SSD":{"ResNet", "VGG", "GoogleNet", "AlexNet"}, 
        # "YOLO_v3":{"ResNet", "VGG", "EfficientNet", "MobileNet", "DarkNet", "CSPDarkNet"}, 
        "YOLO_v4":{"ResNet", "VGG", "EfficientNet", "MobileNet", "DarkNet", "CSPDarkNet"}, 
    },
    # "Semantic Segmentation":{
    #     "ResNet":{"ResNet"}, 
    #     "VGG":{"VGG"}, 
    #     "GoogleNet":{"GoogleNet"}, 
    #     "AlexNet":{"AlexNet"}
    # },
    # "Other":{
    #     None
    # }
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
