from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

#model.train(data='/home/aimlgpu2/SARAN_cv/crop_detection_yolov8/banana_flowering/bananaflowering.yaml',batch=32, epochs=300, imgsz=640,name="crop_detection_bananaflowering_interns_v8x_seg_ep_300_bsize_32_jul_31_test5")
#model.train(data='/home/aimlgpu2/SARAN_cv/crop_detection_yolov8/Banana_500_split/vidhya_banana.yaml',batch=0.90, epochs=15, imgsz=640,name="sep_18_vidhya500_banana_new_withoutbg",lr0=0.001,optimizer="AdamW",lrf=0.001,momentum=0.937)
# ^ gave 66percent accuracy for nmonia 
#model.tune(data="/home/aimlgpu2/SARAN_cv/crop_detection_yolov8/banana_vegetation_output/bananavegetation.yaml", epochs=300, iterations=300, optimizer="AdamW", plots=False, save=False, val=False,name="cd_bananavegetation_interns_v8x_seg_ep_300_bsize_32_aug_6")
model.train(data='/home/aimlgpu2/SARAN_cv/crop_detection_yolov8/Banana_500_split/vidhya_banana.yaml',batch=0.90, epochs=300, imgsz=640,name="sep_18_vidhya500_banana_new_withoutbg_300")
