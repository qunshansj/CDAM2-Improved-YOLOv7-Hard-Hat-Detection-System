
# COCO 2017 dataset http://cocodataset.org
 
# download command/URL (optional)
# download: bash ./scripts/get_coco.sh
 
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: '/home/devuser/pan/yolov7/datasets/VOC2028/SafetyHelmet/images/train2028'  
val: '/home/devuser/pan/yolov7/datasets/VOC2028/SafetyHelmet/images/val2028'  
#test: ./coco/test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794
 
# number of classes
nc: 2
 
# class names
names: ['hat','person']

