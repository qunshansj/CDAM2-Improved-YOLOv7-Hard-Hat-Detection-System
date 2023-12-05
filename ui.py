python

def det_yolov7(info1):
    global model, stride, names, pt, jit, onnx, engine
    if info1[-3:] in ['jpg','png','jpeg','tif','bmp']:
        image = cv2.imread(info1)  # 读取识别对象
        try:
            results = run(model, image, stride, pt)  # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
            for i in results:
                box = i[1]
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                color = [255,0,0]
                if i[0] == 'helmet':
                    color = [0, 0, 255]
                    i[0] = 'NO helmet'
                    ui.printf('警告！检测到工人未戴安全帽')
                if i[0] == 'head':
                    color = [0, 255, 0]
                    i[0] = 'Helmet'
                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(image, str(i[0]) + ' ' + str(i[2])[:5], (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        except:
            pass
        ui.showimg(image)
    if info1[-3:] in ['mp4','avi']:
        capture = cv2.VideoCapture(info1)
        while True:
            _, image = capture.read()
            if image is None:
                break
            try:
                results = run(model, image, stride, pt)  # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
                for i in results:
                    box = i[1]
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    color = [255, 0, 0]
                    if i[0] == 'helmet':
                        color = [0, 0, 255]
                        i[0] = 'NO helmet'
                        ui.printf('警告！检测到

