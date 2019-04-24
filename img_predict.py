import onnxruntime
import numpy as np
import cv2
import argparse

#80个类的标签
label = ["background", "person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

#anchors
anchors_yolo = [[(116,90),(156,198),(373,326)],[(30,61),(62,45),(59,119)],[(10,13),(16,30),(33,23)]]
anchors_yolo_tiny = [[(81,82),(135,169),(344,319)],[(10,14),(23,27),(37,58)]]

#预处理图像,返回原尺寸图像,原尺寸,处理后图像数组
def process_image(img_path,input_shape):
    ori_img = cv2.imread(img_path)
    print("origin image size:",ori_img.shape)
    img = cv2.resize(ori_img, input_shape)
    print("after modify img data shape:",img.shape)
    image =  img[:,:,::-1].transpose((2,0,1))
    image = image[np.newaxis,:,:,:]/255
    image = np.array(image,dtype=np.float32)
    return ori_img,ori_img.shape,image

#sigmoid函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-1*x))
    return s

#获取分数最高的类别,返回分数和索引
def getMaxClassScore(class_scores):
    class_score = 0
    class_index = 0
    for i in range(len(class_scores)):
        if class_scores[i] > class_score:
            class_index = i+1
            class_score = class_scores[i]
    return class_score,class_index

#置信度阈值筛选得到bbox
def getBBox(feat,anchors,image_shape,confidence_threshold):
    box = []
    for i in range(len(anchors)):
        for cx in range(feat.shape[0]):
            for cy in range(feat.shape[1]):
                tx = feat[cx][cy][0 + 85 * i]
                ty = feat[cx][cy][1 + 85 * i]
                tw = feat[cx][cy][2 + 85 * i]
                th = feat[cx][cy][3 + 85 * i]
                cf = feat[cx][cy][4 + 85 * i]
                cp = feat[cx][cy][5 + 85 * i:85 + 85 * i]

                bx = (sigmoid(tx) + cx)/feat.shape[0]
                by = (sigmoid(ty) + cy)/feat.shape[1]
                bw = anchors[i][0]*np.exp(tw)/image_shape[0]
                bh = anchors[i][1]*np.exp(th)/image_shape[1]

                b_confidence = sigmoid(cf)
                b_class_prob = sigmoid(cp)
                b_scores = b_confidence*b_class_prob
                b_class_score,b_class_index = getMaxClassScore(b_scores)

                if b_class_score > confidence_threshold:
                    box.append([bx,by,bw,bh,b_class_score,b_class_index])
    return box

#非极大值抑制阈值筛选得到bbox
def donms(boxes,nms_threshold):
    b_x = boxes[:, 0]
    b_y = boxes[:, 1]
    b_w = boxes[:, 2]
    b_h = boxes[:, 3]
    scores = boxes[:,4]
    areas = (b_w+1)*(b_h+1)
    order = scores.argsort()[::-1]
    keep = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(b_x[i], b_x[order[1:]])
        yy1 = np.maximum(b_y[i], b_y[order[1:]])
        xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
        yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
        #相交面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #相并面积,面积1+面积2-相交面积
        union = areas[i] + areas[order[1:]] - inter
        # 计算IoU：交 /（面积1+面积2-交）
        IoU = inter / union
        # 保留IoU小于阈值的box
        inds = np.where(IoU <= nms_threshold)[0]
        order = order[inds + 1]  # 因为IoU数组的长度比order数组少一个,所以这里要将所有下标后移一位

    final_boxes = [boxes[i] for i in keep]
    return final_boxes

#绘制预测框
def drawBox(boxes,img,img_shape):
    for box in boxes:
        x1 = int((box[0]-box[2]/2)*img_shape[1])
        y1 = int((box[1]-box[3]/2)*img_shape[0])
        x2 = int((box[0]+box[2]/2)*img_shape[1])
        y2 = int((box[1]+box[3]/2)*img_shape[0])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, label[int(box[5])]+":"+str(round(box[4],3)), (x1+5,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(label[int(box[5])]+":"+str(box[4]))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#获取预测框
def getBoxes(prediction,anchors,img_shape,confidence_threshold,nms_threshold):
    boxes = []
    for i in range(len(prediction)):
        feature_map = prediction[i][0].transpose((2, 1, 0))
        box = getBBox(feature_map, anchors[i], img_shape, confidence_threshold)
        boxes.extend(box)
    Boxes = donms(np.array(boxes),nms_threshold)
    return Boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_shape', help="caffe's caffemodel file path", nargs='?',default=(416,416))
    parser.add_argument('img_path', help="test image path", nargs='?', default="yourpath/onnx_yolov3/img/dog.jpg")
    parser.add_argument('onnx_path', help="onnx model file path", nargs='?', default="yourpath/onnx_yolov3/onnxmodel/yolov3-416.onnx")
    parser.add_argument('cf_thres', help="confidence threshold", nargs='?', default=0.25, type=float)
    parser.add_argument('nms_thres', help="nms threshold", nargs='?', default=0.6, type=float)
    parser.add_argument('--isTiny',help="to choice anchors",nargs='?', default=False, type=bool)
    args = parser.parse_args()


    if args.isTiny:
        print("use yolov3-tiny's anchor")
        anchors = anchors_yolo_tiny
    else:
        print("use yolov3's anchor")
        anchors = anchors_yolo
    input_shape = args.input_shape #模型输入尺寸
    confidence_threshold = args.cf_thres
    nms_threshold = args.nms_thres
    img_path = args.img_path
    onnx_path = args.onnx_path
    print("image path:",img_path)
    print("onnx model path:",onnx_path)

    img,img_shape,TestData = process_image(img_path,input_shape)
    session = onnxruntime.InferenceSession(onnx_path)
    inname = [input.name for input in session.get_inputs()][0]
    outname = [output.name for output in session.get_outputs()]

    print("inputs name:",inname,"|| outputs name:",outname)
    prediction = session.run(outname, {inname:TestData})
    boxes = getBoxes(prediction,anchors,input_shape,confidence_threshold,nms_threshold)
    drawBox(boxes,img,img_shape)


if __name__ == '__main__':
    main()





