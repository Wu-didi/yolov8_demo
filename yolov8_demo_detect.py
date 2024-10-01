from ultralytics import YOLO
import cv2
import time

'''
names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', \
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', \
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', \
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', \
                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',\
                    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', \
                        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', \
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
'''

# 相机焦距
foc  = 750
# 预设置行人高度inch
real_hight_person_inch = 66.9
# 预设置车辆高度inch
real_hight_car_inch = 57.08

# 单目测量距离，通过相似三角形
def get_distance(real_hight_inch, h):
    '''返回真的distance'''
    dis_inch = real_hight_inch * foc / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def pixel_to_camera_coords(x, y, z):  
    """  
    将图像坐标系下的像素坐标（x, y）和深度z（单位为m）转换为相机坐标系下的坐标（Xc, Yc, Zc）。  
  
    参数:  
    x, y (int): 图像坐标系下的像素坐标。  
    z (float): 深度值，即相机坐标系下的Z坐标（单位为m）。  
    fx, fy (float): 相机内参矩阵的焦距。  
    cx, cy (float): 相机内参矩阵的主点坐标。  
  
    返回:  
    Xc, Yc, Zc (float): 相机坐标系下的三维坐标。  
    """  
    
    # 相机内参  
    fx = 733.614441  
    fy = 735.059326  
    cx = 604.237122  
    cy = 339.834991  
    
    
    # 将像素坐标转换为归一化相机坐标  
    xc = (x - cx) / fx  
    yc = (y - cy) / fy  
    # 归一化相机坐标的Z分量就是输入的深度z  
    zc = z  
    # 输出相机坐标系下的三维坐标  
    Xc = xc * zc  
    Yc = yc * zc  
    Zc = zc  
    return Xc, Yc, Zc  


# 定义一个绘制矩形框的函数，并将坐标值和类别值传入，写在image上
def draw_rectangle(image, x, y, x2, y2, class_label):
    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return image

# 定义一个显示图像的函数，可以根据传入的图像显示出来，显示视频的时候可以用到
def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    
# Load a model
model = YOLO('yolov8x.pt')  # load an official model


# 记录开始时间  
start_time = time.time()  

# Predict with the model
results = model('cam_0_166.jpg')  # predict on an image
infer_time = time.time()  
elapsed_time = infer_time - start_time  
# 输出运行时间  
print(f"infer took {elapsed_time} seconds to run.")
print(results)
for r in results:
    # 这里的results是以图像维度，表示长度，每一个r表示一个image的
    for single_box in r.boxes:
        if single_box.cls == 0:
            print('detect is person')
            # 获取person pix high
            cx = single_box.xywh[0][0].cpu().numpy()
            cy = single_box.xywh[0][1].cpu().numpy()
            h = single_box.xywh[0][-1].cpu().numpy()
            # 
            real_dist = get_distance(real_hight_person_inch, h)
            print("real distance is ", real_dist)
            
            coords_carm = pixel_to_camera_coords(cx, cy, real_dist)
            
            
        elif single_box.cls == 2:
            print('detect is car')
            # 获取person pix high
            cx = single_box.xywh[0][0].cpu().numpy()
            cy = single_box.xywh[0][1].cpu().numpy()
            h = single_box.xywh[0][-1].cpu().numpy()
            # 
            real_dist = get_distance(real_hight_person_inch, h)
            print("real distance is ", real_dist)
            coords_carm = pixel_to_camera_coords(cx, cy, real_dist)
            print("coords_carm is ", coords_carm)
        # print(r.boxes)
        # print(r.boxes.xywh)
        image = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                                        int(single_box.xyxy[0][1].cpu().numpy()),
                                        int(single_box.xyxy[0][2].cpu().numpy()),
                                        int(single_box.xyxy[0][3].cpu().numpy()),
                                        str(coords_carm))
    show_image(image)
        
    #===========================================================================
    # 可以使用yolo的show函数来显示结果
    # r.plot()  # plot predictions
    # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    # Show results to screen (in supported environments)
    # r.show()
    
    

# im2 = cv2.imread("cam_0_166.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
# print(results)

# 记录结束时间  
end_time = time.time()  
# 计算运行时间  
elapsed_time = end_time - start_time  
# 输出运行时间  
print(f"all infer took {elapsed_time} seconds to run.")


# te[[ol]]