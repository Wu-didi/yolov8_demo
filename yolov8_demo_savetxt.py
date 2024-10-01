from ultralytics import YOLO
import os

def process_yolo_result(results):
    id2class = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', \
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', \
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', \
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', \
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',\
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', \
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', \
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    
    result_boxes = []
    for r in results:
        
        if r.boxes is None:
            continue
        
        # 这里的results是以图像维度，表示长度，每一个r表示一个image的
        for single_box in r.boxes:
            
            # print(single_box)
            coords_carm = None
            
            # Check if the center of the box is within the specified rectangular region
            if 100 <= single_box.xywh[0][0].cpu().numpy() <= 1200 and \
                630 <= single_box.xywh[0][1].cpu().numpy()  <= 700:
                continue  # Ignore the box if its center is within the specified region
            if single_box.cls == 0 or single_box.cls == 1:
                class_index = 0
                # print('detect is person')
                # 获取person pix high
                x = single_box.xywhn[0][0].cpu().numpy()
                y = single_box.xywhn[0][1].cpu().numpy()
                w = single_box.xywhn[0][2].cpu().numpy()
                h = single_box.xywhn[0][3].cpu().numpy()
                
                result_boxes.append([class_index, x, y, w, h])
                
            elif single_box.cls == 2 or single_box.cls == 3  :
                class_index = 1
                # print('detect is person')
                # 获取person pix high
                x = single_box.xywhn[0][0].cpu().numpy()
                y = single_box.xywhn[0][1].cpu().numpy()
                w = single_box.xywhn[0][2].cpu().numpy()
                h = single_box.xywhn[0][3].cpu().numpy()
                
                result_boxes.append([class_index, x, y, w, h])
            
            elif single_box.cls == 5:
                class_index = 2
                # print('detect is person')
                # 获取person pix high
                x = single_box.xywhn[0][0].cpu().numpy()
                y = single_box.xywhn[0][1].cpu().numpy()
                w = single_box.xywhn[0][2].cpu().numpy()
                h = single_box.xywhn[0][3].cpu().numpy()
                
                result_boxes.append([class_index, x, y, w, h])
            
            elif single_box.cls == 7:
                class_index = 3
                # print('detect is person')
                # 获取person pix high
                x = single_box.xywhn[0][0].cpu().numpy()
                y = single_box.xywhn[0][1].cpu().numpy()
                w = single_box.xywhn[0][2].cpu().numpy()
                h = single_box.xywhn[0][3].cpu().numpy()
                
                result_boxes.append([class_index, x, y, w, h]) 
                
            else:
                print('detect is others')
    return result_boxes

def write_detections_to_txt(save_label_root, image_name, detections):
    # 提取图像名称的基础部分，去掉扩展名  
    base_name = os.path.splitext(image_name)[0]  
    # 创建与图像同名的txt文件  
    txt_file_name = f"{base_name}.txt"  
    txt_path = os.path.join(save_label_root, txt_file_name)
    with open(txt_path, 'w') as f:  
        # 遍历检测结果列表  
        for detection in detections:  
            # 检测结果的格式应为 [class_index, x, y, w, h]  
            class_idx, x, y, w, h = detection  
            # 将结果转换为字符串并写入文件，每个结果占一行  
            # 假设x, y, w, h已经是相对于图像宽高的归一化值  
            line = f"{int(class_idx)} {x} {y} {w} {h}\n"  
            f.write(line)  


# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')
# Run inference on 'bus.jpg' with arguments '/home/wudi/python_files/onsite/0519update/e2e/total_results'
image_list = os.listdir('/home/wudi/python_files/onsite/0519update/e2e/total_results')
save_label_root = './labels'
# 如果存在就删除
if os.path.exists(save_label_root):
    os.system(f'rm -rf {save_label_root}')
os.makedirs(save_label_root)

for image_name in image_list:
    print(image_name)
    image_path = os.path.join('/home/wudi/python_files/onsite/0519update/e2e/total_results', image_name)
    result = model.predict(image_path, save=True, conf=0.5,save_txt=True)
    result_boxes = process_yolo_result(result)
    write_detections_to_txt('./labels', image_name, result_boxes)
