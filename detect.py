
# from ultralytics import YOLO

# if __name__ == '__main__':

#     # Load a model
#     model = YOLO(model=r'D:\yolov11\runs\train\C2F_Dcnv4_640x640_20241103\weights\best.pt')  
    
#     model.predict(source=r'D:\yolov11\datasets640x640\test\images\77.jpg',
#                   # save=True,
#                   # show=True,
#                   device=0,
#                   )








########################## 测试帧率##########################
"""                                   
C2F_Dcnv4_640x640_20241103     
        0-30: 14
        30-40: 113
        40-50: 782
        50-60: 265
        60-70: 93
        70-80: 82
        80-90: 60
        90+: 32
                     
C2F_Dcnv4_CSconv_640x640_20241104    
        0-30: 80
        30-40: 241
        40-50: 661
        50-60: 222
        60-70: 141
        70-80: 53
        80-90: 20
        90+: 23       
        
CSconv_640x640_20241103
        0-30: 9
        30-40: 68
        40-50: 203
        50-60: 432
        60-70: 194
        70-80: 158
        80-90: 120
        90+: 257  
        
origin_640x640_20241102
        0-30: 6
        30-40: 38
        40-50: 172
        50-60: 227
        60-70: 343
        70-80: 191
        80-90: 159
        90+: 300
    
origin_1280x1048_20241101   
        0-30: 2
        30-40: 13
        40-50: 72
        50-60: 167
        60-70: 586
        70-80: 161
        80-90: 151
        90+: 289    
        
C2f_FasterBlock_640x640_20241104
        0-30: 5
        30-40: 89
        40-50: 229
        50-60: 638
        60-70: 268
        70-80: 83
        80-90: 47
        90+: 82         
"""

import cv2
from ultralytics import YOLO
import time

# 加载模型
model = YOLO(model=r'D:\yolov11\runs\train\C2f_FasterBlock_640x640_20241104\weights\best.pt')
# 打开视频文件
video_path = 'D://yolov11//runs//test.mp4'
cap = cv2.VideoCapture(video_path)
# print(cap.isOpened())
# 获取视频帧的维度
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
#创建VideoWriter对象
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('D://yolov11//runs//output2.mp4', fourcc, 20.0, (frame_width, frame_height))

# 初始化最小和最大帧率
fps_all = []
frametime_counts = []
fps_counts = {
    '0-30': 0,
    '30-40': 0,
    '40-50': 0,
    '50-60': 0,
    '60-70': 0,
    '70-80': 0,
    '80-90': 0,
    '90+': 0,
}
#循环视频帧
while cap.isOpened():
    # 读取某一帧
    success, frame = cap.read()
    # print('success:', success)
    if success:
        # 使用yolov8进行预测
        #可视化结果
        # annotated_frame = results[0].plot()
        #将带注释的帧写入视频文件
        # out.write(annotated_frame)
        # 计算总处理时间
        start_frame_time = time.time()  # 记录开始时间
        results = model(frame, device=0)
        end_frame_time = time.time()  # 记录结束时间
        
        # 计算单帧处理时间
        frame_time = end_frame_time - start_frame_time
        if frame_time > 0:
            fps = 1 / frame_time  # 计算当前帧率
            frametime_counts.append(frame_time)
            fps_all.append(fps)
       
    else:
        # 最后结尾中断视频帧循环
        break
    
#释放读取和写入对象
cap.release()
# out.release()


# 计算FPS并统计 
for fps in fps_all:
    # 分类并增加计数
    if fps < 30:
        fps_counts['0-30'] += 1
    elif 30 <= fps < 40:
        fps_counts['30-40'] += 1
    elif 40 <= fps < 50:
        fps_counts['40-50'] += 1
    elif 50 <= fps < 60:
        fps_counts['50-60'] += 1
    elif 60 <= fps < 70:
        fps_counts['60-70'] += 1
    elif 70 <= fps < 80:
        fps_counts['70-80'] += 1
    elif 80 <= fps < 90:
        fps_counts['80-90'] += 1
    else:
        fps_counts['90+'] += 1

# 输出结果
for range, count in fps_counts.items():
    print(f"{range}: {count}")