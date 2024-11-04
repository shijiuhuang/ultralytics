
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r'D:\yolov11\ultralytics\ultralytics\yolo11n-pose.pt')
    model = YOLO(model=r'D:\yolov11\ultralytics\ultralytics\cfg\models\11\yolo11-pose_remove0-2.yaml')
    model.load(r'D:\yolov11\ultralytics\ultralytics\yolo11n-pose.pt')
    model.train(data=r'D:\yolov11\ultralytics\ultralytics\coco-data.yaml',
                task='pose',
                mode='train',
                imgsz=640,
                
                epochs=100,
                batch=16,
                device=0,
                workers=8,
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,                     
                mixup=0.0,
                int8=False,
                degrees=5,
                hsv_v=0.5,
                )
