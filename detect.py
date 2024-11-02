
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\yolov11\runs\train\origin_640x640_20241102\weights\best.pt')  
    
    model.predict(source=r'D:\yolov11\datasets640x640\test\images\77.jpg',
                #   save=True,
                #   show=True,
                  device=0,
                  )

