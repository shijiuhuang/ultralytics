
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\yolov11\runs\train\exp2\weights\best.pt')  
    
    model.predict(source=r'D:\yolov11\datasets\val\images\37.png',
                #   save=True,
                #   show=True,
                  device=0,
                  )

