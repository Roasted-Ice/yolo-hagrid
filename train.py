from ultralytics import YOLO
import torch


if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('E:\Gp\GP_V1\YoloPretrainedModel\yolov8s.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    torch.backends.cudnn.enabled = False
    # Train the model
    results = model.train(data='E:\Gp\GP_V1\Code\Hands_Gesture\yolodataset\dataset.yaml', epochs=100,imgsz=416, batch=8)
