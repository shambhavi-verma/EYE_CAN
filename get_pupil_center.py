from ultralytics import YOLO
import cv2
import torch

def center_of_box(coords):
    x_min, y_min, x_max, y_max = coords[:4]  # Extracting the coordinates
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return (int(center_x), int(center_y))

def init():
    model_path = "models/v2_custom.pt"  # Path to the model file
    model = YOLO(model_path)

    return model

def predict_center(model, image):
    # Check if CUDA is available and move the model to GPU
    if torch.cuda.is_available():
        model.to('cuda')
        print("Model moved to GPU.")
    else:
        print("CUDA is not available. Using CPU.")


    # Perform inference on the image
    results = model(image)
    
    if len(results[0].boxes.data.cpu().numpy().tolist()) >= 1:
        # Return the first result (which contains the detections and other information)
        result = results[0].boxes.data.cpu().numpy().tolist()[0]
        
        center_coords = center_of_box(result)

        return center_coords
    else:
        print(f"No detection !!")
        return None

if __name__ == "__main__":
    # Example usage
    model = init()
    image_path = 'pupil.png'  # Path to the image

    # Get the detection result for the image
    result = predict_center(model, image_path)
    
    if result:
        img = cv2.imread(image_path)
        cv2.circle(img, result, 10, (0, 255, 0), 2)
        cv2.imshow("YOLO Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No center detected.")
