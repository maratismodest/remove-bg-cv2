import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x-seg.pt')  # Ensure the path to the model file is correct

# Function to process an image
def process_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image_orig = image.copy()
    h_or, w_or = image.shape[:2]

    # Resize the image for YOLO model
    image_resized = cv2.resize(image, (640, 640))

    # Perform detection
    results = model(image_resized)[0]

    # Extract the classes and masks
    classes = results.boxes.cls.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    # Create an empty alpha channel
    alpha = np.zeros((h_or, w_or), dtype=np.uint8)

    # Refine and overlay masks on the alpha channel
    for i, mask in enumerate(masks):
        class_name = results.names[int(classes[i])]
        if class_name == 'person':
            resized_mask = cv2.resize(mask, (w_or, h_or), interpolation=cv2.INTER_NEAREST)

            # Apply morphological operations to refine the mask
            kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

            alpha[refined_mask > 0] = 255

    # Create an RGBA image
    b, g, r = cv2.split(image_orig)
    rgba = [b, g, r, alpha]
    result = cv2.merge(rgba, 4)

    # Save the result with transparency
    cv2.imwrite(output_path, result)

# Example usage
process_image('template2.jpg', 'accurate.png')
