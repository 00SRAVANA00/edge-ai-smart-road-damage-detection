!pip install ultralytics

from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='/content/cracks/data.yaml', epochs=50, imgsz=640)


!ls -F /content/runs/detect/train-2



print("Now, let's proceed with image testing using the best trained model.")

# Load the best trained model
model = YOLO('/content/runs/detect/train-2/weights/best.pt')

# Perform inference on images from the test set
# First, list images in the test set
!ls /content/cracks/test/images/

import glob
from IPython.display import display
from PIL import Image

test_images = glob.glob('/content/cracks/test/images/*')

if test_images:
    print(f"Found {len(test_images)} test images. Displaying predictions for the first 3 (or fewer if not enough are available).")
    # Iterate through the first 3 images (or fewer if not enough)
    for i, test_image_path in enumerate(test_images[:3]): # Select up to the first 3 images
        print(f"\nRunning inference on image {i+1}: {test_image_path}")
        results = model(test_image_path)

        # Show results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            display(im)
else:
    print("No test images found in /content/cracks/test/images/")



# Re-initialize YOLO model and train on the 'potholes' dataset with corrected data.yaml
model_potholes = YOLO('yolov8n.pt')
model_potholes.train(data='/content/potholes/data.yaml', epochs=50, imgsz=640, project='runs/detect', name='potholes_train')




print("Now, let's proceed with image testing using the best trained 'potholes' model.")

# Load the best trained 'potholes' model
model_potholes_best = YOLO('/content/runs/detect/runs/detect/potholes_train-2/weights/best.pt')

# Perform inference on images from the valid set for potholes, as a test set is not available
# First, list images in the potholes valid set
!ls /content/potholes/valid/images/

import glob
from IPython.display import display
from PIL import Image

potholes_test_images = glob.glob('/content/potholes/valid/images/*') # Changed to valid/images

if potholes_test_images:
    print(f"Found {len(potholes_test_images)} validation images. Displaying predictions for the first 3 (or fewer if not enough are available).")
    # Iterate through the first 3 images (or fewer if not enough)
    for i, test_image_path in enumerate(potholes_test_images[:3]): # Select up to the first 3 images
        print(f"\nRunning inference on image {i+1}: {test_image_path}")
        results = model_potholes_best(test_image_path)

        # Show results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            display(im)
else:
    print("No test or validation images found in /content/potholes/valid/images/")








