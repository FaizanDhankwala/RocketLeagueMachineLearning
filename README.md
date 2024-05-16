# Rocket League Object Detection!
![image](https://github.com/FaizanDhankwala/RocketLeagueMachineLearning/assets/55712375/b483dccc-cf6d-421d-9edf-e25503fefb1e)


*This program was all done with the use of Python and Machine Learning (TensorFlow).

*The entire program can be broken down into 4 parts

# PART 1: GATHERING THE DATASET
```
import numpy as np
import win32gui, win32ui, win32con
from PIL import Image
from time import sleep
import os
class WindowCapture:
    w = 0
    h = 0
    hwnd = None

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[...,:3]
        img = np.ascontiguousarray(img) 
            
        return img

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while(True):
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpg")
            sleep(0.3)
    
    def get_window_size(self):
        return (self.w, self.h)
# Execute this cell to generate a dataset of images for the specified window.

window_name = "Rocket League (64-bit, DX11, Cooked)"

wincap = WindowCapture(window_name)
wincap.generate_image_dataset()
```
The above code serves the purpose of capturing screenshots from a designated window, primarily intended for the generation of image datasets. It begins by importing necessary libraries: numpy for numerical operations, win32gui, win32ui, and win32con from the win32 package for interfacing with the Windows graphical user interface, and PIL (Python Imaging Library) for image processing tasks. The core functionality is encapsulated within the WindowCapture class. Upon instantiation, this class attempts to locate the target window by its name using win32gui.FindWindow, raising an exception if the window is not found. Subsequently, it calculates the dimensions of the window, considering potential borders and the title bar, thereby determining the exact region to capture for screenshots. The get_screenshot method efficiently captures the screenshot by leveraging Win32 API functions to extract pixel data from the specified window. This pixel data is then converted into a numpy array, representing the image, facilitating further processing and analysis. Additionally, the generate_image_dataset method is provided to enable continuous screenshot capture at a predefined interval. Within this method, screenshots are captured iteratively and saved as PIL images, subsequently stored in a directory named "images" with sequential file naming. By executing this script with the desired window name (e.g., "Rocket League (64-bit, DX11, Cooked)"), it initiates the process of generating an image dataset specific to the content displayed within that window. This script finds significant utility in various applications, notably in machine learning, where labeled image data from specific interfaces or applications running on the Windows platform is required for training models effectively.


Next,  we move on to Labeling the Dataset

# PART 2: LABELING THE DATASET

```
import os
import random
import shutil
class LabelUtils:

    def create_shuffled_images_folder(self):
        if not os.path.exists("shuffled_images"):
            os.mkdir("shuffled_images")

        image_files = [f for f in os.listdir("images") if f.endswith(".jpg")]
        random.shuffle(image_files)

        for img in image_files:
            os.rename(f"images/{img}", f"shuffled_images/img_{len(os.listdir('shuffled_images'))}.jpg")

    def create_labeled_images_zip_file(self):
        if not os.path.exists("obj"):
            os.mkdir("obj")

        file_prefixes = [f.split('.')[0] for f in os.listdir("shuffled_images") if f.endswith(".txt")]

        for prefix in file_prefixes:
            os.rename(f"shuffled_images/{prefix}.txt", f"obj/{prefix}.txt")
            os.rename(f"shuffled_images/{prefix}.jpg", f"obj/{prefix}.jpg")

        shutil.make_archive('yolov4-tiny/obj', 'zip', '.', 'obj')

    def update_config_files(self, classes):
        with open("./yolov4-tiny/obj.names", "w") as file:
            file.write("\n".join(classes))

        with open("./yolov4-tiny/yolov4-tiny-custom_template.cfg", 'r') as file:
            cfg_content = file.read()

        updated_cfg_content = cfg_content.replace('_CLASS_NUMBER_', str(len(classes)))
        updated_cfg_content = updated_cfg_content.replace('_NUMBER_OF_FILTERS_', str((len(classes) + 5) * 3))
        updated_cfg_content = updated_cfg_content.replace('_MAX_BATCHES_', str(max(6000, len(classes) * 2000)))

        with open("./yolov4-tiny/yolov4-tiny-custom.cfg", 'w') as file:
            file.write(updated_cfg_content)
lbUtils = LabelUtils()
lbUtils.create_shuffled_images_folder()
```
-Lets break down the above code.
create_shuffled_images_folder():

This method creates a new directory named "shuffled_images" if it doesn't already exist.
It retrieves a list of image files from the "images" directory and shuffles them randomly.
Each image file is then renamed sequentially as "img_1.jpg", "img_2.jpg", etc., within the "shuffled_images" directory.

create_labeled_images_zip_file():

This method prepares labeled images and annotations for training by YOLOv4-tiny.
It creates a new directory named "obj" if it doesn't already exist.
It extracts file prefixes from the image files in the "shuffled_images" directory (considering corresponding annotation files).
For each image, it moves the corresponding annotation file (with a ".txt" extension) and image file (with a ".jpg" extension) to the "obj" directory.
Finally, it creates a ZIP archive named "obj.zip" containing all files within the "obj" directory.

update_config_files(classes):

This method updates configuration files required for training a YOLOv4-tiny model.
It writes the class names (provided as a parameter classes) to the "obj.names" file within the YOLOv4-tiny directory.
It reads the contents of the YOLOv4-tiny configuration template file.
It updates placeholders in the template file, such as _CLASS_NUMBER_, _NUMBER_OF_FILTERS_, and _MAX_BATCHES_, with actual values based on the number of classes.
The modified configuration content is then written to a new file named "yolov4-tiny-custom.cfg" within the YOLOv4-tiny directory.

For the actual labeling process we used *Make Sense AI*, an online application that allowed me to manually label the ~550 Game screenshots.
Below is a screenshot of what the labeling process looked like.
![image](https://github.com/FaizanDhankwala/RocketLeagueMachineLearning/assets/55712375/f891b590-e3a8-4fee-924a-197b23deb184)

After manually labeling every single picture, I exported the annotations as a YOLO file.

From their, I added the two next lines of code:
```
lbUtils = LabelUtils()
lbUtils.create_labeled_images_zip_file()

classes = ["Ball", "Boost"]

lbUtils = LabelUtils()
lbUtils.update_config_files(classes)
```

Lets break them down:
create_labeled_images_zip_file():

First, an instance of the LabelUtils class is created as lbUtils.
The create_labeled_images_zip_file() method is then called on this instance.
This method organizes labeled images and annotations for training by YOLOv4-tiny.
It moves corresponding annotation files and image files into a new directory named "obj".
Finally, it creates a ZIP archive named "obj.zip" containing all files within the "obj" directory.
update_config_files(classes):

Next, a list of classes is defined as classes = ["Ball", "Boost"].
Another instance of the LabelUtils class is created as lbUtils.
The update_config_files(classes) method is called on this instance, passing the list of classes as a parameter.
This method updates configuration files required for training the YOLOv4-tiny model.
It writes the class names ("Ball" and "Boost") to the "obj.names" file.
It updates placeholders in the YOLOv4-tiny configuration template file with actual values based on the number of classes.

After configuring the code, I uploaded the YOLO4-tiny folder to my google drive.

# PART 3: TRAINING THE MACHINE

-After the folder was uploaded to Google Drive, I opened the notebook on Google Collab and decided to use an existing github training module application.
```
%cd ..
from google.colab import drive
drive.mount('/content/gdrive')

!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive/yolov4-tiny
%cd /content/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile
!make
%cd data/
!find -maxdepth 1 -type f -exec rm -rf {} \;
%cd ..

%rm -rf cfg/
%mkdir cfg
!cp /mydrive/yolov4-tiny/obj.zip ../
!unzip ../obj.zip -d data/

!cp /mydrive/yolov4-tiny/yolov4-tiny-custom.cfg ./cfg
!cp /mydrive/yolov4-tiny/obj.names ./data
!cp /mydrive/yolov4-tiny/obj.data  ./data
!cp /mydrive/yolov4-tiny/process.py ./
!cp /mydrive/yolov4-tiny/yolov4-tiny.conv.29 ./
!python process.py
!ls data/
!./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show
```

To make things simple, the above code is simply commands used in Google Collab in order to machine train. Lets break them down below:
%cd ..: Changes the current directory to the parent directory.
from google.colab import drive: Imports the drive module from the google.colab package to mount Google Drive.
drive.mount('/content/gdrive'): Mounts Google Drive to the specified directory /content/gdrive.
!ln -s /content/gdrive/My\ Drive/ /mydrive: Creates a symbolic link named /mydrive pointing to the user's Google Drive directory.
!ls /mydrive/yolov4-tiny: Lists the contents of the yolov4-tiny directory in the user's Google Drive.
%cd /content/darknet/: Changes the current directory to /content/darknet/.
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile: Modifies the Makefile to enable OpenCV support.
!sed -i 's/GPU=0/GPU=1/' Makefile: Modifies the Makefile to enable GPU support.
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile: Modifies the Makefile to enable cuDNN support.
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile: Modifies the Makefile to enable half-precision floating point support in cuDNN.
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile: Modifies the Makefile to build with dynamic linking.
!make: Compiles the Darknet framework by executing the make command.
%cd data/: Changes the current directory to the data/ directory.
!find -maxdepth 1 -type f -exec rm -rf {} \;: Deletes all files in the current directory (except for directories).
%cd ..: Changes the current directory to the parent directory.
%rm -rf cfg/: Deletes the cfg/ directory and its contents recursively.
%mkdir cfg: Creates a new directory named cfg.
!cp /mydrive/yolov4-tiny/obj.zip ../: Copies the obj.zip file from Google Drive to the parent directory.
!unzip ../obj.zip -d data/: Unzips the obj.zip file and extracts its contents into the data/ directory.
!cp /mydrive/yolov4-tiny/yolov4-tiny-custom.cfg ./cfg: Copies the custom YOLOv4-tiny configuration file to the cfg/ directory.
!cp /mydrive/yolov4-tiny/obj.names ./data: Copies the obj.names file (containing class names) to the data/ directory.
!cp /mydrive/yolov4-tiny/obj.data ./data: Copies the obj.data file (containing dataset information) to the data/ directory.
!cp /mydrive/yolov4-tiny/process.py ./: Copies the process.py script to the current directory.
!cp /mydrive/yolov4-tiny/yolov4-tiny.conv.29 ./: Copies the pre-trained YOLOv4-tiny model weights to the current directory.
!python process.py: Executes the process.py script, which performs preprocessing tasks on the dataset.
!ls data/: Lists the contents of the data/ directory.
!./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show: Trains the YOLOv4-tiny model using the provided dataset, configuration file, and pre-trained weights, with the -dont_show flag suppressing visualization during training.

Here is an overivew of what the training looked like at this stage:
![image](https://github.com/FaizanDhankwala/RocketLeagueMachineLearning/assets/55712375/b87b4b5f-ddfb-4005-b1cb-d540c13a0163)

This training took roughly 3 hours.

# PART 4: RUNNING THE OBJECT DETECTION AND RESULTS

Now that the model was done training, the only thing left was to test it out. However, we ran into a pretty big issue when first starting to run the object detection. Of course, we needed a seperate window that mimiked the game but also added annotations on top of it. And so, we decided to add the lines of code:
```
import numpy as np
import win32gui, win32ui, win32con
from PIL import Image
from time import sleep
import cv2 as cv
import os
import random
class WindowCapture:
    w = 0
    h = 0
    hwnd = None

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[...,:3]
        img = np.ascontiguousarray(img) 
            
        return img

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while(True):
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpeg")
            sleep(1)
    
    def get_window_size(self):
        return (self.w, self.h)
class ImageProcessor:
    W = 0
    H = 0
    net = None
    ln = None
    classes = {}
    colors = []

    def __init__(self, img_size, cfg_file, weights_file):
        np.random.seed(42)
        self.net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.W = img_size[0]
        self.H = img_size[1]
        
        with open('yolov4-tiny/obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()
        
        # If you plan to utilize more than six classes, please include additional colors in this list.
        self.colors = [
            (0, 0, 255), 
            (0, 255, 0), 
            (255, 0, 0), 
            (255, 255, 0), 
            (255, 0, 255), 
            (0, 255, 255)
        ]
        

    def proccess_image(self, img):

        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)
        
        coordinates = self.get_coordinates(outputs, 0.5)

        self.draw_identified_objects(img, coordinates)

        return coordinates

    def get_coordinates(self, outputs, conf):

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                p0 = int(x - w//2), int(y - h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)

        if len(indices) == 0:
            return []

        coordinates = []
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            coordinates.append({'x': x, 'y': y, 'w': w, 'h': h, 'class': classIDs[i], 'class_name': self.classes[classIDs[i]]})
        return coordinates

    def draw_identified_objects(self, img, coordinates):
        for coordinate in coordinates:
            x = coordinate['x']
            y = coordinate['y']
            w = coordinate['w']
            h = coordinate['h']
            classID = coordinate['class']
            
            color = self.colors[classID]
            
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, self.classes[classID], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow('window',  img)
window_name = "Rocket League (64-bit, DX11, Cooked)"
cfg_file_name = "./yolov4-tiny/yolov4-tiny-custom.cfg"
weights_file_name = "yolov4-tiny-custom_last.weights"

wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

while(True):
    
    ss = wincap.get_screenshot()
    
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    coordinates = improc.proccess_image(ss)
    
    for coordinate in coordinates:
        print(coordinate)
    print()
    
    # If you have limited computer resources, consider adding a sleep delay between detections.
    # sleep(0.2)

print('Finished.')
```
While it looks lengthy, the code itself is pretty simple. Lets break it down:
WindowCapture Class:

This class is responsible for capturing screenshots from a specified window.
It uses the win32gui and win32ui libraries to interact with the Windows GUI.
The get_screenshot() method retrieves the screenshot and returns it as a NumPy array.
ImageProcessor Class:

This class processes the captured screenshots using a YOLOv4-tiny object detection model.
It utilizes OpenCV's DNN module to load the model and perform inference on the screenshots.
The proccess_image() method processes the image, detects objects, and returns their coordinates and class labels.
Main Execution:

The script initializes instances of the WindowCapture and ImageProcessor classes.
It enters a loop to continuously capture screenshots, process them using the ImageProcessor, and display the results in a window using OpenCV.
Pressing the 'q' key closes the window and terminates the script.


# RESULTS
- Despite the 550 images fed into the machine, since this was a 3D game with lots of angles and lighting, the model sometimes had trouble picking up the Ball and Boost. At times, I would drive around with the ball clearly in my field of view,  but the model would stay silent.
  ![image](https://github.com/FaizanDhankwala/RocketLeagueMachineLearning/assets/55712375/f8763dfb-c67b-4c30-8bfd-83f3b1679489)

-However, at other times- expeically when the car was not moving, the model had no trouble picking up both the ball and boost with high confidence
![image](https://github.com/FaizanDhankwala/RocketLeagueMachineLearning/assets/55712375/5904f03f-81b9-46b9-876b-c2d1a4179c11)

- And lastly, at times where the car was moving extremely fast, the model would sometimes mistake bright lights or wheels as the ball or boost. However, most of these were with low confidence
![image](https://github.com/FaizanDhankwala/RocketLeagueMachineLearning/assets/55712375/4f2f752b-9a5f-4c18-bdf2-6b5ec8482dfb)




# Conclusion
Understanding of Object Detection Concepts: I gained a deeper understanding of the concepts behind object detection, including different techniques and algorithms used in the field.

Practical Implementation: I learned how to implement object detection algorithms in Python using libraries such as OpenCV and frameworks like YOLOv4-tiny.

Image Processing Techniques: I acquired knowledge of various image processing techniques, such as resizing, cropping, and converting between different color spaces, which are essential for preprocessing images before object detection.

Model Training and Fine-Tuning: I learned how to train and fine-tune object detection models using custom datasets, configuration files, and pre-trained weights.

Integration with GUI: I gained experience in integrating object detection algorithms with graphical user interfaces (GUIs) to visualize and interact with the detection results in real-time.

Problem-Solving Skills: Throughout the challenge, I encountered various issues and errors, which required problem-solving skills to diagnose and resolve effectively. This process improved my troubleshooting abilities.

Documentation and Communication: I practiced documenting my code and explaining complex concepts in a clear and concise manner. This skill is crucial for collaboration with others and sharing knowledge effectively.

Iterative Learning Process: I realized that learning object detection is an iterative process that involves continuous experimentation, refinement, and learning from mistakes. Each iteration brings new insights and improvements.

I also learned that I need a LOT of images in order to create a good model!

