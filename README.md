# RocketLeagueMachineLearning
![image](https://github.com/FaizanDhankwala/RocketLeagueMachineLearning/assets/55712375/b483dccc-cf6d-421d-9edf-e25503fefb1e)


*This program was all done with the use of Python and Machine Learning (TensorFlow).

*The entire program can be broken down into 3 parts

# First Part --------------- Labeling the DataSet
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
