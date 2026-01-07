import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import gdown

import argparse
import sys
import os 
import json
import pprint
from contextlib import redirect_stdout
import datetime

sys.path.append("./sinkscraper/FBA_Matting")
sys.path.append("./FBA_Matting")

from demo import pred
from networks.models import build_model

from stl import mesh
from scipy.interpolate import interp1d

# Standard model values for the board.
# OBS: Only works for specific board "scraper_empty.stl" currently
MODEL_SCALE = 0.15
MAX_BOARD_HEIGHT = 1.6

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class buildScraper:
    """ Class used to build the sink-scraper model. """

    def __init__(self
                 , filename
                 , save_intermediate_pictures = False
                 , show_intermediate_pictures = False
                 , log = True):
        self.log = log
        self.log_file = []
        self.save_intermediate_pictures = save_intermediate_pictures
        self.show_intermediate_pictures = show_intermediate_pictures
        self.file = filename

        self.params = self.loadParameters()
        self.image = self.loadModelImage()
        self.board = self.prepareBoard()

        # Applies resnet and FBA model on gpu if available
        if torch.cuda.is_available():
            self.logMessage("Cuda selected", level="MINOR")
            self.device = torch.device("cuda")
        else:
            self.logMessage("CPU selected", level="MINOR")
            self.device = torch.device("cpu")

        self.fba_model = self.buildFBAModel(self.device)
        self.resnet_model = self.buildResnetModel(self.device)

    def writeLog(self):
        """ Writes to log-file and finishes log. """

        with open("log.txt", 'w') as file:
            file.writelines("\n".join(self.log_file))
        self.logMessage("Log saved as log.txt", level="MINOR")

    def logMessage(self, message, level = "INFO"):
        """ Logs a message. """

        if self.log:
            match level: 
                case "FINAL":
                    log_message = "\n" + "[ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] * " + message
                    print(bcolors.OKGREEN + log_message + bcolors.ENDC)
                case "INFO":
                    log_message = "[ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] -> " + message
                    print(bcolors.OKCYAN + log_message + bcolors.ENDC)
                case "MINOR":
                    log_message = "" + message 
                    print(log_message)
                case "ERROR":
                    log_message = "[ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] ! " + message 
                    print(bcolors.FAIL + log_message + bcolors.ENDC)
                case "WARNING":
                    log_message = "[ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] ! " + message
                    print(bcolors.WARNING + log_message + bcolors.ENDC)
            self.log_file.append(log_message)

    def loadParameters(self, parameter_path = "parameters.json"):
        """ Loads parameters from json file. """

        with open(parameter_path) as f:
            js = json.load(f)
            self.logMessage(f"Parameters from  \"{parameter_path}\" loaded successfully")
            self.logMessage(pprint.pformat(js, compact=True).replace("'",'"'), level="MINOR")
            return js

    def loadModelImage(self):
        """ Loads image to model. """

        img_orig = cv2.imread("./pictures/" + self.file, 1)
        self.logMessage(f"Image \"{self.file}\" loaded and scaled successfully")
        return img_orig

    def buildFBAModel(self, device):
        """ Builds FBA model. """

        # First time running the model must download FBA model
        if not os.path.isfile("FBA.pth"):
            url = 'https://drive.google.com/uc?id=1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1'
            output = 'FBA.pth'
            gdown.download(url, output, quiet=False)
            self.logMessage("FBA model parameters downloaded", level="MINOR")

        args = argparse.Namespace(
            encoder='resnet50_GN_WS',
            decoder='fba_decoder',
            weights='FBA.pth'
        )
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                model = build_model(args)      
        self.logMessage("FBA model build successfully")
        return model.to(device)
    
    def buildResnetModel(self, device):
        """ Builds Resnet model. """

        # Load weights for model
        w = "DeepLabV3_ResNet101_Weights.DEFAULT"
        model = deeplabv3_resnet101(weights=w).to(device)
        model.eval()
        self.logMessage(f"Resnet model \"{w}\" build successfully")
        return model

    def applyModel(self, filename_out = None):
        """ Applies Resnet and FBA model alongside image processing. """

        os.chdir("./pictures/")

        # Load smaller version of the image for Resnet model
        k = min(1.0, 1024/max(self.image.shape[0], self.image.shape[1]))
        img = cv2.resize(self.image, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)

        deeplab_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def apply_deeplab(deeplab, img, device):
            input_tensor = deeplab_preprocess(img)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                output = deeplab(input_batch.to(device))['out'][0]
            output_predictions = output.argmax(0).cpu().numpy()
            return (output_predictions == 15)
        
        # Apply Resnet model
        mask = apply_deeplab(self.resnet_model, img, self.device)

        # Compute trimaps of image for FBA model
        trimap = np.zeros((mask.shape[0], mask.shape[1], 2))
        trimap[:, :, 1] = mask > 0
        trimap[:, :, 0] = mask == 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
        trimap[:, :, 0] = cv2.erode(trimap[:, :, 0], kernel)
        trimap[:, :, 1] = cv2.erode(trimap[:, :, 1], kernel)
        trimap_im =  trimap[:,:,1] + (1-np.sum(trimap,-1))/2

        # Apply matting to image
        _, _, alpha = pred((img/255.0)[:, :, ::-1], trimap, self.fba_model)

        # Resize back to original size
        img_ = self.image.astype(np.float32)/255
        alpha_ = cv2.resize(alpha, (img_.shape[1], img_.shape[0]), cv2.INTER_LANCZOS4)
        fg_alpha = np.concatenate([img_, alpha_[:, :, np.newaxis]], axis=2)

        # Calculate grayscale image
        img_gray = (fg_alpha*255).astype(np.uint8) 
        
        # Adjust for RGB skewness of our human eyeballs 
        rows, cols = img_gray.shape[:2]
        avg_brightness = 0
        for i in range(rows):
            for j in range(cols):
                gray = 0.2989 * img_gray[i, j][2] + 0.5870 * img_gray[i, j][1] + 0.1140 * img_gray[i, j][0]
                img_gray[i, j] = [gray, gray, gray, img_gray[i, j][3]]
                avg_brightness += gray * (img_gray[i, j][3] / 255)

        # Find average brightness of image
        avg_brightness /= (rows * cols)

        # Set contrast based on target contrast brightness (not on alpha parameter)
        contrast_mod = max(1, self.params["TARGET_CONTRAST_BRIGHTNESS"]/avg_brightness)
        img_contrasted = img_gray
        img_contrasted[:,0] = img_contrasted[:,0] * contrast_mod
        img_contrasted[:,1] = img_contrasted[:,1] * contrast_mod
        img_contrasted[:,2] = img_contrasted[:,2] * contrast_mod

        self.logMessage(f"Average_brightness: {avg_brightness:.2f}", level="MINOR")
        self.logMessage(f"Contrast mod set at: {contrast_mod:.2f}", level="MINOR")

        # Blur image for more smooth 3D model
        img_blur = cv2.blur(img_contrasted,(self.params["BLUR"], self.params["BLUR"]))

        # Show intermediate pictures
        if self.show_intermediate_pictures:
            plt.title("Original")
            plt.imshow(self.image[:, :, ::-1])
            plt.show()
            plt.title("Mask")
            plt.imshow(mask, cmap="gray")
            plt.show()
            plt.title("Trimap")
            plt.imshow(trimap_im, cmap='gray', vmin=0, vmax=1)
            plt.show()
            plt.title("Gray")
            plt.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
            plt.show()
            # plt.title("Normalized")
            # plt.imshow(img_normalized, cmap='gray', vmin=0, vmax=1)
            # plt.show()
            plt.title(f"Contrasted at {contrast_mod:.2f}")
            plt.imshow(img_contrasted, cmap='gray', vmin=0, vmax=1)
            plt.show()
            plt.title("Blurred")
            plt.imshow(img_blur, cmap='gray', vmin=0, vmax=1)
            plt.show()

        # Save modified picture
        if self.save_intermediate_pictures:
            cv2.imwrite(self.file[:-4] + "_out.png" if filename_out == None else filename_out, img_blur)

        self.logMessage("Person cut out successfully")
        os.chdir("..")

        return img_blur

    def setFaces(self, model_image):
        """ Set height of vertices of top-faces """

        def getColor(mapper, i, j):
            """
            Slightly black magic that computes the height of a vertex based on both the grayscale color as well as the alpha value.
            The alpha value is used for smooth surfaces ot the edge of the model but also to give a distinct outline in these places.
            """
            def minCutoffOverLayers(a):
                return min(a, self.params["CUTOFF"]) / self.params["LAYERS"]

            return (
                self.params["MIN_PICTURE_HEIGHT"] +
                minCutoffOverLayers(int(mapper(model_image[i][j][0]) * mapper(model_image[i][j][3]/255))) * self.params["MAX_PICTURE_HEIGHT"] + 
                minCutoffOverLayers(int(mapper(255 - model_image[i][j][3]))) * (MAX_BOARD_HEIGHT - self.params["MIN_PICTURE_HEIGHT"]))

        rows, cols = model_image.shape[:2]
        num_triangles = rows * cols * 2
        face_data = np.zeros(num_triangles, dtype=mesh.Mesh.dtype)
        mapper = interp1d([0, 255], [0, self.params["LAYERS"]])

        # Inefficient nested for-loops to iterate over each pixel and calculate a height
        for i in range(rows-1):
            for j in range(cols-1):
                c1, c2, c3, c4 = (
                    getColor(mapper, i  ,j  ),
                    getColor(mapper, i+1,j  ),
                    getColor(mapper, i  ,j+1),
                    getColor(mapper, i+1,j+1)
                )

                # Two triangles corresponding to a pixel are made upon each iteration of the loop
                # This is the index in the original array
                base_index = (i*cols+j)*2

                # Offset to center the image on the scraper
                i_val = i - rows/2 + 0.5
                j_val = j - cols/2 + 0.5

                # At the edges of the picture, we must force a triangle between the board and the picture
                if i == 0:
                    c1 = MAX_BOARD_HEIGHT
                    c3 = MAX_BOARD_HEIGHT
                if i == rows-2:
                    c2 = MAX_BOARD_HEIGHT
                    c4 = MAX_BOARD_HEIGHT
                if j == 0:
                    c1 = MAX_BOARD_HEIGHT
                    c2 = MAX_BOARD_HEIGHT
                if j == cols-2:
                    c3 = MAX_BOARD_HEIGHT
                    c4 = MAX_BOARD_HEIGHT
                
                # Setting faces
                face_data["vectors"][base_index  ] = np.array(
                    [[i_val  , j_val  , c1]
                    ,[i_val+1, j_val  , c2]
                    ,[i_val  , j_val+1, c3]])
                face_data["vectors"][base_index+1] = np.array(
                    [[i_val+1, j_val  , c2]
                    ,[i_val+1, j_val+1, c4]
                    ,[i_val  , j_val+1, c3]])
                
                # Scaling vertices on x and y axis
                face_data["vectors"][base_index  ][:,0] *= MODEL_SCALE
                face_data["vectors"][base_index  ][:,1] *= MODEL_SCALE
                face_data["vectors"][base_index+1][:,0] *= MODEL_SCALE
                face_data["vectors"][base_index+1][:,1] *= MODEL_SCALE
                face_data["vectors"][base_index  ][:,2] -= MAX_BOARD_HEIGHT
                face_data["vectors"][base_index+1][:,2] -= MAX_BOARD_HEIGHT

        return face_data

    def prepareBoard(self, model_parameter_path = "scraper_empty.stl"):
        """ Loads and prepares scraper-model. """
        
        board = mesh.Mesh.from_file("./scraper models/" + model_parameter_path)
        board.rotate([0.0, 0.0, 0.5], math.radians(-90))
        board_data = board.data.copy()
        self.logMessage(f"Base scraper board model from  \"{model_parameter_path}\" loaded, rotated, and scaled successfully")
        return board_data
    
    def finishModel(self, model_image, filename_out = None):
        """ Finishes the model by combining board-model and picture-model. Saves outout and writes final log message. """

        stl_model = mesh.Mesh(np.concatenate([
            self.board,
            self.setFaces(model_image),
        ]))
        self.logMessage(f"STL model created successfully")
        name = self.file[0:-4] + ".stl" if filename_out == None else filename_out
        stl_model.save("./stls/" + name)
        self.logMessage(f"STL model saved successfully as {name} in ./stls/")
        self.logMessage("All done!", level="FINAL")
        
        self.writeLog()
