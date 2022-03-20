
from img_filter_gui import Ui_MainWindow
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
from PIL import Image as im
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
import sys
from PyQt5.QtGui import QPixmap

import cv2
import numpy as np
from matplotlib import pyplot as plt


class GUI (Ui_MainWindow):
    def setup(self,MainWindow):
        super().setupUi(MainWindow)
        self.comboBox_filters.addItem("high pass spatial")
        self.comboBox_filters.addItem("low pass spatial")
        self.comboBox_filters.addItem("high pass freq")
        self.comboBox_filters.addItem("low pass freq")
        self.comboBox_filters.addItem("median")
        self.comboBox_filters.addItem("laplace")

        self.comboBox_color_filters.addItem("Gray scale mode")
        self.comboBox_color_filters.addItem("RGB scale mode")

        self.figure_Orig_Spat = Figure(figsize=(3, 3), dpi=100)
        self.axes_Orig_Spat = self.figure_Orig_Spat.add_subplot()
        self.canvas_Orig_Spat = FigureCanvas(self.figure_Orig_Spat)
        self.gridLayout_2.addWidget(self.canvas_Orig_Spat, 1, 1, 1, 1)

        self.figure_Orig_Freq = Figure(figsize=(3, 3), dpi=100)
        self.axes_Orig_Freq = self.figure_Orig_Freq.add_subplot()
        self.canvas_Orig_Freq = FigureCanvas(self.figure_Orig_Freq)
        self.gridLayout_2.addWidget(self.canvas_Orig_Freq, 1, 2, 1, 1)


        self.figure_Filt_Spat = Figure(figsize=(3, 3), dpi=100)
        self.axes_Filt_Spat = self.figure_Filt_Spat.add_subplot()
        self.canvas_Filt_Spat = FigureCanvas(self.figure_Filt_Spat)
        self.gridLayout_2.addWidget(self.canvas_Filt_Spat, 2, 1, 1, 1)



        self.figure_Filt_Freq = Figure(figsize=(3, 3), dpi=100)
        self.axes_Filt_Freq = self.figure_Filt_Freq.add_subplot()
        self.canvas_Filt_Freq = FigureCanvas(self.figure_Filt_Freq)
        self.gridLayout_2.addWidget(self.canvas_Filt_Freq, 2, 2, 1, 1)

        self.figure_Orig_Hist = Figure(figsize=(3, 3), dpi=100)
        self.axes_Orig_Hist = self.figure_Orig_Hist.add_subplot()
        self.canvas_Orig_Hist = FigureCanvas(self.figure_Orig_Hist)
        self.gridLayout_4.addWidget(self.canvas_Orig_Hist, 1, 0, 1, 1)


        self.figure_Filt_Hist = Figure(figsize=(3, 3), dpi=100)
        self.axes_Filt_Hist = self.figure_Filt_Hist.add_subplot()
        self.canvas_Filt_Hist = FigureCanvas(self.figure_Filt_Hist)
        self.gridLayout_4.addWidget(self.canvas_Filt_Hist, 1, 1, 1, 1)

        



        self.axes=[ self.axes_Orig_Spat,self.axes_Orig_Freq,self.axes_Filt_Spat,self.axes_Filt_Freq,self.axes_Orig_Hist,self.axes_Filt_Hist]
        for axis in self.axes: ## removing axes from the figure so the image would look nice
            axis.set_xticks([])
            axis.set_yticks([])

        
        self.actionImage.triggered.connect(lambda : self.browse())
        self.comboBox_filters.currentIndexChanged.connect(self.filtering)
        self.comboBox_color_filters.currentIndexChanged.connect(self.filtering)
        self.img=0
        self.img_rgb=0

       
    
    def browse(self):
        filename = QFileDialog.getOpenFileName()##reading file
        imagePath = filename[0] ##reading file
        self.img_bgr = cv2.imread(imagePath)##reading file
        self.img = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        self.img_rgb=cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
      
        dft = cv2.dft(np.float32(self.img),flags = cv2.DFT_COMPLEX_OUTPUT)##calculate dft for frequency domain
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
         ##original image
        self.axes_Orig_Hist.imshow(self.img, cmap = 'gray') ##original image in the histogram tab
        self.axes_Orig_Freq.imshow(magnitude_spectrum, cmap = 'gray')
        self.canvas_Orig_Freq.draw()##apply changes
        ##apply changes
        self.canvas_Orig_Hist.draw()##apply changes
        self.canvas_Filt_Hist.draw()##apply changes
        self.filtering()


    # the input is rbg or grayscale image
    def low_pass_spatial_filter(self,image):
        kernel1 = np.ones((5, 5))/25
        # Applying the filter2D() function
        filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
        try:
            _,_=image.shape
            low_passF = np.fft.fft2(filtered_image)
            low_passFshift = np.fft.fftshift(low_passF)
            low_pass_magnitude_spectrum = 20*np.log(np.abs((low_passFshift)+1)) 
        except:
            img_g=cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
            low_passF = np.fft.fft2(img_g)
            low_passFshift = np.fft.fftshift(low_passF)
            low_pass_magnitude_spectrum = 20*np.log(np.abs((low_passFshift)+1)) 
        
        return filtered_image,low_pass_magnitude_spectrum 

    # the input is rbg or grayscale image

    def high_pass_spatial_filter(self,image):
        kernel2 = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
        # Applying the filter2D() function
        filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
        try:
            _,_=image.shape
            high_passF = np.fft.fft2(filtered_image)
            high_passFshift = np.fft.fftshift(high_passF)
            high_pass_magnitude_spectrum = 20*np.log(np.abs((high_passFshift)+1))
        except:
            img_g=cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
            high_passF = np.fft.fft2(img_g)
            high_passFshift = np.fft.fftshift(high_passF)
            high_pass_magnitude_spectrum = 20*np.log(np.abs((high_passFshift)+1))

        
        return filtered_image,high_pass_magnitude_spectrum
    
    # the input is rbg or grayscale image

    def low_pass_freq_filter(self,img):
        #Output is a 2D complex array. 1st channel real and 2nd imaginary
        #For fft in opencv input image needs to be converted to float32
        #dft=cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        try:
            flag=0
            _,_=img.shape
            dft=np.fft.fft2(img)
        except:
            flag=1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            dft=np.fft.fft2(img[:,:,2])
        #Rearranges a Fourier transform X by shifting the zero-frequency 
        #component to the center of the array.
        #Otherwise it starts at the tope left corenr of the image (array)
        dft_shift=np.fft.fftshift(dft)
        ##Magnitude of the function is 20.log(abs(f))
        #For values that are 0 we may end up with indeterminate values for log. 
        #So we can add 1 to the array to avoid seeing a warning.
        magnitude_spectrum=20 * np.log(np.abs(dft_shift)+1)
        #--------------------------------------------------------------------------------------------------
        # Circular HPF mask, center circle is 0, remaining all ones
        #Can be used for edge detection because low frequencies at center are blocked
        #and only high frequencies are allowed. Edges are high frequency components.
        #Amplifies noise.
        
        rows, cols= dft.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols), np.uint8)
        


        r = 75
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r
        mask[mask_area] = 0
        # apply mask and inverse DFT
        fshift = dft_shift * mask
        #--------------------------------------------------------------------------------------------------
        # revers from the freq domain to spatial domain
        
        fshift_mask_mag = 20 * np.log(np.abs(fshift)+1)
        
        f_ishift=np.fft.ifftshift(fshift)
        img_back=np.fft.ifft2(f_ishift)
        img_filtered=np.abs(img_back)


        if(flag==1):
            img[:,:,2]=img_filtered
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            return img,magnitude_spectrum,fshift_mask_mag

        

        return img_filtered,magnitude_spectrum,fshift_mask_mag


        




    # the input is rbg or grayscale image

    def high_pass_freq_filter(self,img):
        #Output is a 2D complex array. 1st channel real and 2nd imaginary
        #For fft in opencv input image needs to be converted to float32
        #dft=cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        try:
            flag=0
            _,_=img.shape
            dft=np.fft.fft2(img)
        except:
            flag=1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            dft=np.fft.fft2(img[:,:,2])
        #Rearranges a Fourier transform X by shifting the zero-frequency 
        #component to the center of the array.
        #Otherwise it starts at the tope left corenr of the image (array)
        dft_shift=np.fft.fftshift(dft)
        ##Magnitude of the function is 20.log(abs(f))
        #For values that are 0 we may end up with indeterminate values for log. 
        #So we can add 1 to the array to avoid seeing a warning.
        magnitude_spectrum=20 * np.log(np.abs(dft_shift)+1)
        #--------------------------------------------------------------------------------------------------
        # Circular HPF mask, center circle is 0, remaining all ones
        #Can be used for edge detection because low frequencies at center are blocked
        #and only high frequencies are allowed. Edges are high frequency components.
        #Amplifies noise.
        
        rows, cols= dft.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols), np.uint8)
        


        r = 40
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        # apply mask and inverse DFT
        fshift = dft_shift * mask
        #--------------------------------------------------------------------------------------------------
        # revers from the freq domain to spatial domain
        
        fshift_mask_mag = 20 * np.log(np.abs(fshift)+1)
        
        f_ishift=np.fft.ifftshift(fshift)
        img_back=np.fft.ifft2(f_ishift)
        img_filtered=np.abs(img_back)


        if(flag==1):
            img[:,:,2]=img_filtered
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            return img,magnitude_spectrum,fshift_mask_mag

        

        return img_filtered,magnitude_spectrum,fshift_mask_mag


    def median(self,img):
   
        median = cv2.medianBlur(img,11)
        try:
            median_gray=cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
        except:
            median_gray=median
        medianF = np.fft.fft2(median_gray)
        medianFshift = np.fft.fftshift(medianF)
        median_magnitude_spectrum = 20*np.log(np.abs(medianFshift)+1) 
        return median,median_magnitude_spectrum

    def laplace(self,img):

        try:
            flag=0
            _,_=img.shape
            laplaceSrc = cv2.GaussianBlur(img, (3, 3), 0)
            filteredLaplacian = cv2.Laplacian(laplaceSrc, cv2.CV_16S, ksize=7)

        except:
            filteredLaplacian_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_v=filteredLaplacian_hsv[:,:,2]
            flag=1
            laplaceSrc = cv2.GaussianBlur(img_v, (3, 3), 0)
            filteredLaplacian = cv2.Laplacian(laplaceSrc, cv2.CV_16S, ksize=7) 
    


        laplaceF = np.fft.fft2(filteredLaplacian)
        laplaceFshift = np.fft.fftshift(laplaceF)
        laplace_magnitude_spectrum = 20*np.log(np.abs(laplaceFshift))
        
        if(flag==1):
            filteredLaplacian_hsv[:,:,2]=filteredLaplacian
            filteredLaplacian_rgb=cv2.cvtColor(filteredLaplacian_hsv, cv2.COLOR_HSV2RGB)
            return filteredLaplacian_rgb,laplace_magnitude_spectrum


        return filteredLaplacian,laplace_magnitude_spectrum


    def filtering(self): #this function is called when the combobox is changed
        if self.comboBox_color_filters.currentText() == 'Gray scale mode' :
            self.axes_Orig_Spat.imshow(self.img, cmap = 'gray')



            if self.comboBox_filters.currentText() == 'laplace':
               
                filteredLaplacian,laplace_magnitude_spectrum=self.laplace(self.img)
                self.axes_Filt_Spat.imshow(filteredLaplacian, cmap='gray')
                self.axes_Filt_Freq.imshow(laplace_magnitude_spectrum.astype('uint8'), cmap='gray')
        
            elif self.comboBox_filters.currentText() == 'median':
                median,median_magnitude_spectrum=self.median(self.img) 
                self.axes_Filt_Spat.imshow(median, cmap='gray')
                self.axes_Filt_Freq.imshow(median_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'low pass spatial':
                filtered_img,low_pass_magnitude_spectrum= self.low_pass_spatial_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'high pass spatial':
                filtered_img,high_pass_magnitude_spectrum= self.high_pass_spatial_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'low pass freq':
                filtered_img,_,low_pass_magnitude_spectrum= self.low_pass_freq_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'high pass freq':
                filtered_img,_,high_pass_magnitude_spectrum= self.high_pass_freq_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')


        elif self.comboBox_color_filters.currentText() == 'RGB scale mode':
            
            self.axes_Orig_Spat.imshow(self.img_rgb)
            ## write the code of the right filter depending on the combobox's current index
            if self.comboBox_filters.currentText() == 'laplace':
                filteredLaplacian,laplace_magnitude_spectrum=self.laplace(self.img_rgb)
                self.axes_Filt_Spat.imshow(filteredLaplacian, cmap='gray')
                self.axes_Filt_Freq.imshow(laplace_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'median':
                median,median_magnitude_spectrum=self.median(self.img_rgb) 
                self.axes_Filt_Spat.imshow(median, cmap='gray')
                self.axes_Filt_Freq.imshow(median_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'low pass spatial':
                filtered_img,low_pass_magnitude_spectrum= self.low_pass_spatial_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'high pass spatial':
                filtered_img,high_pass_magnitude_spectrum= self.high_pass_spatial_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'low pass freq':
                filtered_img,_,low_pass_magnitude_spectrum= self.low_pass_freq_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'high pass freq':
                filtered_img,_,high_pass_magnitude_spectrum= self.high_pass_freq_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
          
        ##Write filters code here## 



        #############


        self.canvas_Filt_Freq.draw()##apply changes
        self.canvas_Filt_Spat.draw()##apply changes
        self.canvas_Orig_Spat.draw()
    
 
        



    

class application(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.gui=GUI()
        self.gui.setup(self)
    




    

    
        




def window():
    app = QApplication(sys.argv)
    win = application()
    win.show()
    sys.exit(app.exec_())


# main code
if __name__ == "__main__":
    window()
    
    
