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
        self.comboBox_filters.addItem("No filter")
        self.comboBox_filters.addItem("blur")
        self.comboBox_filters.addItem("median")
        self.comboBox_filters.addItem("laplace")


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

        self.img=0 #global variable for the image

    def browse(self):
        filename = QFileDialog.getOpenFileName()##reading file
        imagePath = filename[0] ##reading file
        self.img = cv2.imread(imagePath,0)##reading file
        dft = cv2.dft(np.float32(self.img),flags = cv2.DFT_COMPLEX_OUTPUT)##calculate dft for frequency domain
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        self.axes_Orig_Freq.imshow(magnitude_spectrum, cmap = 'gray')## frequency domian
        self.axes_Orig_Spat.imshow(self.img, cmap = 'gray') ##original image
        self.axes_Orig_Hist.imshow(self.img, cmap = 'gray') ##original image in the histogram tab

        ##write histogram equalization code here##




        ##############


        self.canvas_Orig_Freq.draw()##apply changes
        self.canvas_Orig_Spat.draw()##apply changes
        self.canvas_Orig_Hist.draw()##apply changes
        self.canvas_Filt_Hist.draw()##apply changes
        self.filtering()


    def filtering(self): #this function is called when the combobox is changed

        ## write the code of the right filter depending on the combobox's current index
        if self.comboBox_filters.currentText() == 'laplace':
            laplaceSrc = cv2.GaussianBlur(self.img, (3, 3), 0)
            filteredLaplacian = cv2.Laplacian(laplaceSrc, cv2.CV_16S, ksize=7)
            laplaceF = np.fft.fft2(filteredLaplacian)
            laplaceFshift = np.fft.fftshift(laplaceF)
            laplace_magnitude_spectrum = 20*np.log(np.abs(laplaceFshift))
            self.axes_Filt_Spat.imshow(filteredLaplacian, cmap='gray')
            self.axes_Filt_Freq.imshow(laplace_magnitude_spectrum.astype('uint8'), cmap='gray')

        elif self.comboBox_filters.currentText() == 'median':
            median = cv2.medianBlur(self.img,11)
            medianF = np.fft.fft2(median)
            medianFshift = np.fft.fftshift(medianF)
            median_magnitude_spectrum = 20*np.log(np.abs(medianFshift)) 
            self.axes_Filt_Spat.imshow(median, cmap='gray')
            self.axes_Filt_Freq.imshow(median_magnitude_spectrum.astype('uint8'), cmap='gray')
          
        ##Write filters code here## 



        #############


        self.canvas_Filt_Freq.draw()##apply changes
        self.canvas_Filt_Spat.draw()##apply changes
    
 
        



    

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
    
    
