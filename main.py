import sys

import cv2
import numpy as np
import qdarkstyle
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# class GUI (Ui_MainWindow):
#     def setup(self,MainWindow):
#         super().setupUi(MainWindow)
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('img_filter_gui_2.ui', self)
        self.show()  # Show the GUI

        self.comboBox_filters.addItem("No Filter")
        self.comboBox_filters.addItem("High Pass Spatial")
        self.comboBox_filters.addItem("Low Pass Spatial")
        self.comboBox_filters.addItem("High Pass Freq")
        self.comboBox_filters.addItem("Low Pass Freq")
        self.comboBox_filters.addItem("Median")
        self.comboBox_filters.addItem("Laplace")
        self.comboBox_color_filters.addItem("Gray Scale Mode")
        self.comboBox_color_filters.addItem("RGB Scale Mode")

        self.figure_Orig_Spat = Figure(figsize=(3, 3), dpi=100)
        self.axes_Orig_Spat = self.figure_Orig_Spat.add_subplot()
        self.canvas_Orig_Spat = FigureCanvas(self.figure_Orig_Spat)
        self.canvas_Orig_Spat.figure.set_facecolor("#19232D")
        self.gridLayout_2.addWidget(self.canvas_Orig_Spat, 1, 1, 1, 1)

        self.figure_Orig_Freq = Figure(figsize=(3, 3), dpi=100)
        self.axes_Orig_Freq = self.figure_Orig_Freq.add_subplot()
        self.canvas_Orig_Freq = FigureCanvas(self.figure_Orig_Freq)
        self.canvas_Orig_Freq.figure.set_facecolor("#19232D")
        self.gridLayout_2.addWidget(self.canvas_Orig_Freq, 1, 2, 1, 1)

        self.figure_Filt_Spat = Figure(figsize=(3, 3), dpi=100)
        self.axes_Filt_Spat = self.figure_Filt_Spat.add_subplot()
        self.canvas_Filt_Spat = FigureCanvas(self.figure_Filt_Spat)
        self.canvas_Filt_Spat.figure.set_facecolor("#19232D")
        self.gridLayout_2.addWidget(self.canvas_Filt_Spat, 2, 1, 1, 1)

        self.figure_Filt_Freq = Figure(figsize=(3, 3), dpi=100)
        self.axes_Filt_Freq = self.figure_Filt_Freq.add_subplot()
        self.canvas_Filt_Freq = FigureCanvas(self.figure_Filt_Freq)
        self.canvas_Filt_Freq.figure.set_facecolor("#19232D")
        self.gridLayout_2.addWidget(self.canvas_Filt_Freq, 2, 2, 1, 1)

        self.figure_Orig_image = Figure(figsize=(3, 3), dpi=100)
        self.axes_Orig_image = self.figure_Orig_image.add_subplot()
        self.canvas_Orig_image = FigureCanvas(self.figure_Orig_image)
        self.canvas_Orig_image.figure.set_facecolor("#19232D")
        self.gridLayout_4.addWidget(self.canvas_Orig_image, 1, 0, 1, 1)

        self.figure_Orig_Hist = Figure(figsize=(3, 3), dpi=100)
        self.axes_Orig_Hist = self.figure_Orig_Hist.add_subplot()
        self.canvas_Orig_Hist = FigureCanvas(self.figure_Orig_Hist)
        self.canvas_Orig_Hist.figure.set_facecolor("#19232D")
        self.axes_Orig_Hist.xaxis.label.set_color('white')
        self.axes_Orig_Hist.yaxis.label.set_color('white')
        self.axes_Orig_Hist.axes.tick_params(axis="x", colors="white")
        self.axes_Orig_Hist.axes.tick_params(axis="y", colors="white")
        self.axes_Orig_Hist.axes.set_title("Histogram")
        self.axes_Orig_Hist.axes.title.set_color('white')
        self.gridLayout_4.addWidget(self.canvas_Orig_Hist, 2, 0, 1, 1)

        self.figure_Filt_image = Figure(figsize=(3, 3), dpi=100)
        self.axes_Filt_image = self.figure_Filt_image.add_subplot()
        self.canvas_Filt_image = FigureCanvas(self.figure_Filt_image)
        self.canvas_Filt_image.figure.set_facecolor("#19232D")
        self.gridLayout_4.addWidget(self.canvas_Filt_image, 1, 1, 1, 1)

        self.figure_Filt_Hist = Figure(figsize=(3, 3), dpi=100)
        self.axes_Filt_Hist = self.figure_Filt_Hist.add_subplot()
        self.canvas_Filt_Hist = FigureCanvas(self.figure_Filt_Hist)
        self.axes_Filt_Hist.xaxis.label.set_color('white')
        self.axes_Filt_Hist.yaxis.label.set_color('white')
        self.canvas_Filt_Hist.figure.set_facecolor("#19232D")
        self.axes_Filt_Hist.axes.tick_params(axis="x", colors="white")
        self.axes_Filt_Hist.axes.tick_params(axis="y", colors="white")
        self.axes_Filt_Hist.axes.set_title("Histogram")
        self.axes_Filt_Hist.axes.title.set_color('white')
        self.gridLayout_4.addWidget(self.canvas_Filt_Hist, 2, 1, 1, 1)

        self.axes = [self.axes_Orig_Spat, self.axes_Orig_Freq, self.axes_Filt_Spat, self.axes_Filt_Freq,
                     self.axes_Orig_image, self.axes_Filt_image]
        for axis in self.axes:  ## removing axes from the figure so the image would look nice
            axis.set_xticks([])
            axis.set_yticks([])

        self.actionImage.triggered.connect(lambda: self.browse())
        self.open_1.clicked.connect(lambda: self.browse())
        self.comboBox_filters.currentIndexChanged.connect(self.filtering)
        self.comboBox_color_filters.currentIndexChanged.connect(self.filtering)
        self.comboBox_Hist_Scale.currentIndexChanged.connect(lambda: self.Histogram())

        self.img = 0
        self.img_rgb = 0

    def browse(self):
        filename = QFileDialog.getOpenFileName()  ##reading file
        imagePath = filename[0]  ##reading file
        self.img_bgr = cv2.imread(imagePath)  ##reading file
        self.img_bgr = cv2.resize(self.img_bgr, (520, 265))
        self.img = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)  ##calculate dft for frequency domain
        dft_shift = np.fft.fftshift(dft)
        global magnitude_spectrum
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        self.axes_Orig_Freq.imshow(magnitude_spectrum, cmap='gray')  ## frequency domian
        self.axes_Orig_Spat.imshow(self.img, cmap='gray')  ##original image

        self.axes_Filt_Spat.imshow(self.img, cmap='gray')  ##original image
        self.axes_Filt_Freq.imshow(magnitude_spectrum, cmap='gray')  ## frequency domian

        self.Histogram()

                 ################ HISTOGRAM EQUALIZATION ################################
    def Histogram(self):

        if self.comboBox_Hist_Scale.currentText() == 'RGB Scale':
            self.axes_Orig_image.imshow(self.img_rgb)  ##original image in the histogram tab

            img_hsv = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2HSV)
            hist_gray, bins_gray = np.histogram(img_hsv[:, :, 2], 256, [0, 255])
            cdf = hist_gray.cumsum()
            cdf_normalized = cdf * float(hist_gray.max()) / cdf.max()

            self.axes_Orig_Hist.clear()
            self.axes_Orig_Hist.hist(img_hsv[:, :, 2].flatten(), bins=bins_gray)
            self.axes_Orig_Hist.plot(cdf_normalized, color='r')
            self.axes_Orig_Hist.legend(('CDF', 'Histogram'), loc='upper right')
            self.axes_Orig_Hist.set_xlabel("Intensity Level")
            self.axes_Orig_Hist.set_ylabel("Number of Pixels")
            self.axes_Orig_Hist.axes.set_title("Histogram")
            self.axes_Orig_Hist.axes.title.set_color('white')

            equalize = (cdf - cdf.min()) * 255 / ((self.img.shape[0] * self.img.shape[1]) - cdf.min())
            equalizedImage = img_hsv
            equalizedImage[:, :, 2] = equalize[img_hsv[:, :, 2]]

            img_output = cv2.cvtColor(equalizedImage, cv2.COLOR_HSV2RGB)

            hist_gray_equalized, bins_gray = np.histogram(equalizedImage[:, :, 2], 256, [0, 255])
            cdf_equalized = hist_gray_equalized.cumsum()
            cdf_normalized_equalized = cdf_equalized * float(hist_gray_equalized.max()) / cdf_equalized.max()

            self.axes_Filt_image.imshow(img_output)
            self.axes_Filt_Hist.clear()
            self.axes_Filt_Hist.hist(equalizedImage[:, :, 2].flatten(), bins=bins_gray)
            self.axes_Filt_Hist.plot(cdf_normalized_equalized, color='r')
            self.axes_Filt_Hist.legend(('CDF', 'Histogram'), loc='upper right')
            self.axes_Filt_Hist.set_xlabel("Intensity Level")
            self.axes_Filt_Hist.set_ylabel("Number of Pixels")
            self.axes_Filt_Hist.axes.set_title("Histogram")
            self.axes_Filt_Hist.axes.title.set_color('white')

        elif self.comboBox_Hist_Scale.currentText() == 'Gray Scale':
            self.axes_Orig_image.imshow(self.img, cmap='gray')  ##original image in the histogram tab
            hist_gray, bins_gray = np.histogram(self.img.flatten(), 256, [0, 255])
            cdf = hist_gray.cumsum()
            cdf_normalized = cdf * float(hist_gray.max()) / cdf.max()

            self.axes_Orig_Hist.clear()
            self.axes_Orig_Hist.hist(self.img.flatten(), bins=bins_gray)
            self.axes_Orig_Hist.plot(cdf_normalized, color='r')
            self.axes_Orig_Hist.legend(('CDF', 'Histogram'), loc='upper right')
            self.axes_Orig_Hist.set_xlabel("Gray Level")
            self.axes_Orig_Hist.set_ylabel("Number of Pixels")
            self.axes_Orig_Hist.axes.set_title("Histogram")
            self.axes_Orig_Hist.axes.title.set_color('white')

            equalize = (cdf - cdf.min()) * 255 / ((self.img.shape[0] * self.img.shape[1]) - cdf.min())
            equalizedImage = equalize[self.img]

            hist_gray_equalized, bins_gray = np.histogram(equalizedImage.flatten(), 256, [0, 255])
            cdf_equalized = hist_gray_equalized.cumsum()
            cdf_normalized_equalized = cdf_equalized * float(hist_gray_equalized.max()) / cdf_equalized.max()

            self.axes_Filt_image.imshow(equalizedImage, cmap='gray')
            self.axes_Filt_Hist.clear()
            self.axes_Filt_Hist.hist(equalizedImage.flatten(), bins=bins_gray)
            self.axes_Filt_Hist.plot(cdf_normalized_equalized, color='r')
            self.axes_Filt_Hist.legend(('CDF', 'Histogram'), loc='upper right')
            self.axes_Filt_Hist.set_xlabel("Gray Level")
            self.axes_Filt_Hist.set_ylabel("Number of Pixels")
            self.axes_Filt_Hist.axes.set_title("Histogram")
            self.axes_Filt_Hist.axes.title.set_color('white')

        ##############

        self.canvas_Orig_Freq.draw()  ##apply changes
        self.canvas_Orig_Spat.draw()  ##apply changes
        self.canvas_Orig_image.draw()  ##apply changes
        self.canvas_Filt_image.draw()  ##apply changes

        self.canvas_Orig_Hist.draw()  ##apply changes
        self.canvas_Filt_Hist.draw()  ##apply changes
        self.filtering()

    # the input is rbg or grayscale image
    def low_pass_spatial_filter(self, image):
        kernel1 = np.ones((5, 5)) / 25
        # Applying the filter2D() function
        filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
        try:
            _, _ = image.shape
            low_passF = np.fft.fft2(filtered_image)
            low_passFshift = np.fft.fftshift(low_passF)
            low_pass_magnitude_spectrum = 20 * np.log(np.abs((low_passFshift) + 1))
        except:
            img_g = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
            low_passF = np.fft.fft2(img_g)
            low_passFshift = np.fft.fftshift(low_passF)
            low_pass_magnitude_spectrum = 20 * np.log(np.abs((low_passFshift) + 1))

        return filtered_image, low_pass_magnitude_spectrum

        # the input is rbg or grayscale image

    def high_pass_spatial_filter(self, image):
        kernel2 = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
        # Applying the filter2D() function
        filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
        try:
            _, _ = image.shape
            high_passF = np.fft.fft2(filtered_image)
            high_passFshift = np.fft.fftshift(high_passF)
            high_pass_magnitude_spectrum = 20 * np.log(np.abs((high_passFshift) + 1))
        except:
            img_g = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
            high_passF = np.fft.fft2(img_g)
            high_passFshift = np.fft.fftshift(high_passF)
            high_pass_magnitude_spectrum = 20 * np.log(np.abs((high_passFshift) + 1))

        return filtered_image, high_pass_magnitude_spectrum

    # the input is rbg or grayscale image

    def low_pass_freq_filter(self, img):
        # Output is a 2D complex array. 1st channel real and 2nd imaginary
        # For fft in opencv input image needs to be converted to float32
        # dft=cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        try:
            flag = 0
            _, _ = img.shape
            dft = np.fft.fft2(img)
        except:
            flag = 1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            dft = np.fft.fft2(img[:, :, 2])
        # Rearranges a Fourier transform X by shifting the zero-frequency
        # component to the center of the array.
        # Otherwise it starts at the tope left corenr of the image (array)
        dft_shift = np.fft.fftshift(dft)
        ##Magnitude of the function is 20.log(abs(f))
        # For values that are 0 we may end up with indeterminate values for log.
        # So we can add 1 to the array to avoid seeing a warning.
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
        # --------------------------------------------------------------------------------------------------
        # Circular HPF mask, center circle is 0, remaining all ones
        # Can be used for edge detection because low frequencies at center are blocked
        # and only high frequencies are allowed. Edges are high frequency components.
        # Amplifies noise.

        rows, cols = dft.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols), np.uint8)

        r = 75
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r * r
        mask[mask_area] = 0
        # apply mask and inverse DFT
        fshift = dft_shift * mask
        # --------------------------------------------------------------------------------------------------
        # revers from the freq domain to spatial domain

        fshift_mask_mag = 20 * np.log(np.abs(fshift) + 1)

        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_back)

        if (flag == 1):
            img[:, :, 2] = img_filtered
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            return img, magnitude_spectrum, fshift_mask_mag

        return img_filtered, magnitude_spectrum, fshift_mask_mag

    # the input is rbg or grayscale image

    def high_pass_freq_filter(self, img):
        # Output is a 2D complex array. 1st channel real and 2nd imaginary
        # For fft in opencv input image needs to be converted to float32
        # dft=cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        try:
            flag = 0
            _, _ = img.shape
            dft = np.fft.fft2(img)
        except:
            flag = 1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            dft = np.fft.fft2(img[:, :, 2])
        # Rearranges a Fourier transform X by shifting the zero-frequency
        # component to the center of the array.
        # Otherwise it starts at the tope left corenr of the image (array)
        dft_shift = np.fft.fftshift(dft)
        ##Magnitude of the function is 20.log(abs(f))
        # For values that are 0 we may end up with indeterminate values for log.
        # So we can add 1 to the array to avoid seeing a warning.
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
        # --------------------------------------------------------------------------------------------------
        # Circular HPF mask, center circle is 0, remaining all ones
        # Can be used for edge detection because low frequencies at center are blocked
        # and only high frequencies are allowed. Edges are high frequency components.
        # Amplifies noise.

        rows, cols = dft.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols), np.uint8)

        r = 40
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0
        # apply mask and inverse DFT
        fshift = dft_shift * mask
        # --------------------------------------------------------------------------------------------------
        # revers from the freq domain to spatial domain

        fshift_mask_mag = 20 * np.log(np.abs(fshift) + 1)

        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_back)

        if (flag == 1):
            img[:, :, 2] = img_filtered
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            return img, magnitude_spectrum, fshift_mask_mag

        return img_filtered, magnitude_spectrum, fshift_mask_mag

    def median(self, img):

        median = cv2.medianBlur(img, 11)
        try:
            median_gray = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
        except:
            median_gray = median
        medianF = np.fft.fft2(median_gray)
        medianFshift = np.fft.fftshift(medianF)
        median_magnitude_spectrum = 20 * np.log(np.abs(medianFshift) + 1)
        return median, median_magnitude_spectrum

    def laplace(self, img):

        try:
            flag = 0
            _, _ = img.shape
            laplaceSrc = cv2.GaussianBlur(img, (3, 3), 0)
            filteredLaplacian = cv2.Laplacian(laplaceSrc, cv2.CV_16S, ksize=7)

        except:
            filteredLaplacian_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_v = filteredLaplacian_hsv[:, :, 2]
            flag = 1
            laplaceSrc = cv2.GaussianBlur(img_v, (3, 3), 0)
            filteredLaplacian = cv2.Laplacian(laplaceSrc, cv2.CV_16S, ksize=7)

        laplaceF = np.fft.fft2(filteredLaplacian)
        laplaceFshift = np.fft.fftshift(laplaceF)
        laplace_magnitude_spectrum = 20 * np.log(np.abs(laplaceFshift))

        if (flag == 1):
            filteredLaplacian_hsv[:, :, 2] = filteredLaplacian
            filteredLaplacian_rgb = cv2.cvtColor(filteredLaplacian_hsv, cv2.COLOR_HSV2RGB)
            return filteredLaplacian_rgb, laplace_magnitude_spectrum

        return filteredLaplacian, laplace_magnitude_spectrum

    def filtering(self):  # this function is called when the combobox is changed
        if self.comboBox_color_filters.currentText() == 'Gray Scale Mode':
            self.axes_Orig_Spat.imshow(self.img, cmap='gray')

            if self.comboBox_filters.currentText() == 'Laplace':

                filteredLaplacian, laplace_magnitude_spectrum = self.laplace(self.img)
                self.axes_Filt_Spat.imshow(filteredLaplacian, cmap='gray')
                self.axes_Filt_Freq.imshow(laplace_magnitude_spectrum.astype('uint8'), cmap='gray')

            elif self.comboBox_filters.currentText() == 'Median':
                median, median_magnitude_spectrum = self.median(self.img)
                self.axes_Filt_Spat.imshow(median, cmap='gray')
                self.axes_Filt_Freq.imshow(median_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'Low Pass Spatial':
                filtered_img, low_pass_magnitude_spectrum = self.low_pass_spatial_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'High Pass Spatial':
                filtered_img, high_pass_magnitude_spectrum = self.high_pass_spatial_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'Low Pass Freq':
                filtered_img, _, low_pass_magnitude_spectrum = self.low_pass_freq_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'High Pass Freq':
                filtered_img, _, high_pass_magnitude_spectrum = self.high_pass_freq_filter(self.img)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'No Filter':
                self.axes_Filt_Spat.imshow(self.img, cmap='gray')
                self.axes_Filt_Freq.imshow(magnitude_spectrum, cmap='gray')  ## frequency domian




        elif self.comboBox_color_filters.currentText() == 'RGB Scale Mode':

            self.axes_Orig_Spat.imshow(self.img_rgb)
            if self.comboBox_filters.currentText() == 'Laplace':
                filteredLaplacian, laplace_magnitude_spectrum = self.laplace(self.img)
                self.axes_Filt_Spat.imshow(filteredLaplacian, cmap='gray')
                self.axes_Filt_Freq.imshow(laplace_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'Median':
                median, median_magnitude_spectrum = self.median(self.img_rgb)
                self.axes_Filt_Spat.imshow(median, cmap='gray')
                self.axes_Filt_Freq.imshow(median_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'Low Pass Spatial':
                filtered_img, low_pass_magnitude_spectrum = self.low_pass_spatial_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'High Pass Spatial':
                filtered_img, high_pass_magnitude_spectrum = self.high_pass_spatial_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'Low Pass Freq':
                filtered_img, _, low_pass_magnitude_spectrum = self.low_pass_freq_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(low_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'High Pass Freq':
                filtered_img, _, high_pass_magnitude_spectrum = self.high_pass_freq_filter(self.img_rgb)
                self.axes_Filt_Spat.imshow(filtered_img, cmap='gray')
                self.axes_Filt_Freq.imshow(high_pass_magnitude_spectrum.astype('uint8'), cmap='gray')
            elif self.comboBox_filters.currentText() == 'No Filter':
                self.axes_Filt_Spat.imshow(self.img_rgb)
                self.axes_Filt_Freq.imshow(magnitude_spectrum, cmap='gray')  ## frequency domian

        self.canvas_Filt_Freq.draw()  ##apply changes
        self.canvas_Filt_Spat.draw()  ##apply changes
        self.canvas_Orig_Spat.draw()  ##apply changes


app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet())
window = MainWindow()
app.exec_()

# class application(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.gui=GUI()
#         self.gui.setup(self)


# def window():
#     app = QApplication(sys.argv)
#     app.setStyleSheet(qdarkstyle.load_stylesheet())
#     win = application()
#     win.show()
#     sys.exit(app.exec_())


# # main code
# if __name__ == "__main__":
#     window()
