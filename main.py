from img_filter_gui import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap


class GUI (Ui_MainWindow):
    def setup(self,MainWindow):
        super().setupUi(MainWindow)
        ##wriet gui modificaion here##
        # pixmap = QPixmap('testimage.jpeg')
        # self.label_Orig_Spat.setPixmap(pixmap)

        # self.label_Orig_Friq.setPixmap(pixmap)
        
                





class application(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.gui=GUI()
        self.gui.setup(self)


    ##Write code here##

    
        




def window():
    app = QApplication(sys.argv)
    win = application()
    win.show()
    sys.exit(app.exec_())


# main code
if __name__ == "__main__":
    window()
    
    
