# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Mansour's Sons\OneDrive\Desktop\img_filter_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1209, 738)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_image_filtering = QtWidgets.QWidget()
        self.tab_image_filtering.setObjectName("tab_image_filtering")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_image_filtering)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_Fil_Spat = QtWidgets.QLabel(self.tab_image_filtering)
        self.label_Fil_Spat.setText("")
        self.label_Fil_Spat.setObjectName("label_Fil_Spat")
        self.gridLayout_2.addWidget(self.label_Fil_Spat, 2, 1, 1, 1)
        self.label_Fil_Freq = QtWidgets.QLabel(self.tab_image_filtering)
        self.label_Fil_Freq.setText("")
        self.label_Fil_Freq.setObjectName("label_Fil_Freq")
        self.gridLayout_2.addWidget(self.label_Fil_Freq, 2, 2, 1, 1)
        self.label_Orig_Friq = QtWidgets.QLabel(self.tab_image_filtering)
        self.label_Orig_Friq.setText("")
        self.label_Orig_Friq.setObjectName("label_Orig_Friq")
        self.gridLayout_2.addWidget(self.label_Orig_Friq, 1, 2, 1, 1)
        self.label_Title_Spat = QtWidgets.QLabel(self.tab_image_filtering)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_Title_Spat.setFont(font)
        self.label_Title_Spat.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Title_Spat.setObjectName("label_Title_Spat")
        self.gridLayout_2.addWidget(self.label_Title_Spat, 0, 1, 1, 1)
        self.label_Title_Freq = QtWidgets.QLabel(self.tab_image_filtering)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_Title_Freq.setFont(font)
        self.label_Title_Freq.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Title_Freq.setObjectName("label_Title_Freq")
        self.gridLayout_2.addWidget(self.label_Title_Freq, 0, 2, 1, 1)
        self.label_Title_Orig = QtWidgets.QLabel(self.tab_image_filtering)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(True)
        self.label_Title_Orig.setFont(font)
        self.label_Title_Orig.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_Title_Orig.setObjectName("label_Title_Orig")
        self.gridLayout_2.addWidget(self.label_Title_Orig, 1, 0, 1, 1)
        self.label_Title_Fil = QtWidgets.QLabel(self.tab_image_filtering)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(True)
        self.label_Title_Fil.setFont(font)
        self.label_Title_Fil.setObjectName("label_Title_Fil")
        self.gridLayout_2.addWidget(self.label_Title_Fil, 2, 0, 1, 1)
        self.label_Orig_Spat = QtWidgets.QLabel(self.tab_image_filtering)
        self.label_Orig_Spat.setText("")
        self.label_Orig_Spat.setScaledContents(False)
        self.label_Orig_Spat.setObjectName("label_Orig_Spat")
        self.gridLayout_2.addWidget(self.label_Orig_Spat, 1, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setColumnStretch(2, 1)
        self.gridLayout_2.setRowStretch(1, 1)
        self.gridLayout_2.setRowStretch(2, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.comboBox_filters = QtWidgets.QComboBox(self.tab_image_filtering)
        self.comboBox_filters.setEditable(True)
        self.comboBox_filters.setObjectName("comboBox_filters")
        self.horizontalLayout.addWidget(self.comboBox_filters)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)
        self.gridLayout_3.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_image_filtering, "")
        self.tab_histogram = QtWidgets.QWidget()
        self.tab_histogram.setObjectName("tab_histogram")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_histogram)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_hist_filt = QtWidgets.QLabel(self.tab_histogram)
        self.label_hist_filt.setText("")
        self.label_hist_filt.setObjectName("label_hist_filt")
        self.gridLayout_4.addWidget(self.label_hist_filt, 1, 1, 1, 1)
        self.label_hist_title_filt = QtWidgets.QLabel(self.tab_histogram)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(True)
        self.label_hist_title_filt.setFont(font)
        self.label_hist_title_filt.setAlignment(QtCore.Qt.AlignCenter)
        self.label_hist_title_filt.setObjectName("label_hist_title_filt")
        self.gridLayout_4.addWidget(self.label_hist_title_filt, 0, 1, 1, 1)
        self.label_hist_title_orig = QtWidgets.QLabel(self.tab_histogram)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(True)
        self.label_hist_title_orig.setFont(font)
        self.label_hist_title_orig.setAlignment(QtCore.Qt.AlignCenter)
        self.label_hist_title_orig.setObjectName("label_hist_title_orig")
        self.gridLayout_4.addWidget(self.label_hist_title_orig, 0, 0, 1, 1)
        self.label_hist_orig = QtWidgets.QLabel(self.tab_histogram)
        self.label_hist_orig.setText("")
        self.label_hist_orig.setObjectName("label_hist_orig")
        self.gridLayout_4.addWidget(self.label_hist_orig, 1, 0, 1, 1)
        self.graphicsView_orig = PlotWidget(self.tab_histogram)
        self.graphicsView_orig.setObjectName("graphicsView_orig")
        self.gridLayout_4.addWidget(self.graphicsView_orig, 2, 0, 1, 1)
        self.graphicsView_filt = PlotWidget(self.tab_histogram)
        self.graphicsView_filt.setObjectName("graphicsView_filt")
        self.gridLayout_4.addWidget(self.graphicsView_filt, 2, 1, 1, 1)
        self.gridLayout_4.setRowStretch(1, 1)
        self.gridLayout_4.setRowStretch(2, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_histogram, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1209, 26))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImage = QtWidgets.QAction(MainWindow)
        self.actionImage.setObjectName("actionImage")
        self.menuOpen.addAction(self.actionImage)
        self.menubar.addAction(self.menuOpen.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_Title_Spat.setText(_translate("MainWindow", "Spatial Domain"))
        self.label_Title_Freq.setText(_translate("MainWindow", "Frequency Domain"))
        self.label_Title_Orig.setText(_translate("MainWindow", "Original"))
        self.label_Title_Fil.setText(_translate("MainWindow", "Filtered"))
        self.comboBox_filters.setCurrentText(_translate("MainWindow", "Choose a Filter"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_image_filtering), _translate("MainWindow", "Image Filtering"))
        self.label_hist_title_filt.setText(_translate("MainWindow", "Filtered"))
        self.label_hist_title_orig.setText(_translate("MainWindow", "Original"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_histogram), _translate("MainWindow", "Histogram Equalization"))
        self.menuOpen.setTitle(_translate("MainWindow", "File"))
        self.actionImage.setText(_translate("MainWindow", "Open"))
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
