# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QTableView, QHeaderView
from PyQt5.QtCore import QAbstractTableModel, Qt
from graphics import Ui_MainWindow as graphicsWindow

import pandas as pd
from detection import Detection

class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class Ui_MainWindow(object):
    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = graphicsWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    def train(self):
        test_size = float(self.test_size_lineEdit.text())
        det = Detection()
        results = det.run(test_size)
        model = pandasModel(results)
        self.train_tableView.setModel(model)
        header = self.train_tableView.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.train_tableView.show()

    def sample(self):
        det = Detection()
        results, pred = det.predict()
        model = pandasModel(results)
        self.sample_tableView.setModel(model)
        header = self.sample_tableView.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.sample_tableView.show()
        if pred==1:
            pred = "Predicted: Malicious"
        else:
            pred = "Predicted: Benign"
        self.predicted_label.setText(pred)
        label = results.iloc[0]['label']
        print(label)
        if label=="bad":
            label = "Actual: Malicious"
        else:
            label = "Actual: Benign"  
        self.actual_label.setText(label)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(10, 10, 781, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.test_size_label = QtWidgets.QLabel(self.centralwidget)
        self.test_size_label.setGeometry(QtCore.QRect(130, 80, 61, 31))
        self.test_size_label.setAlignment(QtCore.Qt.AlignCenter)
        self.test_size_label.setObjectName("test_size_label")
        self.test_size_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.test_size_lineEdit.setGeometry(QtCore.QRect(200, 80, 91, 31))
        self.test_size_lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.test_size_lineEdit.setObjectName("test_size_lineEdit")
        self.train_pushButton = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.train())
        self.train_pushButton.setGeometry(QtCore.QRect(130, 130, 161, 31))
        self.train_pushButton.setObjectName("train_pushButton")
        self.results_pushButton = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.openWindow())
        self.results_pushButton.setGeometry(QtCore.QRect(510, 130, 161, 31))
        self.results_pushButton.setObjectName("results_pushButton")
        self.train_tableView = QtWidgets.QTableView(self.centralwidget)
        self.train_tableView.setGeometry(QtCore.QRect(125, 200, 550, 201))
        self.train_tableView.setBaseSize(QtCore.QSize(5, 5))
        self.train_tableView.setObjectName("train_tableView")
        self.sample_pushButton = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.sample())
        self.sample_pushButton.setGeometry(QtCore.QRect(120, 430, 161, 31))
        self.sample_pushButton.setObjectName("sample_pushButton")
        self.sample_tableView = QtWidgets.QTableView(self.centralwidget)
        self.sample_tableView.setGeometry(QtCore.QRect(120, 480, 551, 61))
        self.sample_tableView.setObjectName("sample_tableView")
        self.predicted_label = QtWidgets.QLabel(self.centralwidget)
        self.predicted_label.setGeometry(QtCore.QRect(400, 430, 101, 31))
        self.predicted_label.setObjectName("predicted_label")
        self.actual_label = QtWidgets.QLabel(self.centralwidget)
        self.actual_label.setGeometry(QtCore.QRect(570, 430, 101, 31))
        self.actual_label.setObjectName("actual_label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title.setText(_translate("MainWindow", "MALICIOUS WEBPAGE DETECTION"))
        self.test_size_label.setText(_translate("MainWindow", " Test Size"))
        self.test_size_lineEdit.setPlaceholderText(_translate("MainWindow", "0.3"))
        self.train_pushButton.setText(_translate("MainWindow", "Train"))
        self.results_pushButton.setText(_translate("MainWindow", "Results"))
        self.sample_pushButton.setText(_translate("MainWindow", "Sample"))
        self.predicted_label.setText(_translate("MainWindow", "Predicted:"))
        self.actual_label.setText(_translate("MainWindow", "Actual:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())