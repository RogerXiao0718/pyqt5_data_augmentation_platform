import math
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import random
import cv2

random.seed(7414)


class MainWindow_controller(QtWidgets.QMainWindow):

   ROISelectionActivated = False

   # 建構子
   def __init__(self):
      super(MainWindow_controller, self).__init__()
      self.ui = Ui_MainWindow()
      self.ui.setupUi(self)
      self.setup_control()
      self.event_binding()


   # 設定元件
   def setup_control(self):
      pass

   # 負責綁定事件
   def event_binding(self):
      pass