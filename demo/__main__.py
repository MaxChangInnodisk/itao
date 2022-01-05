#!/usr/bin/python3
# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets
import sys

from qt_tabs import Tab1, Tab2, Tab3, Tab4

class UI(Tab1, Tab2, Tab3, Tab4):
    def __init__(self):
        super(UI, self).__init__() # Call the inherited classes __init__ method

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    itao = UI()
    itao.showFullScreen()
    itao.show()
    itao.show_warning_msg()
    sys.exit(app.exec_())