#!/usr/bin/python3
# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets
import sys
import argparse

from qt_tabs import Tab1, Tab2, Tab3, Tab4

class UI(Tab1, Tab2, Tab3, Tab4):
    def __init__(self, debug=False, debug_page=None, debug_opt=None, is_docker=False):
        super(UI, self).__init__() # Call the inherited classes __init__ method


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action='store_true', default=False, help="is in docker")
    parser.add_argument("--debug", action='store_true', default=False, help="is debug mode")
    parser.add_argument("--page", type=int, default=0, help="")
    parser.add_argument("--opt", type=str, default=None, help="")
    args = parser.parse_args()

    itao = UI()
    
    itao.start(debug=args.debug,
        debug_page=args.page,
        debug_opt=args.opt,
        is_docker=args.docker
        )
    
    sys.exit(app.exec_())