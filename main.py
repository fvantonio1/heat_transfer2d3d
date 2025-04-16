import sys

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QStackedWidget)
from PySide6.QtGui import QFont

from comparison_screen import DataComparisonScreen
from parameter_screen import ParameterScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("temperatura de pico - modelo neural")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create widgets
        self.parameter_widget = ParameterScreen(self)
        # self.image_widget = ImageDisplayWidget()
        self.comparison_screen = DataComparisonScreen(self)

        # Stacked widget to handle screens
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.parameter_widget)
        self.stacked_widget.addWidget(self.comparison_screen)

        # Navigation controls
        nav_layout = QHBoxLayout()
        param_button = QPushButton("entrada de parâmetros")
        param_button.clicked.connect(lambda: self.switch_screen(0))
        compare_button = QPushButton("comparação entre modelo neural e simulação")
        compare_button.clicked.connect(lambda: self.switch_screen(1))
        nav_layout.addWidget(param_button)
        nav_layout.addWidget(compare_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.stacked_widget)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
    def switch_screen(self, index):
        self.stacked_widget.setCurrentIndex(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamily("mry_KacstQurn")
    font.setPointSize(10)
    font.setWeight(QFont.Weight.Normal)
    app.setFont(font)

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())