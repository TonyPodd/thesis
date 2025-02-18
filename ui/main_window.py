import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sport Mafia Automation - Demo UI")
        self.setGeometry(100, 100, 400, 300)

        # Элементы интерфейса
        self.log_area = QTextEdit(self)
        self.log_area.setReadOnly(False)
        self.log_area.append("Это тестовая зона лога.\n")

        self.btn_test = QPushButton("Нажми меня", self)
        self.btn_test.clicked.connect(self.on_button_click)

        # Размещаем всё в вертикальном layout
        layout = QVBoxLayout()
        layout.addWidget(self.btn_test)
        layout.addWidget(self.log_area)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_button_click(self):
        self.log_area.append("Кнопка была нажата!\n")

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()