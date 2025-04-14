import sys

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QTextEdit, QPushButton, 
                               QDoubleSpinBox, QFormLayout, QFileDialog, 
                               QSpinBox)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from onnxruntime import InferenceSession
import joblib
from src.utils import scale_data
from src.plots import plot_temperatura_pico


class NeuralNetwork:
    @staticmethod
    def inference(parameters):
        model = InferenceSession('models/model_0.5')
        scalers = joblib.load('files/scalers.joblib')

        input_name = model.get_inputs()[0].name

        x_step = 0.02
        y_step = 0.02
        xx = np.linspace(0.0, parameters['largura']/1000, int((parameters['espessura'])/x_step))
        yy = np.linspace(0.0, parameters['espessura']/1000, int((parameters['espessura'])/y_step))
        
        inputs = np.array(np.meshgrid(xx, yy)).T.reshape(-1,2)

        fusao = parameters['temp. fusao']
        parameters = np.array([
            parameters['espessura'],
            parameters['comprimento'], 
            parameters['largura'], 
            parameters['velocidade'],
            parameters['sigma'],
            parameters['potencia']/1000,
            parameters['tamb'],
            parameters['cal. esp.'],
            parameters['cond. term.'],
            parameters['rho']
        ])
        
        features = np.tile(parameters, (inputs.shape[0], 1))
        inputs = np.concatenate((features, inputs), axis=1).astype(np.float32)

        print('predicting:', inputs.shape)

        inputs = scale_data(inputs, scalers, scale_temp=False)

        outputs = np.zeros((inputs.shape[0], 1))

        for i in range(0, inputs.shape[0], 128):
            outputs[i:i+128] = model.run(None, {input_name: inputs[i:i+128]})[0]

        #outputs = model.run(None, {input_name: inputs})[0]

        x = scalers[-2].inverse_transform(inputs[:, -2].reshape(-1, 1)).reshape(-1)
        y = scalers[-1].inverse_transform(inputs[:, -1].reshape(-1, 1)).reshape(-1)

        return np.column_stack((x, y, np.clip(outputs, None, fusao)))


class ParameterInputWidget(QWidget):
    """Widget for entering neural network parameters"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout()
        
        # Create parameter inputs
        self.param1_input = QSpinBox()
        self.param1_input.setRange(10, 500)
        self.param1_input.setValue(100)
        self.param1_input.setSingleStep(1)
        param1_layout = QHBoxLayout()
        param1_layout.addWidget(self.param1_input)
        param1_layout.addWidget(QLabel(f"(Range: {self.param1_input.minimum()}-{self.param1_input.maximum()})"))
        layout.addRow("comprimento da chapa (mm) (10 - 500):", self.param1_input)
        
        self.param2_input = QSpinBox()
        self.param2_input.setRange(10, 500)
        self.param2_input.setValue(100)
        self.param2_input.setSingleStep(1)
        layout.addRow("largura da chapa (mm) (10 - 500):", self.param2_input)
        
        self.param3_input = QDoubleSpinBox()
        self.param3_input.setRange(1.0, 20.0)
        self.param3_input.setValue(5.0)
        self.param3_input.setSingleStep(0.1)
        layout.addRow("espessura da chapa (mm) (1.0 - 20.0):", self.param3_input)

        self.param4_input = QSpinBox()
        self.param4_input.setRange(1000, 10000)
        self.param4_input.setValue(2000)
        self.param4_input.setSingleStep(1)
        layout.addRow("potência (W) (1000 - 10000):", self.param4_input)

        self.param5_input = QDoubleSpinBox()
        self.param5_input.setRange(6.0, 600.0)
        self.param5_input.setValue(100.0)
        self.param5_input.setSingleStep(0.1)
        layout.addRow("velocidade (cm/min) (6.0 - 600.0):", self.param5_input)

        self.param6_input = QDoubleSpinBox()
        self.param6_input.setRange(0.1, 10.0)
        self.param6_input.setValue(2.4)
        self.param6_input.setSingleStep(0.1)
        layout.addRow("sigma fonte (mm) (0.1 - 10.0):", self.param6_input)

        self.param7_input = QSpinBox()
        self.param7_input.setRange(10, 45)
        self.param7_input.setValue(25)
        self.param7_input.setSingleStep(1)
        layout.addRow("temperatura ambiente (C) (10 - 45):", self.param7_input)

        self.param8_input = QSpinBox()
        self.param8_input.setRange(2500, 10000)
        self.param8_input.setValue(4000)
        self.param8_input.setSingleStep(1)
        layout.addRow("densidade do material (kg/m3) (2500 - 10000):", self.param8_input)

        self.param9_input = QSpinBox()
        self.param9_input.setRange(5, 450)
        self.param9_input.setValue(100)
        self.param9_input.setSingleStep(1)
        layout.addRow("condutividade térmica (W/mC) (5 - 450):", self.param9_input)

        self.param10_input = QSpinBox()
        self.param10_input.setRange(250, 2000)
        self.param10_input.setValue(500)
        self.param10_input.setSingleStep(1)
        layout.addRow("calor especifico (J/kgC) (250 - 2000):", self.param10_input)

        self.param11_input = QSpinBox()
        self.param11_input.setRange(0, 10000)
        self.param11_input.setValue(1450)
        self.param11_input.setSingleStep(1)
        layout.addRow("temperatura de fusão (C):", self.param11_input)
        
        self.setLayout(layout)
    
    def get_parameters(self):
        """Return the current parameters as a dictionary"""
        return {
            'comprimento': self.param1_input.value(),
            'largura': self.param2_input.value(),
            'espessura': self.param3_input.value(),
            'potencia': self.param4_input.value(),
            'velocidade': self.param5_input.value(),
            'sigma': self.param6_input.value(),
            'tamb': self.param7_input.value(),
            'rho': self.param8_input.value(),
            'cond. term.': self.param9_input.value(),
            'cal. esp.': self.param10_input.value(),
            'temp. fusao': self.param11_input.value()
        }


class ImageDisplayWidget(QWidget):
    """Widget for displaying matplotlib figures"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        self.figure = None
        self.canvas = None
        self.figure_container = QWidget()
        self.figure_layout = QVBoxLayout(self.figure_container)

        # Annotations panel on the right
        self.annotations_panel = QTextEdit()
        self.annotations_panel.setReadOnly(True)
        self.annotations_panel.setMinimumWidth(200)
        self.annotations_panel.setStyleSheet("""
            QTextEdit {
                font-family: monospace;
            }
        """)
        
        # Add widgets to main layout
        layout.addWidget(self.figure_container, 4)  # 80% width
        layout.addWidget(self.annotations_panel, 1)  # 20% width
        
        self.setLayout(layout)
    
    def display_figure(self, figure, parameters=None):
        """Display a matplotlib figure in the widget"""
        if self.canvas:
            self.layout().removeWidget(self.canvas)
            self.canvas.deleteLater()
        
        self.canvas = FigureCanvas(figure)
        self.layout().addWidget(self.canvas)
        self.canvas.draw()

        if parameters:
            self.update_annotations(parameters)

    def update_annotations(self, parameters):
        """Update the annotation panel with parameters"""
        annotation_text = (
            "<table>"
            "<p>Respostas(mm):<p>"
            f"<tr><td><b>x (1ºC):</b></td><td>{parameters.get('x1', 'not find')}</td></tr>"
            f"<tr><td><b>x (0.5ºC):</b></td><td>{parameters.get('x05', 'not find')}</td></tr>"
            f"<tr><td><b>x (0.1ºC):</b></td><td>{parameters.get('x01', 'not find')}</td></tr>"
            "</table>"
            "<p>Temperaturas médias(C)<p>"
            "<table>"
        )

        if parameters.get('x1', None) is not None:
            annotation_text += f"<tr><td><b>{parameters.get('x1')}mm:</b></td><td>{parameters.get('t1'):.2f} C</td></tr>"
        if parameters.get('x05', None) is not None:
            annotation_text += f"<tr><td><b>{parameters.get('x05')}mm:</b></td><td>{parameters.get('t05'):.2f} C</td></tr>"
        if parameters.get('x01', None) is not None:
            annotation_text += f"<tr><td><b>{parameters.get('x01')}mm:</b></td><td>{parameters.get('t01'):.2f} C</td></tr>"

        self.annotations_panel.setHtml(annotation_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("temperatura de pico - modelo neural")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create widgets
        self.parameter_widget = ParameterInputWidget()
        self.image_widget = ImageDisplayWidget()
        
        # Create buttons
        self.generate_button = QPushButton("gerar mapa de cores")
        self.generate_button.clicked.connect(self.generate_image)
        
        self.save_button = QPushButton("salvar imagem")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        
        # Current image storage
        self.current_image = None
        
        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.save_button)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.parameter_widget)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_widget)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    
    def generate_image(self):
        """Generate and display an image based on current parameters"""
        parameters = self.parameter_widget.get_parameters()
        
        # In a real application, you would send these parameters to your neural network
        print("Sending parameters to neural network:", parameters)
        
        # For this demo, we'll use our simulator
        outputs = NeuralNetwork.inference(parameters)

        xs = np.unique(outputs[:, 0])
        lim = {}
        for x in np.sort(xs):
            i = np.where(outputs[:, 0] == x)[0]

            if (max(outputs[i, 2]) - min(outputs[i, 2])) < 1 and not lim.get('x1', None):
                lim['x1'] = round(x * 1000, 1)
                lim['t1'] = np.mean(outputs[i, 2])
    
            if (max(outputs[i, 2]) - min(outputs[i, 2])) < 0.5 and not lim.get('x05', None):
                lim['x05'] = round(x * 1000,1)
                lim['t05'] = np.mean(outputs[i, 2])

            if (max(outputs[i, 2]) - min(outputs[i, 2])) < 0.1 and not lim.get('x01', None):
                lim['x01'] = round(x * 1000,1)
                lim['t01'] = np.mean(outputs[i, 2])
                break

        fig = plt.figure(figsize=(12, 8))
        fig = plot_temperatura_pico(fig, outputs, plot_points=False)

        if lim.get('x1', None):
            plt.vlines(x=lim['x1']/1000, ymin=-0.00, ymax=parameters['espessura']/1000, colors='black', label=f'x 1ºC = {str(round(lim["x1"], 1))}mm')
            if lim.get('x05', None):
                plt.vlines(x=lim['x05']/1000, ymin=-0.00, ymax=parameters['espessura']/1000, colors='black', label=f'x 0.5ºC = {str(round(lim["x05"], 1))}mm')
                if lim.get('x01', None):
                    plt.vlines(x=lim['x01']/1000, ymin=-0.00, ymax=parameters['espessura']/1000, colors='black', label=f'x 0.1ºC = {str(round(lim["x01"], 1))}mm')
        
            plt.legend(loc='upper left')

        self.current_figure = fig
        
        # Display the image
        self.image_widget.display_figure(fig, parameters=lim)
        self.save_button.setEnabled(True)
    
    def save_image(self):
        """Save the current figure to a file"""
        if not self.current_figure:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            self.current_figure.savefig(file_path)


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