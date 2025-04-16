from PySide6.QtWidgets import (QGroupBox, QLineEdit, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QTextEdit, QPushButton, 
                               QMessageBox, QFormLayout, QFileDialog, 
                               QSpinBox)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from src.read import read_data_from_txt
from src.plots import plot_temperatura_pico
from neural import NeuralNetwork

class DataComparisonScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File selection controls
        file_group = QGroupBox("entrada de dados")
        file_layout = QVBoxLayout()
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        
        browse_button = QPushButton("procurar...")
        browse_button.clicked.connect(self.browse_file)
        
        file_controls = QHBoxLayout()
        file_controls.addWidget(QLabel("arquivo selecionado:"))
        file_controls.addWidget(self.file_path_edit)
        file_controls.addWidget(browse_button)
        
        file_layout.addLayout(file_controls)
        file_group.setLayout(file_layout)
        
        # Comparison visualization
        self.comparison_figure = plt.figure(figsize=(10, 6))
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        # Compare button
        compare_button = QPushButton("comparar com a rede neural")
        compare_button.clicked.connect(self.compare_data)
        
        # Add widgets to main layout
        layout.addWidget(file_group)
        layout.addWidget(self.comparison_canvas)
        layout.addWidget(QLabel("resultados:"))
        layout.addWidget(self.results_text)
        layout.addWidget(compare_button)
        
        self.setLayout(layout)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
    
    def compare_data(self):
        if not self.file_path_edit.text():
            QMessageBox.warning(self, "Warning", "Please select a data file first")
            return
        
        try:
            # Load data from file
            print("reading data...")
            data = read_data_from_txt(self.file_path_edit.text()).astype(np.float32)


            data = pd.DataFrame(data, columns=[
                'espessura', 'comprimento', 'largura', 'velocidade', 'sigma',
                'potencia', 'temp. amb.', 'cal. esp.', 'cond. term.', 'densidade',
                'x', 'y', 'tempo', 'temperatura'
            ])

            data = data.loc[data.groupby([
                'espessura', 'comprimento', 'largura', 'velocidade', 'sigma',
                'potencia', 'temp. amb.', 'cal. esp.', 'cond. term.', 'densidade',
                'x', 'y'
            ])['temperatura'].idxmax()]

            data = data.drop(columns=['tempo']).to_numpy().astype(np.float32)

            print(data.shape)
            
            yhat = NeuralNetwork.inference_data(data[:, :-1])
            yhat = np.clip(yhat, None, 1450).reshape(-1)
            data[:, -1] = np.clip(data[:, -1], None, 1450)

            # Plot comparison
            self.plot_comparison(data[:, -3:], yhat)
            
            # Calculate and display metrics
            self.display_comparison_metrics(data[:, -1], yhat)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process file: {str(e)}")
    
    def generate_nn_comparison_data(self, parameters, length):
        """Generate data from neural network for comparison"""
        # This would be replaced with your actual NN code
        x = np.linspace(0, 10, length)
        y = np.sin(x) * parameters.get('param2', 5) + np.random.normal(0, 0.5, length)
        return y
    
    def plot_comparison(self, file_data, yhat):
        """Plot the comparison between file data and NN output"""
        self.comparison_figure.clear()

        ax1 = self.comparison_figure.add_subplot(121)

        ax1 = plot_temperatura_pico(ax1, file_data)

        #file_data[:, -1] = yhat

        ax2 = self.comparison_figure.add_subplot(122)

        ax2 = plot_temperatura_pico(ax2, np.column_stack((
            file_data[:, 0], file_data[:, 1], yhat
        )))
        
        self.comparison_canvas.draw()
    
    def display_comparison_metrics(self, y, yhat):
        """Calculate and display comparison metrics"""
        if len(y) != len(yhat):
            self.results_text.setText("Error: Data lengths don't match")
            return
        
        # Calculate metrics
        mae = np.mean(np.abs(y - yhat))
        mse = np.mean((y - yhat) ** 2)
        correlation = np.corrcoef(y, yhat)[0, 1]
        
        # Display results
        results_html = f"""
        <table border="1">
            <tr><th>métrica</th><th>valor</th></tr>
            <tr><td>erro absoluto médio (MAE)</td><td>{mae:.4f}</td></tr>
            <tr><td>erro quadrático médio (MSE)</td><td>{mse:.4f}</td></tr>
            <tr><td>coeficiente de correlação</td><td>{correlation:.4f}</td></tr>
        </table>
        <p>Number of data points compared: {len(y)}</p>
        """
        self.results_text.setHtml(results_html)