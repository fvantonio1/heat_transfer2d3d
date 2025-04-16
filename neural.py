import numpy as np
from onnxruntime import InferenceSession
import joblib
from src.utils import scale_data

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
    
    @staticmethod
    def inference_data(data):
        model = InferenceSession('models/model_0.5')
        scalers = joblib.load('files/scalers.joblib')

        input_name = model.get_inputs()[0].name

        inputs = scale_data(data, scalers, scale_temp=False)

        outputs = np.zeros((inputs.shape[0], 1))

        for i in range(0, inputs.shape[0], 128):
            outputs[i:i+128] = model.run(None, {input_name: inputs[i:i+128]})[0]

        return outputs