import numpy as np
import tensorflow as tf

from src.utils import load_label_map


class EmotionPredictor(object):
    def __init__(self, path_to_model, path_to_labels, nms_threshold=0.15, score_threshold=0.3):
        self.path_to_model = path_to_model
        self.path_to_labels = path_to_labels
        self.category_index = load_label_map.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

        # load model
        self.interpreter = self.load_model()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def load_model(self):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=self.path_to_model)
        interpreter.allocate_tensors()

        return interpreter
    
    def predict(self, mel_spec):
        self.interpreter.set_tensor(self.input_details[0]['index'], mel_spec)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        arr_result = []
        for k, v in self.category_index.items():
            mapped_data = {}
            mapped_data['key'] = v['name']
            mapped_data['value'] = round(float(output_data[k-1]), 2)
            arr_result.append(mapped_data)

        return arr_result