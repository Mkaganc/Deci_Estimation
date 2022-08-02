from tensorflow.keras.models import *
import numpy as np

def load_neural_network_model_from_path(json_file_path):
    """Loads neural network and weights from the given file paths """

    json_file = open(json_file_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    print("Loaded model from disk.")
    return loaded_model


def load_h5_weights_file_to_model(model, h5_path):
    """Returns model with weights from given path."""
    model.load_weights(h5_path)
    print("Loaded weights from disk.")
    return model


def predict_model(model, calculated_deci, total_volume_cm3, total_weight_gram):
    numpy_obj = np.array([calculated_deci, total_volume_cm3, total_weight_gram], ndmin=2)
    # numpy_obj = tf.convert_to_tensor(numpy_obj, dtype=tf.float32)
    # prediction = np.argmax(my_model.predict(numpy_obj), axis=-1)
    prediction = model.predict(numpy_obj)
    print(prediction*100)


json_file_path = 'presmodel.json'          # you must specify json files path
weights_file_path = 'presmodel.h5'       # you must specify h5 files path

my_model = load_neural_network_model_from_path(json_file_path)

my_weights_loaded_model = load_h5_weights_file_to_model(my_model, weights_file_path)

predict_model(my_weights_loaded_model, 0.056, 168.912, 126.6)  # This call prints screen the prediction.

