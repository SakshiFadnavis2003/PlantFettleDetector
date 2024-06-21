# import h5py
# from keras.models import load_model, model_from_json

# # Load the HDF5 model
# hdf5_file = "Plant_Disease.h5"
# loaded_model = load_model(hdf5_file)

# # Convert the model to JSON
# json_model = loaded_model.to_json()

# # Save the JSON representation to a file
# json_file = "Plant_Disease.json"
# with open(json_file, "w") as file:
#     file.write(json_model)

# # You can also load the JSON model back if needed
# loaded_json_model = model_from_json(json_model)
import tensorflowjs as tfjs
# from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the Keras model
MODEL = tf.keras.models.load_model("Plant_Disease.h5")

# Convert the Keras model to TensorFlow.js format
tfjs.converters.save_keras_model(MODEL, "tfjs_model")

print("Conversion completed. TensorFlow.js model saved in the 'tfjs_model' directory.")
