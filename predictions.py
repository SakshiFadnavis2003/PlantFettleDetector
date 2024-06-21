import numpy as np
import cv2
import tensorflow as tf

# Load the TFLite model
model_path = "Fruit_quality.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load an image from C drive (replace 'path/to/your/image.jpg' with the actual path)
# Corrected image_path with a raw string
image_path = "pinta.jpg"

image = cv2.imread(image_path)

# Preprocess the image based on your model requirements
# Replace this preprocessing code with the specific preprocessing steps for your model
# Example: resizing the image to match the model input size
input_shape = input_details[0]['shape']
preprocessed_image = cv2.resize(image, (input_shape[2], input_shape[1]))
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

# Convert image data type to FLOAT32
preprocessed_image = preprocessed_image.astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

# Run the interpreter
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])

# Assuming actual_class is the actual class label for the given image
actual_class = 'Bad Quality_Fruits'  # Replace with the actual class label

# Predicted class and probability
predicted_class_index = np.argmax(output_data[0])  # Use [0] to access the array of probabilities
predicted_probability = output_data[0][predicted_class_index]

# Load class labels (replace with your actual class labels)
class_labels = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Qualit_Fruits']

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]

# Print results
print("Class Labels:", class_labels)
print("Predicted Class Index:", predicted_class_index)
print("Predicted Class:", predicted_class_label)
print("Predicted Probability:", predicted_probability)
print("Output Probabilities:", output_data)
print("Input Image Shape:", image.shape)
print("Input Tensor Shape:", preprocessed_image.shape)

