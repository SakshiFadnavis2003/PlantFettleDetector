async function loadAndRunModels() {
    const model1 = await tf.loadGraphModel('Fruit_quality.tflite');
    const model2 = await tf.loadGraphModel('Plant_Disease.tflite');
    // Replace 'model1/model.json' and 'model2/model.json' with the paths to your model files

    const video = document.getElementById('camera');
    const canvas = document.getElementById('capturedCanvas');
    const captureButton = document.getElementById('captureButton');

    let stream;
    let capturedImageData;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (cameraStream) {
            stream = cameraStream;
            video.srcObject = cameraStream;
            video.onloadedmetadata = function (e) {
                video.play();
            };
        })
        .catch(function (err) {
            console.error('Error accessing camera:', err);
        });

    captureButton.addEventListener('click', function () {
        if (stream) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            capturedImageData = context.getImageData(0, 0, canvas.width, canvas.height);

            // Perform inference using TensorFlow.js models
            performInference(model1, model2, capturedImageData);

            // Optionally, stop the camera stream
            stream.getTracks().forEach(track => track.stop());
        }
    });
}

async function performInference(model1, model2, imageData) {
    // Preprocess the image data if needed
    const inputTensor = tf.browser.fromPixels(imageData).expandDims();

    // Use model1 for inference
    const result1 = model1.predict(inputTensor);
    console.log('Result from Model 1:');
    result1.print();
    result1.dispose();

    // Use model2 for inference
    const result2 = model2.predict(inputTensor);
    console.log('Result from Model 2:');
    result2.print();
    result2.dispose();

    // Clean up
    inputTensor.dispose();
    model1.dispose();
    model2.dispose();
}

// Call the function to load and run the models
loadAndRunModels();