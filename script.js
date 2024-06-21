// Access the camera
navigator.mediaDevices.getUserMedia({ video: true })
.then(function (stream) {
    var video = document.getElementById('video');
    video.srcObject = stream;
    video.onloadedmetadata = function (e) {
    video.play();
    }
})
.catch(function (error) {
    console.error('Error accessing the camera: ', error);
});

// Capture image from the camera
function captureImage() {
var video = document.getElementById('video');
var canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
var context = canvas.getContext('2d');
context.drawImage(video, 0, 0, canvas.width, canvas.height);

// Convert the canvas to a data URL and pass it to the next page
var imageData = canvas.toDataURL('image/png');
window.location.href = 'disease.html?image=' + encodeURIComponent(imageData);
}

document.addEventListener('DOMContentLoaded', (event) => {
    const video = document.getElementById('camera');
    const captureButton = document.getElementById('captureButton');

    // Get user media (camera)
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((error) => {
            console.error('Error accessing camera:', error);
        });

    // Capture Image
    captureButton.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        // Draw the current frame from the video onto the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the canvas image to data URL
        const imageData = canvas.toDataURL('image/png');

        // Redirect to the next page with the image data as a URL parameter
        window.location.href = `nextPage.html?imageData=${encodeURIComponent(imageData)}`;
    });
});
