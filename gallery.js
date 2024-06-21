document.getElementById('upload').addEventListener('change', handleFileSelect);

function handleFileSelect(event) {
    const files = event.target.files;
    const gallery = document.getElementById('gallery');

    for (const file of files) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();

            reader.onload = function (e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                gallery.appendChild(img);
            };

            reader.readAsDataURL(file);
        }
    }
}
