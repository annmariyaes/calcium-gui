
// Function to switch between tabs
function openTabs(opName, button) {
    var i;
    var x = document.getElementsByClassName("tab-content");

    // Hide all tab content
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";
    }

    // Display the selected tab content
    document.getElementById(opName).style.display = "grid";

    // Remove 'active' class from all buttons
    var buttons = document.getElementsByClassName("w3-button");
    for (i = 0; i < buttons.length; i++) {
        buttons[i].classList.remove("active");
    }

    // Add 'active' class to the clicked button
    button.classList.add("active");
}

// Set the first tab ('Intensity') as selected by default
document.addEventListener("DOMContentLoaded", function() {
    var defaultTab = document.getElementsByClassName("w3-button")[0];
    openTabs('Intensity', defaultTab);

})

function submitForm(action, tabName) {
    var form = document.getElementById(tabName);
    form.querySelector('input[name="action"]').value = action;
    form.submit();
}

var fileList = {};

// Dropzone.js splits the file into chunks and uploads them one at a time.
Dropzone.options.dropper = {
    paramName: "zipfile",
    chunking: true,
    forceChunking: true,
    addRemoveLinks: true,
    acceptedFiles: ".zip",
    url: "/intensity",
    maxFilesize: '10GB',
    chunkSize: 9000000000000000000, // bytes

    init: function() {

        // Append additional form data
        this.on("sending", function(file, xhr, formData) {
            formData.append('chemical', document.getElementById('chemical').value);
            formData.append('fps', document.getElementById('fps').value);
            formData.append('time1_textbox', document.getElementById('time1_textbox').value);
            formData.append('time2_textbox', document.getElementById('time2_textbox').value);
            formData.append('textbox', document.getElementById('textbox').value);
            // formData.append('zipfile', file);
        });

        // Handle success, you can update UI or display a message
        this.on("success", function (file, response) {
            console.log(response);
            file.previewElement.classList.add("dz-success");
        });

    }
};

