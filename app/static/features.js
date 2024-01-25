
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

// Dropzone.js splits the file into chunks and uploads them one at a time.
Dropzone.options.dropper1 = {
    paramName: "zipfile",
    chunking: true,
    forceChunking: true,
    timeout: null,
    acceptedFiles: ".zip",
    url: "/intensity",
    parallelUploads: 5,
    maxFilesize: '10GB',
    chunkSize: 90000000000000, // bytes
    dictDefaultMessage: "Drop zip files here to upload",

    init: function() {
        this.on("sending", function(file, xhr, formData) {
            // Append additional form data
            formData.append('chemical1', document.getElementById('chemical1').value);
            formData.append('fps1', document.getElementById('fps1').value);
            formData.append('time1_textbox1', document.getElementById('time1_textbox1').value);
            formData.append('time2_textbox1', document.getElementById('time2_textbox1').value);
        });
        this.on("success", function (file, response) {
            // Handle success, you can update UI or display a message
            console.log(response.message);
            file.previewElement.classList.add("dz-success");
        });
    }
};

Dropzone.options.dropper2 = {
    paramName: "zipfile",
    chunking: true,
    forceChunking: true,
    timeout: null,
    acceptedFiles: ".zip",
    url: "/rate",
    parallelUploads: 5,
    maxFilesize: '10GB',
    chunkSize: 90000000000000, // bytes
    dictDefaultMessage: "Drop zip files here to upload",
    init: function() {
        this.on("sending", function(file, xhr, formData) {
            // Append additional form data
            formData.append('chemical2', document.getElementById('chemical2').value);
            formData.append('fps2', document.getElementById('fps2').value);
            formData.append('time1_textbox2', document.getElementById('time1_textbox2').value);
            formData.append('time2_textbox2', document.getElementById('time2_textbox2').value);
            formData.append('textbox', document.getElementById('textbox').value);
        });
        this.on("success", function (file, response) {
            // Handle success, you can update UI or display a message
            console.log(response.message);
            file.previewElement.classList.add("dz-success");
        });
    }
};
