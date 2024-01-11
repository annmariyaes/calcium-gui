
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
});
