// comment

/*
Longer
comments
*/

function openTabs(opName) {
    var i;
    var x = document.getElementsByClassName("option");
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";
    }
    document.getElementById(opName).style.display = "block";
}