/**
 * Created by adelineciurumelea on 15/05/17.
 */

console.log(">>>>>>>>>>>>>> Loaded...")

function disableButtonState(elem) {
    if(confirm('Are you sure you want to disable this button?') == true) {
        elem.disabled = true;
        alert("its done.");
    }
    else {
        return false;
    }
}