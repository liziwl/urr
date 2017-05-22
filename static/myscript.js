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

$(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip();
});

var options = [];

$( '.dropdown-menu a' ).on( 'click', function( event ) {
    console.log(">>>>>>>>>>> ON CLICK!!!")
   var $target = $( event.currentTarget ),
       val = $target.attr( 'data-value' ),
       $inp = $target.find( 'input' ),
       idx;

   if ( ( idx = options.indexOf( val ) ) > -1 ) {
      options.splice( idx, 1 );
      setTimeout( function() { $inp.prop( 'checked', false ) }, 0);
   } else {
      options.push( val );
      setTimeout( function() { $inp.prop( 'checked', true ) }, 0);
   }

   $( event.target ).blur();

   console.log( options );
   return false;
});