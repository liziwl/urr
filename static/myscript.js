$(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip();
});

var options = [];

$( '.dropdown-menu a' ).on( 'click', function( event ) {
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

   handle_filtering(options)
   return false;
});

var handle_filtering = function(filtering_categories) {
    data = {
        "filtering_categories": filtering_categories
    }
    $.ajax({
        type : "POST",
        url : "/reviews",
        data: JSON.stringify(data, null, '\t'),
        contentType: 'application/json;charset=UTF-8',
        success: function(result) {
            // Not showing anything
        }
    });
};