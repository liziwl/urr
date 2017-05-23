$(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip();
});

$('#myDropdown').on('hide.bs.dropdown', function () {
    var selected = [];
    $('#categories_dropdown input:checked').each(function() {
        selected.push($(this).attr('name'));
    });
    if (selected.length > 0) {
        handle_filtering(selected)
    } else {
        handle_filtering(["ALL"])
    }
   return true;
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
            var newDoc = document.open("text/html", "replace");
            newDoc.write(result);
            /*newDoc.close();*/
        }
    });
};