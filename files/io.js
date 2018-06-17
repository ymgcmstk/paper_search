$(function(){

    // --------- core ---------
    function clear_data() {
            $('#db').empty();
    }

    function get_values(keys) {
        var input_data = {keys: keys.join(",")};
        console.log(input_data);
        clear_data();
        $.ajax({
            url: './load',
            type: 'POST',
            dataType: 'json',
            data: input_data,
            timeout: 5000,
        }).done(function(data) {
            console.log(data);
            for (key in data) {
                $('#db').append('<div>key: ' + key + '</div>');
                $('#db').append('<div>value: ' + data[key] + '</div>');
            }
        }).fail(function(jqXHR, textStatus, errorThrown) {
            console.log(jqXHR);
            console.log(textStatus);
            console.log(errorThrown);
            alert('search: failure');
        });
    }
    // --------- core ---------

    // --------- bindings ---------
    $('#searchbtn').on('click', function() {
        if($('#skey').val() == null) {
            return;
        }
        get_values([$('#skey').val()]);
    });
    // --------- bindings ---------
});
