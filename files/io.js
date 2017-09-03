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

    function update(input_data) {
        $.ajax({
            url: './save',
            type: 'POST',
            data: input_data,
            timeout: 5000
        }).done(function(data) { // console.log(data);
            var keys = [];
            for (key in input_data) {
                keys.push(key);
            }
            get_values(keys);
        }).fail(function(jqXHR, textStatus, errorThrown) {
            console.log(jqXHR);
            console.log(textStatus);
            console.log(errorThrown);
            alert('update: failure');
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

    $('#updatebtn').on('click', function() {
        if($('#ukey').val() == null) {
            return;
        }
        if($('#uvalue').val() == null) {
            return;
        }
        var data = {};
        data[$('#ukey').val()] = $('#uvalue').val();
        update(data);
    });
    // --------- bindings ---------
});
