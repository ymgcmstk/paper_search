$(function(){

    // --------- core ---------
    function clear_data() {
            $('#db').empty();
    }

    function search(key) {
        clear_data();
        $.ajax({
            url: './load',
            type: 'POST',
            dataType: 'json',
            data: {key},
            timeout: 5000,
        }).done(function(data) {
            console.log(data);
            $('#db').append('<div>key: ' + key + '</div>');
            $('#db').append('<div>value: ' + data[key] + '</div>');
        }).fail(function(jqXHR, textStatus, errorThrown) {
            console.log(jqXHR);
            console.log(textStatus);
            console.log(errorThrown);
            alert('search: failure');
        });
    }

    function update(key, value) {
        var data = {};
        data[key] = value;
        $.ajax({
            url: './save',
            type: 'POST',
            data: data,
            timeout: 5000
        }).done(function(data) {
            console.log(data);
            search(key);
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
        search($('#skey').val());
    });

    $('#updatebtn').on('click', function() {
        if($('#ukey').val() == null) {
            return;
        }
        if($('#uvalue').val() == null) {
            return;
        }
        update($('#ukey').val(), $('#uvalue').val());
    });
    // --------- bindings ---------
});
