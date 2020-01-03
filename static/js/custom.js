$(document).ready(function () {
    $(document).on('change', '#image_input :file', function () {
        var input = $(this),
            label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
        input.trigger('fileselect', [label]);
    });

    $('#image_input :file').on('fileselect', function (event, label) {

        var input = $(this).parents('.input-group').find(':text'),
            log = label;

        if (input.length) {
            input.val(log);
        } else {
            if (log) alert(log);
        }

    });
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $('#show_input').attr('src', e.target.result);
                $('#show_input').attr('width', 400);
                $('#show_input').attr('height', 400);
            }

            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#image_input").change(function () {
        readURL(this);
        $("#btn-sub").show("slow");
        $("#tab").show("slow");
    });
});