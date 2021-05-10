

$(document).ready(function () {

    var fileInput = document.getElementById('input-img');

    fileInput.addEventListener('change', function (e) {
        var image = document.getElementById('show-input-image');
	    image.src = URL.createObjectURL(e.target.files[0]);
    });

    $('#btn-upload').click(function(e){
        e.preventDefault();
        $('#input-img').click();}
    );
    
    var masks = document.getElementById('masks');

    masks.addEventListener('change', function (e) {
        var image = document.getElementById('show-masked-image');
        var optVal= e.target.value;
        if(optVal=="1"){
            $('#show-masked-image').attr('src','../static/Mask/mask1.jpg');
            // image.src = "./Mask/mask1.jpg"
        }
        else if(optVal=="2"){
            $('#show-masked-image').attr('src','../static/Mask/mask2.jpg');
            // image.src = "./Mask/mask2.jpg"
        }
        else if(optVal=="3"){
            $('#show-masked-image').attr('src','../static/Mask/mask3.jpg');
            // image.src = "./Mask/mask3.jpg"
        }
        else if(optVal=="4"){
            $('#show-masked-image').attr('src','../static/Mask/mask4.jpg');
            // image.src = "./Mask/mask4.jpg"
        } 
    });

})