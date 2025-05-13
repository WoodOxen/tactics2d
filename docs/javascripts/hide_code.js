document.addEventListener("DOMContentLoaded", function () {
  var cells = document.querySelectorAll('.celltag_hide_input');

  cells.forEach(function (cell) {
    var input = cell.querySelector('.jp-InputArea');

    if (input) {
      input.style.display = 'none';

      var button = document.createElement('button');
      button.innerText = 'Show Code';
      button.classList.add('show-code-button');

      button.onclick = function () {
        if (input.style.display === 'none') {
          input.style.display = '';
          button.innerText = 'Hide Code';
        } else {
          input.style.display = 'none';
          button.innerText = 'Show Code';
        }
      };

      cell.insertBefore(button, input);
    }
  });
});
