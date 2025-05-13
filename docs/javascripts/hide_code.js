window.addEventListener("load", function () {
  const cells = document.querySelectorAll(".celltag_hide_input");

  cells.forEach((cell) => {
    const inputWrapper = cell.querySelector(".jp-Cell-inputWrapper");
    const inputArea = inputWrapper?.querySelector(".jp-InputArea.jp-Cell-inputArea");

    if (inputWrapper && inputArea) {
      inputArea.style.display = "none";

      const button = document.createElement("button");
      button.innerText = "Show Code";
      button.className = "show-code-button";

      button.onclick = () => {
        const isHidden = inputArea.style.display === "none";
        inputArea.style.display = isHidden ? "" : "none";
        button.innerText = isHidden ? "Hide Code" : "Show Code";
      };

      inputWrapper.insertBefore(button, inputArea);
    }
  });
});
