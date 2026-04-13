function insertHideButtons() {
  const cells = document.querySelectorAll(".celltag_hide_input[id]");

  cells.forEach((cell) => {
    if (cell.classList.contains("button-inserted")) return;

    const inputWrapper = cell.querySelector(".jp-Cell-inputWrapper");
    const inputArea = inputWrapper?.querySelector(".jp-InputArea.jp-Cell-inputArea");

    if (inputWrapper && inputArea) {
      inputArea.style.display = "none";

      const buttonWrapper = document.createElement("div");
      buttonWrapper.style.width = "100%";
      buttonWrapper.style.textAlign = "center";
      buttonWrapper.style.margin = "4px 0";

      const button = document.createElement("button");
      button.innerText = "  ▶  Show Code";
      button.className = "show-code-button";

      button.onclick = () => {
        const isHidden = inputArea.style.display === "none";
        inputArea.style.display = isHidden ? "" : "none";
        button.innerText = isHidden ? "  ▼  Hide Code" : "  ▶  Show Code";
      };

      buttonWrapper.appendChild(button);
      inputWrapper.parentNode.insertBefore(buttonWrapper, inputWrapper);

      cell.classList.add("button-inserted");
    }
  });
}

window.addEventListener("DOMContentLoaded", insertHideButtons);

const observer = new MutationObserver((mutationsList, observer) => {
  insertHideButtons();
});
observer.observe(document.body, { childList: true, subtree: true });
