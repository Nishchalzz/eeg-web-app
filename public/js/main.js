document.addEventListener("DOMContentLoaded", function () {
  const collapsibles = document.querySelectorAll(".collapsible");

  collapsibles.forEach((item) => {
    const header = item.querySelector("h2");

    header.addEventListener("click", () => {
      item.classList.toggle("active");
    });
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatOutput = document.getElementById("chat-output");
  chatForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    const userInput = chatInput.value;
    const pageType = chatForm.getAttribute("data-page");
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userInput, pageType }),
    });
    const data = await response.json();
    chatOutput.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
    chatOutput.innerHTML += `<p><strong>AI:</strong> ${data.reply}</p>`;
    chatInput.value = "";
  });
});
