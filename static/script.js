// static/script.js

let session_id = localStorage.getItem("chat_session") || Date.now().toString();
localStorage.setItem("chat_session", session_id);

const msgBox = document.getElementById("messages");

function sendMessage() {
  const input = document.getElementById("user-input");
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";

  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id, message: text }),
  })
    .then(res => res.json())
    .then(data => {
      appendMessage("bot", data.reply);
      if (Object.keys(data.session || {}).length === 0) {
        localStorage.removeItem("chat_session"); // auto logout
      }
    });
}

function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.className = sender;
  div.innerText = text;
  msgBox.appendChild(div);
  msgBox.scrollTop = msgBox.scrollHeight;
}
