let currentChatId = null;
let chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || {};
let chatOrder = JSON.parse(localStorage.getItem("chatOrder")) || [];

// ✅ per-chat state
let chatStates = {}; // { chatId: { isAnswering: bool, abortController: AbortController|null } }
let chatTitles = JSON.parse(localStorage.getItem("chatTitles") || "{}");

// Save chat history + order to localStorage
function saveChatHistory() {
  localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
  localStorage.setItem("chatOrder", JSON.stringify(chatOrder));
  localStorage.setItem("chatTitles", JSON.stringify(chatTitles)); // ✅ new
}

function bumpChatToTop(chatId) {
  chatOrder = chatOrder.filter(id => id !== chatId);
  chatOrder.unshift(chatId);
  saveChatHistory();
}

// Create a new chat session
function createNewChat() {
  currentChatId = Date.now().toString();
  chatHistory[currentChatId] = [];
  chatStates[currentChatId] = { isAnswering: false, abortController: null };

  chatOrder = chatOrder.filter(id => id !== currentChatId);
  chatOrder.unshift(currentChatId);

  saveChatHistory();
  renderChat();
  updateChatList();

  // ✅ Reset input box when a new chat is created
  const input = document.getElementById("question");
  input.placeholder = "Ask any question...";
  input.disabled = false;
  resetCaretIfEmpty();

  refreshArrowState(currentChatId); // explicitly for the new chat
}

// Append a user/bot message
function appendMessage(chatId, sender, text) {
  if (!chatId) return;
  chatHistory[chatId].push({ sender, text });
  saveChatHistory();
  if (chatId === currentChatId) renderChat();
}

// Render current chat messages
function renderChat() {
  const chatBox = document.getElementById("chatBox");
  chatBox.innerHTML = "";

  const messages = chatHistory[currentChatId] || [];
  for (const { sender, text } of messages) {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender}`;
    bubble.textContent = text;
    chatBox.appendChild(bubble);
  }

  // Re-add bottom spacer
  const spacer = document.createElement("div");
  spacer.className = "chat-bottom-spacer";
  chatBox.appendChild(spacer);

  chatBox.scrollTop = chatBox.scrollHeight;
}

// Render sidebar chat list
// --- updateChatList ---
function updateChatList() {
  const chatList = document.getElementById("chatList");
  chatList.innerHTML = "";

  for (let i = 0; i < chatOrder.length; i++) {
    const chatId = chatOrder[i];
    const messages = chatHistory[chatId];
    if (!messages || messages.length === 0) continue;

    // ✅ ensure every chat has a state object
    if (!chatStates[chatId]) {
      chatStates[chatId] = { isAnswering: false, abortController: null };
    }

    const title = chatTitles[chatId] || "Untitled Chat";

    const wrapper = document.createElement("div");
    wrapper.className = "chat-title-wrapper";

    const button = document.createElement("button");
    button.className = "chat-button";
    button.innerText = title;
    button.setAttribute("data-id", chatId);
    
    // ✅ switch chat (fixed)
    button.onclick = () => {
      currentChatId = chatId;

      // 🧠 Ensure state object exists (don’t overwrite isAnswering)
      if (!chatStates[currentChatId]) {
        chatStates[currentChatId] = { isAnswering: false, abortController: null };
      }

      renderChat();

      // 🔧 Important: refresh the arrow for THIS chat explicitly
      refreshArrowState(chatId);
    };

    // ✅ double-click rename
    button.ondblclick = () => {
      const input = document.createElement("input");
      input.className = "chat-title-input";
      input.type = "text";
      input.value = chatTitles[chatId] || "Untitled Chat";

      wrapper.replaceChild(input, button);
      input.focus();

      const save = () => {
        const newTitle = input.value.trim();
        if (newTitle) {
          chatTitles[chatId] = newTitle;
          saveChatHistory();
          updateChatList();
        } else {
          updateChatList();
        }
      };

      input.onblur = save;
      input.onkeydown = (e) => {
        if (e.key === "Enter") save();
      };
    };

    // Three-dot menu
    const menuBtn = document.createElement("div");
    menuBtn.className = "menu-btn";
    menuBtn.innerHTML = "&#x22EE;";
    menuBtn.onclick = (e) => {
      e.stopPropagation();
      toggleMenu(chatId, wrapper);
    };

    wrapper.appendChild(button);
    wrapper.appendChild(menuBtn);
    chatList.appendChild(wrapper);
  }
}

function toggleMenu(chatId, wrapper) {
  const existing = document.querySelector(".chat-menu");
  if (existing) existing.remove();

  const menu = document.createElement("div");
  menu.className = "chat-menu";

  // Rename
  const rename = document.createElement("div");
  rename.className = "chat-menu-item";
  rename.innerText = "Rename";
  rename.onclick = () => {
    const input = document.createElement("input");
    input.className = "chat-title-input";
    input.type = "text";
    input.value = chatTitles[chatId] || "Untitled Chat"; // ✅ use chatTitles
    wrapper.replaceChild(input, wrapper.children[0]);
    input.focus();

    const save = () => {
      const newTitle = input.value.trim();
      if (newTitle) {
        chatTitles[chatId] = newTitle;        // ✅ update chatTitles, not chatHistory
        saveChatHistory();
        updateChatList();
      }
    };

    input.onblur = save;
    input.onkeydown = (e) => {
      if (e.key === "Enter") save();
    };

    menu.remove();
  };

  // Delete
  const del = document.createElement("div");
  del.className = "chat-menu-item delete";
  del.innerText = "Delete";
  del.onclick = () => {
    delete chatHistory[chatId];
    delete chatTitles[chatId];                 // ✅ also remove from chatTitles
    chatOrder = chatOrder.filter(id => id !== chatId);
    saveChatHistory();
    updateChatList();
    if (currentChatId === chatId) {
      currentChatId = null;
      renderChat();
      refreshArrowState();
    }
  };

  menu.appendChild(rename);
  menu.appendChild(del);
  wrapper.appendChild(menu);
}

// Submit user question (non-streaming JSON backend)
async function submitQuestion() {
  const input = document.getElementById("question");

  // ✅ FIX: read input value first
  const rawValue = input.value;
  const question = rawValue.trim();
  if (!question) return;

  // ✅ THEN create chat
  if (!currentChatId) createNewChat();

  const chatIdAtSubmit = currentChatId;
  appendMessage(chatIdAtSubmit, "user", question);

  bumpChatToTop(chatIdAtSubmit);
  updateChatList();  // make sidebar reflect updated order

  // ✅ set title if none exists yet
  if (!chatTitles[chatIdAtSubmit] || chatTitles[chatIdAtSubmit] === "Untitled Chat") {
    chatTitles[chatIdAtSubmit] = question.slice(0, 40);
    saveChatHistory();
    updateChatList();
  }

  input.value = "";
  toggleArrow();
  autoResize();

  // ✅ NEW: setup AbortController for cancel
  const state = chatStates[chatIdAtSubmit];
  if (state.abortController) state.abortController.abort(); // cleanup any old one
  const controller = new AbortController();
  state.abortController = controller;
  state.isAnswering = true;
  refreshArrowState(chatIdAtSubmit);

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal, // ✅ link cancel signal
    });

    if (!response.ok) throw new Error(`Server returned ${response.status}`);

    const data = await response.json();

    // ✅ Don't append answer if user cancelled
    if (!controller.signal.aborted) {
      const answer = data.answer || "⚠️ No answer returned.";
      appendMessage(chatIdAtSubmit, "bot", answer);
    }

  } catch (err) {
    if (controller.signal.aborted) {
      console.log("❌ Request aborted by user.");
    } else {
      console.error("Error fetching bot response:", err);
      const chatBox = document.getElementById("chatBox");
      const errorBubble = document.createElement("div");
      errorBubble.className = "bubble bot";
      errorBubble.textContent = "⚠️ Error getting response.";
      chatBox.appendChild(errorBubble);
    }
  } finally {
    finishAnswering(chatIdAtSubmit);
  }
}

function cancelQuestion(chatId = currentChatId) {
  const state = chatStates[chatId];
  if (state?.abortController) {
    state.abortController.abort();
  }
  finishAnswering(chatId);
}

function finishAnswering(chatId) {
  if (!chatStates[chatId]) return;
  chatStates[chatId].isAnswering = false;
  chatStates[chatId].abortController = null;

  if (chatId === currentChatId) {
    refreshArrowState(chatId);
  }
}

function refreshArrowState(forChatId = null) {
  const chatId = forChatId ?? currentChatId;
  const input = document.getElementById("question");
  const arrow = document.getElementById("sendArrow");

  // create chat state if missing
  if (!chatStates[chatId]) {
    chatStates[chatId] = { isAnswering: false, abortController: null };
  }

  const state = chatStates[chatId];
  const isAnswering = !!state.isAnswering;

  // 🧠 only update the DOM if the chat we're refreshing is *actually the one being viewed*
  if (chatId !== currentChatId) return;

  if (isAnswering) {
    input.disabled = true;
    arrow.classList.add("cancel");
    arrow.innerText = "";
    arrow.onclick = () => cancelQuestion(chatId);
  } else {
    input.disabled = false;
    arrow.classList.remove("cancel");
    arrow.innerText = "↑";
    arrow.onclick = submitQuestion;
  }
}

function handleKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
}

function toggleArrow() {
  const input = document.getElementById("question");
  const arrow = document.getElementById("sendArrow");
  arrow.style.opacity = input.value.trim() ? "1" : "0.3";
}

function autoResize() {
  const input = document.getElementById("question");
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
}

window.onload = () => {
  updateChatList();

  if (chatOrder.length > 0) {
    currentChatId = chatOrder[0];
    renderChat();
  } else {
    createNewChat();
  }

  // ✅ Ensure input is empty, placeholder shows, and caret is at far left
  const input = document.getElementById("question");
  input.value = "";                           // clear hidden whitespace
  input.placeholder = "Ask any question...";  // force placeholder on load
  input.disabled = false;
  if (input.setSelectionRange) {
    input.setSelectionRange(0, 0);            // caret flush left
  }
};

// close menus on click outside
document.addEventListener("click", (e) => {
  const openMenus = document.querySelectorAll(".chat-menu");
  openMenus.forEach(menu => {
    if (!menu.contains(e.target) && !menu.previousSibling?.contains(e.target)) {
      menu.remove();
    }
  });
});

// --- global helper so it's accessible everywhere ---
function resetCaretIfEmpty() {
  const input = document.getElementById("question");
  if (input && input.value.trim() === "" && input.setSelectionRange) {
    input.setSelectionRange(0, 0);
  }
}

// --- run once on DOM load, and rebind the focus listener ---
document.addEventListener("DOMContentLoaded", () => {
  resetCaretIfEmpty();
  const input = document.getElementById("question");
  if (input) input.addEventListener("focus", resetCaretIfEmpty);
});
