const state = {
  currentDialogId: null,
  socket: null,
  history: [],
  tasks: [],
  categories: [],
};

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Ошибка запроса: ${response.status}`);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function formatDate(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  return date.toLocaleString();
}

function clearChatWindow() {
  const container = document.getElementById("chat-window");
  container.innerHTML = "";
}

function renderMessage(message) {
  const template = document.getElementById("message-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.toggle("user", message.sender === "user");
  node.querySelector(".sender").textContent = message.sender === "user" ? "Вы" : "Бот";
  node.querySelector(".timestamp").textContent = formatDate(message.created_at);
  node.querySelector(".message-content").textContent = message.content;
  const attachments = node.querySelector(".attachments");
  attachments.innerHTML = "";
  (message.attachments || []).forEach((fileId) => {
    const link = document.createElement("a");
    link.href = `/api/dialogs/${state.currentDialogId}/files/${fileId}`;
    link.textContent = "📎 вложение";
    link.target = "_blank";
    attachments.appendChild(link);
  });
  document.getElementById("chat-window").appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function renderMessages(messages) {
  clearChatWindow();
  messages.forEach(renderMessage);
}

function renderHistory(dialogs) {
  const container = document.getElementById("history-list");
  container.innerHTML = "";
  const template = document.getElementById("history-card-template");
  dialogs.forEach((dialog) => {
    const card = template.content.firstElementChild.cloneNode(true);
    card.dataset.id = dialog.id;
    card.querySelector("h3").textContent = dialog.title;
    card.querySelector(".history-preview").textContent = dialog.preview || "Нет сообщений";
    card.querySelector(".history-bot").textContent = dialog.bot_type;
    card.querySelector(".history-updated").textContent = formatDate(dialog.updated_at);
    card.querySelector(".history-open").addEventListener("click", () => openDialog(dialog.id));
    card.querySelector(".history-export").addEventListener("click", () => exportDialog(dialog.id));
    card.querySelector(".history-delete").addEventListener("click", () => deleteDialog(dialog.id));
    container.appendChild(card);
  });
}

function renderTasks(tasks) {
  const container = document.getElementById("tasks-list");
  container.innerHTML = "";
  const template = document.getElementById("task-card-template");
  tasks.forEach((task) => {
    const card = template.content.firstElementChild.cloneNode(true);
    card.dataset.id = task.id;
    card.querySelector("h3").textContent = task.title;
    card.querySelector(".task-category").textContent = `Категория: ${task.category}`;
    card.querySelector(".task-description").textContent = task.description;
    card.addEventListener("click", () => startTaskDialog(task));
    container.appendChild(card);
  });
}

function renderCategories(categories) {
  const select = document.getElementById("tasks-category");
  select.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = "Все категории";
  select.appendChild(allOption);
  categories.forEach((category) => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    select.appendChild(option);
  });
}

async function loadHistory() {
  const data = await fetchJSON("/api/dialogs");
  state.history = data.dialogs;
  renderHistory(state.history);
}

async function loadTasks() {
  const data = await fetchJSON("/api/tasks");
  state.tasks = data.tasks;
  renderTasks(state.tasks);
}

async function loadCategories() {
  const data = await fetchJSON("/api/tasks/categories");
  state.categories = data.categories;
  renderCategories(state.categories);
}

function updateDialogHeader(dialog) {
  const titleEl = document.getElementById("dialog-title");
  const metaEl = document.getElementById("dialog-meta");
  titleEl.textContent = dialog.title;
  metaEl.textContent = `Тип: ${dialog.bot_type} • Обновлен: ${formatDate(dialog.updated_at)}`;
  document.getElementById("export-dialog").disabled = false;
  document.getElementById("delete-dialog").disabled = false;
}

function disconnectSocket() {
  if (state.socket) {
    state.socket.close(1000, "switch dialog");
    state.socket = null;
  }
}

function connectSocket(dialogId) {
  disconnectSocket();
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/dialogs/${dialogId}`);
  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.event === "ack" || payload.event === "bot_message") {
      renderMessage(payload.message);
    }
  });
  socket.addEventListener("close", () => {
    if (state.socket === socket) {
      state.socket = null;
    }
  });
  state.socket = socket;
}

async function openDialog(dialogId) {
  const data = await fetchJSON(`/api/dialogs/${dialogId}`);
  state.currentDialogId = dialogId;
  updateDialogHeader(data);
  renderMessages(data.messages);
  connectSocket(dialogId);
}

async function exportDialog(dialogId) {
  const response = await fetch(`/api/dialogs/${dialogId}/export`);
  if (!response.ok) {
    alert("Не удалось экспортировать диалог");
    return;
  }
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `dialog-${dialogId}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

async function deleteDialog(dialogId) {
  if (!confirm("Удалить диалог?")) {
    return;
  }
  await fetchJSON(`/api/dialogs/${dialogId}`, { method: "DELETE" });
  if (state.currentDialogId === dialogId) {
    state.currentDialogId = null;
    disconnectSocket();
    document.getElementById("dialog-title").textContent = "Выберите диалог";
    document.getElementById("dialog-meta").textContent = "";
    document.getElementById("export-dialog").disabled = true;
    document.getElementById("delete-dialog").disabled = true;
    clearChatWindow();
  }
  await loadHistory();
}

async function createDialog() {
  const title = prompt("Название диалога", "Новый диалог");
  if (title === null) {
    return;
  }
  const botType = prompt("Тип бота", "assistant") || "assistant";
  const dialog = await fetchJSON("/api/dialogs", {
    method: "POST",
    body: JSON.stringify({ title, bot_type: botType }),
  });
  await loadHistory();
  await openDialog(dialog.id);
}

async function startTaskDialog(task) {
  const dialog = await fetchJSON("/api/dialogs", {
    method: "POST",
    body: JSON.stringify({ title: task.title, bot_type: task.category }),
  });
  await loadHistory();
  await openDialog(dialog.id);
  const promptText = `Задача: ${task.title}\n${task.description}`;
  await sendMessage(promptText);
}

async function uploadAttachment(dialogId, fileInput) {
  if (!fileInput.files || fileInput.files.length === 0) {
    return null;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("uploaded_file", file);
  const response = await fetch(`/api/dialogs/${dialogId}/files`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error("Не удалось загрузить файл");
  }
  const data = await response.json();
  fileInput.value = "";
  return data.id;
}

async function sendMessage(content) {
  if (!state.currentDialogId) {
    alert("Выберите диалог");
    return;
  }
  const fileInput = document.getElementById("file-input");
  let attachmentId = null;
  if (fileInput.files && fileInput.files.length > 0) {
    attachmentId = await uploadAttachment(state.currentDialogId, fileInput);
  }
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(
      JSON.stringify({
        content,
        attachments: attachmentId ? [attachmentId] : [],
      }),
    );
  } else {
    const payload = {
      content,
      attachments: attachmentId ? [attachmentId] : [],
    };
    await fetchJSON(`/api/dialogs/${state.currentDialogId}/messages`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const simulated = await fetchJSON(`/api/dialogs/${state.currentDialogId}/messages`);
    renderMessages(simulated.messages);
  }
}

function setupEventListeners() {
  document.getElementById("new-dialog").addEventListener("click", createDialog);
  document.getElementById("export-dialog").addEventListener("click", () => {
    if (state.currentDialogId) {
      exportDialog(state.currentDialogId);
    }
  });
  document.getElementById("delete-dialog").addEventListener("click", () => {
    if (state.currentDialogId) {
      deleteDialog(state.currentDialogId);
    }
  });
  document.getElementById("history-search-btn").addEventListener("click", async () => {
    const query = document.getElementById("history-search").value.trim();
    if (!query) {
      renderHistory(state.history);
      return;
    }
    const data = await fetchJSON(`/api/dialogs/search?q=${encodeURIComponent(query)}`);
    renderHistory(data.dialogs);
  });
  document.getElementById("tasks-search-btn").addEventListener("click", async () => {
    const query = document.getElementById("tasks-search").value.trim();
    const category = document.getElementById("tasks-category").value || undefined;
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    if (category) params.set("category", category);
    const url = params.toString() ? `/api/tasks?${params.toString()}` : "/api/tasks";
    const data = await fetchJSON(url);
    renderTasks(data.tasks);
  });
  document.getElementById("tasks-category").addEventListener("change", async (event) => {
    const category = event.target.value;
    const query = document.getElementById("tasks-search").value.trim();
    const params = new URLSearchParams();
    if (category) params.set("category", category);
    if (query) params.set("q", query);
    const url = params.toString() ? `/api/tasks?${params.toString()}` : "/api/tasks";
    const data = await fetchJSON(url);
    renderTasks(data.tasks);
  });
  document.getElementById("message-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const textarea = document.getElementById("message-input");
    const content = textarea.value.trim();
    if (!content) {
      return;
    }
    textarea.value = "";
    try {
      await sendMessage(content);
    } catch (error) {
      alert(error.message);
    }
  });
}

async function bootstrap() {
  setupEventListeners();
  await Promise.all([loadHistory(), loadTasks(), loadCategories()]);
}

bootstrap().catch((error) => {
  console.error(error);
  alert("Не удалось загрузить данные приложения");
});
