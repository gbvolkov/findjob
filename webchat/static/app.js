const state = {
  currentDialogId: null,
  socket: null,
  history: [],
  tasks: [],
  categories: [],
  user: null,
  allowedFileTypes: [],
  allowedFileExtensions: [],
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

function updateFileInputAccept() {
  const input = document.getElementById("file-input");
  if (!input) {
    return;
  }
  const acceptValues = [];
  if (Array.isArray(state.allowedFileExtensions) && state.allowedFileExtensions.length > 0) {
    acceptValues.push(...state.allowedFileExtensions);
  }
  if (Array.isArray(state.allowedFileTypes) && state.allowedFileTypes.length > 0) {
    acceptValues.push(...state.allowedFileTypes);
  }
  if (acceptValues.length > 0) {
    input.setAttribute("accept", acceptValues.join(","));
  } else {
    input.removeAttribute("accept");
  }
}

function isFileAllowed(file) {
  const allowedTypes = Array.isArray(state.allowedFileTypes) ? state.allowedFileTypes : [];
  const allowedExts = Array.isArray(state.allowedFileExtensions) ? state.allowedFileExtensions : [];
  const type = (file.type || "").toLowerCase();
  const extension = file.name && file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
  const dotExtension = extension ? `.${extension}` : "";
  const typeAllowed = allowedTypes.length === 0 || allowedTypes.includes(type);
  const extAllowed = allowedExts.length === 0 || (dotExtension && allowedExts.includes(dotExtension));
  return typeAllowed || extAllowed;
}

function allowedFilesHint() {
  const exts = Array.isArray(state.allowedFileExtensions) ? state.allowedFileExtensions : [];
  const types = Array.isArray(state.allowedFileTypes) ? state.allowedFileTypes : [];
  const parts = [];
  if (exts.length > 0) {
    parts.push(exts.join(", "));
  }
  if (types.length > 0) {
    parts.push(types.join(", "));
  }
  return parts.join(", ");
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


const DESIRED_POSITIONS_LABEL = "\u0416\u0435\u043b\u0430\u0435\u043c\u044b\u0435 \u043f\u043e\u0437\u0438\u0446\u0438\u0438:";
const LINK_LABEL = "\u0421\u0441\u044b\u043b\u043a\u0430:";

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const DESIRED_POSITIONS_ANY_RE = new RegExp(escapeRegex(DESIRED_POSITIONS_LABEL), "i");
const DESIRED_POSITIONS_PREFIX_RE = new RegExp(`${escapeRegex(DESIRED_POSITIONS_LABEL)}\\s*`, "i");
const LINK_PREFIX_RE = new RegExp(`${escapeRegex(LINK_LABEL)}\\s*`, "i");

function appendWithHighlights(parent, text) {
  if (text === null || text === undefined) {
    return;
  }
  const raw = String(text);
  const pattern = /<highlighttext>(.*?)<\/highlighttext>/gis;
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(raw)) !== null) {
    const before = raw.slice(lastIndex, match.index);
    if (before) {
      parent.append(before);
    }
    const em = document.createElement("em");
    appendWithHighlights(em, match[1]);
    parent.appendChild(em);
    lastIndex = match.index + match[0].length;
  }
  const remaining = raw.slice(lastIndex);
  if (remaining) {
    parent.append(remaining);
  }
}

function shouldFormatJobAgentContent(content) {
  if (typeof content !== "string") {
    return false;
  }
  return DESIRED_POSITIONS_ANY_RE.test(content) || /^\s*\d+\.\s+/m.test(content);
}

function buildJobSummary(lines) {
  const summary = document.createElement("div");
  summary.className = "job-summary";
  let hasContent = false;

  lines.forEach((line) => {
    const value = line.trim();
    if (!value) {
      return;
    }
    if (DESIRED_POSITIONS_PREFIX_RE.test(value)) {
      const heading = document.createElement("p");
      heading.className = "job-summary-heading";
      heading.textContent = DESIRED_POSITIONS_LABEL;
      summary.appendChild(heading);

      const allPositions = value.replace(DESIRED_POSITIONS_PREFIX_RE, "");
      const items = allPositions
        .split(/[,;]\s*/)
        .map((item) => item.trim())
        .filter(Boolean);

      if (items.length > 0) {
        const list = document.createElement("ul");
        list.className = "job-desired-positions";
        items.forEach((item) => {
          const li = document.createElement("li");
          const strong = document.createElement("strong");
          appendWithHighlights(strong, item);
          li.appendChild(strong);
          list.appendChild(li);
        });
        summary.appendChild(list);
      }
      hasContent = true;
      return;
    }
    const lineElement = document.createElement("p");
    lineElement.className = "job-summary-line";
    appendWithHighlights(lineElement, value);
    summary.appendChild(lineElement);
    hasContent = true;
  });

  return hasContent ? summary : null;
}

function buildJobVacancy(block) {
  const normalized = block.replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  const vacancy = document.createElement("article");
  vacancy.className = "job-vacancy";
  let hasContent = false;

  const headerLine = (lines.shift() || "").trim();
  if (headerLine) {
    const match = headerLine.match(/^(\d+)\.\s*(.+)$/);
    const header = document.createElement("div");
    header.className = "job-vacancy-header";
    if (match) {
      const index = document.createElement("span");
      index.className = "job-vacancy-index";
      index.textContent = `${match[1]}.`;
      header.appendChild(index);

      const title = document.createElement("strong");
      title.className = "job-vacancy-title";
      appendWithHighlights(title, match[2].trim());
      header.appendChild(title);
    } else {
      const title = document.createElement("strong");
      title.className = "job-vacancy-title";
      appendWithHighlights(title, headerLine);
      header.appendChild(title);
    }
    vacancy.appendChild(header);
    hasContent = true;
  }

  lines.forEach((line) => {
    const value = line.trim();
    if (!value) {
      return;
    }
    if (LINK_PREFIX_RE.test(value)) {
      const url = value.replace(LINK_PREFIX_RE, "").trim();
      if (url) {
        const linkContainer = document.createElement("p");
        linkContainer.className = "job-vacancy-link";
        linkContainer.append(`${LINK_LABEL} `);

        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.textContent = url;
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";

        linkContainer.appendChild(anchor);
        vacancy.appendChild(linkContainer);
        hasContent = true;
        return;
      }
    }
    const paragraph = document.createElement("p");
    paragraph.className = "job-vacancy-line";
    appendWithHighlights(paragraph, value);
    vacancy.appendChild(paragraph);
    hasContent = true;
  });

  return hasContent ? vacancy : null;
}

function buildJobAgentFragment(content) {
  const normalized = content.replace(/\r\n/g, "\n");
  const blocks = normalized.split(/\n{2,}/);
  const summaryLines = [];
  const vacancyBlocks = [];

  blocks.forEach((block) => {
    const trimmed = block.trim();
    if (!trimmed) {
      return;
    }
    if (/^\d+\.\s/.test(trimmed)) {
      vacancyBlocks.push(trimmed);
    } else {
      summaryLines.push(...trimmed.split("\n"));
    }
  });

  const fragment = document.createDocumentFragment();
  const summary = buildJobSummary(summaryLines);
  if (summary) {
    fragment.appendChild(summary);
  }

  if (vacancyBlocks.length > 0) {
    const vacanciesContainer = document.createElement("div");
    vacanciesContainer.className = "job-vacancies";
    vacancyBlocks.forEach((block) => {
      const vacancy = buildJobVacancy(block);
      if (vacancy) {
        vacanciesContainer.appendChild(vacancy);
      }
    });
    if (vacanciesContainer.childElementCount > 0) {
      fragment.appendChild(vacanciesContainer);
    }
  }

  return fragment.childNodes.length > 0 ? fragment : null;
}

function renderMessageContent(container, content) {
  container.classList.remove("job-agent");
  container.textContent = "";
  if (content === null || content === undefined) {
    return;
  }
  const raw = typeof content === "string" ? content : String(content);
  const normalized = raw.replace(/\r\n/g, "\n");
  if (shouldFormatJobAgentContent(normalized)) {
    const fragment = buildJobAgentFragment(normalized);
    if (fragment) {
      container.classList.add("job-agent");
      container.appendChild(fragment);
      return;
    }
  }
  appendWithHighlights(container, raw);
}
function renderMessage(message) {
  const template = document.getElementById("message-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.toggle("user", message.sender === "user");
  node.querySelector(".sender").textContent = message.sender === "user" ? "Вы" : "Бот";
  node.querySelector(".timestamp").textContent = formatDate(message.created_at);
  const contentNode = node.querySelector(".message-content");

  renderMessageContent(contentNode, message.content);
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

async function loadSession() {
  const data = await fetchJSON("/api/session");
  state.user = data.user || null;
  state.allowedFileTypes = data.allowed_file_types || [];
  state.allowedFileExtensions = data.allowed_file_extensions || [];
  updateFileInputAccept();
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
  const textarea = document.getElementById("message-input");
  if (textarea) {
    textarea.value = promptText;
    textarea.focus();
  }
}

async function uploadAttachment(dialogId, fileInput) {
  if (!fileInput.files || fileInput.files.length === 0) {
    return null;
  }
  const file = fileInput.files[0];
  if (!isFileAllowed(file)) {
    const hint = allowedFilesHint();
    throw new Error(hint ? `Unsupported file type. Allowed: ${hint}` : "Unsupported file type.");
  }
  const formData = new FormData();
  formData.append("uploaded_file", file);
  const response = await fetch(`/api/dialogs/${dialogId}/files`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Failed to upload the selected file.");
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
  await loadSession();
  await Promise.all([loadHistory(), loadTasks(), loadCategories()]);
}

bootstrap().catch((error) => {
  console.error(error);
  alert("Не удалось загрузить данные приложения");
});
