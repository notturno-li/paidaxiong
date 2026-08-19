"use strict";

const palette = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#6d4c41", "#3949ab", "#c0a000", "#546e7a"];
const state = {
  data: null,
  current: null,
  annotation: null,
  boxes: [],
  selectedBox: -1,
  activeClass: 0,
  image: new Image(),
  history: [],
  gesture: null,
  clientId: "",
  pollTimer: null,
  trainingTimer: null,
};

const el = (id) => document.getElementById(id);
const canvas = el("annotationCanvas");
const ctx = canvas.getContext("2d");

function newClientId() {
  const stored = localStorage.getItem("datasetStudioClient");
  if (stored) return stored;
  const value = `标注员-${Math.random().toString(36).slice(2, 7)}`;
  localStorage.setItem("datasetStudioClient", value);
  return value;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`服务器返回异常 HTTP ${response.status}`);
  }
  if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload.data;
}

function post(path, body) {
  return api(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
}

function notify(message, error = false) {
  const toast = el("toast");
  toast.textContent = message;
  toast.className = `toast show${error ? " error" : ""}`;
  window.clearTimeout(notify.timer);
  notify.timer = window.setTimeout(() => { toast.className = "toast"; }, 2800);
}

function setOnline(online, text) {
  el("connectionDot").classList.toggle("online", online);
  el("connectionText").textContent = text;
}

async function refreshState(silent = false) {
  try {
    state.data = await api("/api/state");
    setOnline(true, "局域网服务正常");
    renderState();
  } catch (error) {
    setOnline(false, "连接中断");
    if (!silent) notify(error.message, true);
  }
}

function renderState() {
  const data = state.data;
  const project = data.project;
  el("projectTitle").textContent = `${project.name} / ${project.project_id}`;
  if (document.activeElement !== el("projectName")) el("projectName").value = project.name;
  if (document.activeElement !== el("classNames")) el("classNames").value = project.classes.join(", ");
  el("cameraMode").textContent = data.camera.mode;
  el("totalCount").textContent = data.summary.total;
  el("doneCount").textContent = data.summary.annotated;
  el("pendingCount").textContent = data.summary.pending;
  renderQueue();
  renderClasses();
  renderServerUrls();
  renderTrainingInputs();
  renderTrainingSummary();
  renderTrainingStatus(data.training);
}

function renderQueue() {
  const queue = el("imageQueue");
  queue.replaceChildren();
  for (const image of state.data.images) {
    const button = document.createElement("button");
    button.className = `queue-item${state.current?.image_id === image.image_id ? " active" : ""}${image.locked_by && image.locked_by !== state.clientId ? " locked" : ""}`;
    button.disabled = Boolean(image.locked_by && image.locked_by !== state.clientId);
    const dot = document.createElement("span");
    dot.className = `queue-state${image.annotated ? " done" : ""}`;
    const name = document.createElement("span");
    name.className = "queue-name";
    name.textContent = image.file_name;
    const meta = document.createElement("span");
    meta.className = "queue-meta";
    meta.textContent = image.locked_by && image.locked_by !== state.clientId ? "占用" : `${image.box_count} 框`;
    button.append(dot, name, meta);
    button.addEventListener("click", () => openImage(image));
    queue.appendChild(button);
  }
}

function renderClasses() {
  const classes = state.data.project.classes;
  const select = el("activeClass");
  const oldValue = Number(select.value || state.activeClass || 0);
  select.replaceChildren();
  classes.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index}: ${name}`;
    select.appendChild(option);
  });
  state.activeClass = Math.min(oldValue, classes.length - 1);
  select.value = String(state.activeClass);
  const container = el("classPalette");
  container.replaceChildren();
  classes.forEach((name, index) => {
    const button = document.createElement("button");
    button.className = `class-chip${index === state.activeClass ? " active" : ""}`;
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = palette[index % palette.length];
    const label = document.createElement("span");
    label.textContent = `${index} ${name}`;
    button.append(swatch, label);
    button.addEventListener("click", () => setActiveClass(index));
    container.appendChild(button);
  });
}

function setActiveClass(index) {
  state.activeClass = Number(index);
  el("activeClass").value = String(state.activeClass);
  renderClasses();
  if (state.selectedBox >= 0) {
    pushHistory();
    state.boxes[state.selectedBox].class_id = state.activeClass;
    redraw();
    renderBoxList();
  }
}

function renderServerUrls() {
  const container = el("serverUrls");
  container.replaceChildren();
  state.data.server_urls.forEach((url) => {
    const row = document.createElement("div");
    row.textContent = url;
    container.appendChild(row);
  });
}

function renderTrainingInputs() {
  const select = el("baseModel");
  const current = select.value;
  select.replaceChildren();
  state.data.base_models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = model.name;
    select.appendChild(option);
  });
  if (current && state.data.base_models.some((item) => item.path === current)) select.value = current;
  else {
    const preferred = state.data.base_models.find((item) => item.name === "yolov8s.pt") || state.data.base_models[0];
    if (preferred) select.value = preferred.path;
  }
}

function renderTrainingSummary() {
  const summary = state.data.summary;
  const metrics = [
    [summary.total, "图片"], [summary.annotated, "已完成"], [summary.boxes, "目标框"],
    [summary.negative, "负样本"], [summary.pending, "待标注"], [state.data.project.classes.length, "类别"],
  ];
  const container = el("trainingSummary");
  container.replaceChildren();
  metrics.forEach(([value, label]) => {
    const item = document.createElement("div");
    item.className = "metric";
    const strong = document.createElement("strong");
    strong.textContent = value;
    const span = document.createElement("span");
    span.textContent = label;
    item.append(strong, span);
    container.appendChild(item);
  });
  const counts = el("classCounts");
  counts.replaceChildren();
  Object.entries(summary.per_class).forEach(([name, count]) => {
    const row = document.createElement("div");
    row.className = "class-count-row";
    const left = document.createElement("span"); left.textContent = name;
    const right = document.createElement("strong"); right.textContent = count;
    row.append(left, right); counts.appendChild(row);
  });
}

function renderTrainingStatus(training) {
  const stateName = training.state || "idle";
  el("trainingState").textContent = stateName.toUpperCase();
  el("trainingState").className = `state-badge ${stateName}`;
  el("trainingMessage").textContent = training.message || "-";
  el("trainingLog").textContent = training.log_tail || "等待训练日志";
  el("trainingLog").scrollTop = el("trainingLog").scrollHeight;
  const running = ["starting", "running"].includes(stateName);
  el("startTrainingButton").disabled = running;
  el("cancelTrainingButton").disabled = !running;
  const output = el("modelOutput");
  if (training.model_profile) {
    output.textContent = `模型配置：${training.model_profile}\n权重：${training.weights}`;
  } else {
    output.textContent = "训练完成后，模型包会出现在 models/field_models，比赛 GUI 可直接选择。";
  }
}

async function openImage(image) {
  try {
    if (state.current && state.current.image_id !== image.image_id) await releaseCurrent();
    await post("/api/lock", {image_id: image.image_id, client_id: state.clientId});
    const annotation = await api(`/api/annotation/${encodeURIComponent(image.image_id)}`);
    state.current = image;
    state.annotation = annotation;
    state.boxes = annotation.boxes.map((box) => ({...box}));
    state.selectedBox = -1;
    state.history = [];
    el("negativeCheck").checked = annotation.completed && annotation.boxes.length === 0;
    state.image.onload = () => {
      canvas.width = state.image.naturalWidth;
      canvas.height = state.image.naturalHeight;
      el("canvasShell").classList.remove("empty");
      redraw();
    };
    state.image.src = `${image.image_url}?v=${Date.now()}`;
    el("currentImageName").textContent = image.file_name;
    setEditorEnabled(true);
    renderBoxList();
    await refreshState(true);
  } catch (error) {
    notify(error.message, true);
    await refreshState(true);
  }
}

async function releaseCurrent() {
  if (!state.current) return;
  try {
    await post("/api/unlock", {image_id: state.current.image_id, client_id: state.clientId});
  } catch (_error) {}
}

async function nextPending() {
  try {
    await releaseCurrent();
    const image = await post("/api/next", {client_id: state.clientId, only_pending: true});
    if (!image) {
      notify("没有可领取的待标注图片");
      return;
    }
    await openImage(image);
  } catch (error) { notify(error.message, true); }
}

function setEditorEnabled(enabled) {
  ["saveButton", "saveNextButton", "deleteImageButton"].forEach((id) => { el(id).disabled = !enabled; });
  updateBoxButtons();
}

function updateBoxButtons() {
  const selected = state.selectedBox >= 0;
  el("deleteBoxButton").disabled = !selected;
  el("undoButton").disabled = state.history.length === 0;
}

function pushHistory() {
  state.history.push(state.boxes.map((box) => ({...box})));
  if (state.history.length > 30) state.history.shift();
  updateBoxButtons();
}

function undo() {
  if (!state.history.length) return;
  state.boxes = state.history.pop();
  state.selectedBox = -1;
  redraw(); renderBoxList(); updateBoxButtons();
}

function deleteSelectedBox() {
  if (state.selectedBox < 0) return;
  pushHistory();
  state.boxes.splice(state.selectedBox, 1);
  state.selectedBox = -1;
  redraw(); renderBoxList(); updateBoxButtons();
}

function renderBoxList() {
  const container = el("boxList");
  container.replaceChildren();
  el("boxCount").textContent = String(state.boxes.length);
  const classes = state.data?.project.classes || [];
  state.boxes.forEach((box, index) => {
    const button = document.createElement("button");
    button.className = `box-item${index === state.selectedBox ? " selected" : ""}`;
    const swatch = document.createElement("span"); swatch.className = "swatch"; swatch.style.background = palette[box.class_id % palette.length];
    const name = document.createElement("span"); name.textContent = classes[box.class_id] || `class ${box.class_id}`;
    const coords = document.createElement("span"); coords.className = "box-coords"; coords.textContent = `${Math.round(box.width * 100)}% x ${Math.round(box.height * 100)}%`;
    button.append(swatch, name, coords);
    button.addEventListener("click", () => { state.selectedBox = index; setActiveClass(box.class_id); redraw(); renderBoxList(); updateBoxButtons(); });
    container.appendChild(button);
  });
  updateBoxButtons();
}

function redraw() {
  if (!state.current || !state.image.complete) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, canvas.width / 320);
  ctx.font = `${Math.max(13, canvas.width / 45)}px Microsoft YaHei`;
  state.boxes.forEach((box, index) => drawBox(box, index === state.selectedBox));
  if (state.gesture?.type === "draw") drawBox(state.gesture.box, true, true);
}

function drawBox(box, selected, provisional = false) {
  const x = box.x * canvas.width;
  const y = box.y * canvas.height;
  const width = box.width * canvas.width;
  const height = box.height * canvas.height;
  const color = provisional ? "#ffffff" : palette[box.class_id % palette.length];
  ctx.strokeStyle = color;
  ctx.setLineDash(provisional ? [8, 5] : []);
  ctx.strokeRect(x, y, width, height);
  ctx.setLineDash([]);
  if (selected && !provisional) {
    const size = Math.max(8, canvas.width / 70);
    ctx.fillStyle = "#fff";
    ctx.fillRect(x + width - size / 2, y + height - size / 2, size, size);
    ctx.strokeStyle = color;
    ctx.strokeRect(x + width - size / 2, y + height - size / 2, size, size);
  }
  const name = (state.data?.project.classes || [])[box.class_id] || `class ${box.class_id}`;
  const textWidth = ctx.measureText(name).width + 10;
  const textHeight = Math.max(18, canvas.width / 35);
  const fontSize = Math.max(13, canvas.width / 45);
  ctx.fillStyle = color;
  ctx.fillRect(x, Math.max(0, y - textHeight), textWidth, textHeight);
  ctx.fillStyle = "#fff";
  ctx.fillText(name, x + 5, Math.max(fontSize, y - 4));
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
}

function hitBox(point) {
  let selected = -1;
  let selectedArea = Infinity;
  state.boxes.forEach((box, index) => {
    if (point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y && point.y <= box.y + box.height) {
      const area = box.width * box.height;
      if (area < selectedArea) { selected = index; selectedArea = area; }
    }
  });
  return selected;
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.current || el("negativeCheck").checked) return;
  canvas.setPointerCapture(event.pointerId);
  const point = pointerPosition(event);
  const hit = hitBox(point);
  if (hit >= 0) {
    state.selectedBox = hit;
    const box = state.boxes[hit];
    const nearHandle = Math.abs(point.x - (box.x + box.width)) < 0.025 && Math.abs(point.y - (box.y + box.height)) < 0.025;
    pushHistory();
    state.gesture = {type: nearHandle ? "resize" : "move", start: point, original: {...box}, index: hit};
    setActiveClass(box.class_id);
  } else {
    pushHistory();
    state.selectedBox = -1;
    state.gesture = {type: "draw", start: point, box: {class_id: state.activeClass, x: point.x, y: point.y, width: 0, height: 0}};
  }
  redraw(); renderBoxList();
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.gesture) return;
  const point = pointerPosition(event);
  const gesture = state.gesture;
  if (gesture.type === "draw") {
    gesture.box.x = Math.min(gesture.start.x, point.x);
    gesture.box.y = Math.min(gesture.start.y, point.y);
    gesture.box.width = Math.abs(point.x - gesture.start.x);
    gesture.box.height = Math.abs(point.y - gesture.start.y);
  } else if (gesture.type === "move") {
    const dx = point.x - gesture.start.x;
    const dy = point.y - gesture.start.y;
    const box = state.boxes[gesture.index];
    box.x = Math.max(0, Math.min(1 - box.width, gesture.original.x + dx));
    box.y = Math.max(0, Math.min(1 - box.height, gesture.original.y + dy));
  } else if (gesture.type === "resize") {
    const box = state.boxes[gesture.index];
    box.width = Math.max(0.005, Math.min(1 - box.x, point.x - box.x));
    box.height = Math.max(0.005, Math.min(1 - box.y, point.y - box.y));
  }
  redraw();
});

canvas.addEventListener("pointerup", (event) => {
  if (!state.gesture) return;
  const gesture = state.gesture;
  if (gesture.type === "draw" && gesture.box.width > 0.005 && gesture.box.height > 0.005) {
    state.boxes.push({...gesture.box});
    state.selectedBox = state.boxes.length - 1;
  } else if (gesture.type === "draw") {
    state.history.pop();
  }
  state.gesture = null;
  redraw(); renderBoxList(); updateBoxButtons();
  canvas.releasePointerCapture(event.pointerId);
});

async function saveAnnotation(goNext) {
  if (!state.current || !state.annotation) return;
  const negative = el("negativeCheck").checked;
  if (!negative && state.boxes.length === 0) {
    notify("没有标注框；若图片确实无目标，请勾选“无目标负样本”", true);
    return;
  }
  try {
    const saved = await post(`/api/annotation/${encodeURIComponent(state.current.image_id)}`, {
      client_id: state.clientId,
      version: state.annotation.version,
      completed: true,
      boxes: negative ? [] : state.boxes,
    });
    state.annotation = saved;
    notify(`已保存 ${saved.boxes.length} 个标注框`);
    await refreshState(true);
    if (goNext) await nextPending();
  } catch (error) { notify(error.message, true); }
}

async function captureImage() {
  try {
    el("captureButton").disabled = true;
    const record = await post("/api/capture", {});
    notify(`已采集 ${record.file_name}`);
    await refreshState(true);
    const image = state.data.images.find((item) => item.image_id === record.image_id);
    if (image) await openImage(image);
  } catch (error) { notify(error.message, true); }
  finally { el("captureButton").disabled = false; }
}

async function uploadImages(files) {
  if (!files.length) return;
  const form = new FormData();
  Array.from(files).forEach((file) => form.append("images", file));
  try {
    const records = await api("/api/upload", {method: "POST", body: form});
    notify(`已导入 ${records.length} 张图片`);
    await refreshState(true);
  } catch (error) { notify(error.message, true); }
  el("uploadInput").value = "";
}

async function saveProject() {
  const classes = el("classNames").value.split(/[,，;；]+/).map((value) => value.trim()).filter(Boolean);
  try {
    await post("/api/project", {name: el("projectName").value.trim(), classes});
    notify("项目设置已保存");
    await refreshState(true);
  } catch (error) { notify(error.message, true); }
}

async function deleteImage() {
  if (!state.current) return;
  if (!window.confirm(`确认删除图片 ${state.current.file_name} 及其标注？`)) return;
  try {
    await post("/api/delete", {image_id: state.current.image_id, client_id: state.clientId, confirm: true});
    state.current = null; state.annotation = null; state.boxes = [];
    el("canvasShell").classList.add("empty"); el("currentImageName").textContent = "未选择图片";
    setEditorEnabled(false); renderBoxList();
    await refreshState(true); notify("图片已删除");
  } catch (error) { notify(error.message, true); }
}

async function startTraining() {
  const settings = {
    base_model: el("baseModel").value,
    epochs: Number(el("epochs").value),
    imgsz: Number(el("imgsz").value),
    batch: Number(el("batch").value),
    patience: Number(el("patience").value),
    val_ratio: Number(el("valRatio").value),
    device: el("device").value,
    seed: 42,
  };
  if (!window.confirm(`开始训练？\n已完成标注：${state.data.summary.annotated} 张\n轮数：${settings.epochs}`)) return;
  try {
    const training = await post("/api/training/start", settings);
    renderTrainingStatus(training); notify("训练任务已启动"); startTrainingPolling();
  } catch (error) { notify(error.message, true); }
}

async function cancelTraining() {
  if (!window.confirm("确认停止当前训练？已完成的 epoch 结果仍保留。")) return;
  try {
    const training = await post("/api/training/cancel", {});
    renderTrainingStatus(training); notify("训练已停止");
  } catch (error) { notify(error.message, true); }
}

function startTrainingPolling() {
  window.clearInterval(state.trainingTimer);
  state.trainingTimer = window.setInterval(async () => {
    try {
      const training = await api("/api/training/status");
      renderTrainingStatus(training);
      if (!["starting", "running"].includes(training.state)) {
        window.clearInterval(state.trainingTimer);
        await refreshState(true);
      }
    } catch (_error) {}
  }, 2000);
}

function bindEvents() {
  document.querySelectorAll(".tab-button").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".tab-button").forEach((item) => item.classList.toggle("active", item === button));
    document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === button.dataset.view));
  }));
  el("clientName").addEventListener("change", () => {
    const value = el("clientName").value.trim() || newClientId();
    state.clientId = value; localStorage.setItem("datasetStudioClient", value);
  });
  el("saveProjectButton").addEventListener("click", saveProject);
  el("previewButton").addEventListener("click", () => {
    el("cameraPreview").src = `/api/camera/preview?v=${Date.now()}`;
    el("cameraPreview").classList.add("ready"); el("cameraPlaceholder").style.display = "none";
  });
  el("cameraPreview").addEventListener("error", () => { el("cameraPreview").classList.remove("ready"); el("cameraPlaceholder").style.display = "grid"; notify("相机预览失败", true); });
  el("captureButton").addEventListener("click", captureImage);
  el("uploadInput").addEventListener("change", (event) => uploadImages(event.target.files));
  el("nextPendingButton").addEventListener("click", nextPending);
  el("activeClass").addEventListener("change", (event) => setActiveClass(Number(event.target.value)));
  el("negativeCheck").addEventListener("change", () => {
    if (el("negativeCheck").checked && state.boxes.length && window.confirm("勾选负样本会在保存时清空所有框，是否继续？")) {
      pushHistory(); state.boxes = []; state.selectedBox = -1; redraw(); renderBoxList();
    } else if (el("negativeCheck").checked && state.boxes.length) {
      el("negativeCheck").checked = false;
    }
  });
  el("undoButton").addEventListener("click", undo);
  el("deleteBoxButton").addEventListener("click", deleteSelectedBox);
  el("deleteImageButton").addEventListener("click", deleteImage);
  el("saveButton").addEventListener("click", () => saveAnnotation(false));
  el("saveNextButton").addEventListener("click", () => saveAnnotation(true));
  el("startTrainingButton").addEventListener("click", startTraining);
  el("cancelTrainingButton").addEventListener("click", cancelTraining);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); saveAnnotation(false); }
    if (event.key === "Delete" && document.activeElement?.tagName !== "INPUT") deleteSelectedBox();
    if (event.key >= "0" && event.key <= "9" && document.activeElement?.tagName !== "INPUT") {
      const index = Number(event.key); if (index < (state.data?.project.classes.length || 0)) setActiveClass(index);
    }
  });
  window.addEventListener("beforeunload", () => {
    if (state.current) navigator.sendBeacon("/api/unlock", new Blob([JSON.stringify({image_id: state.current.image_id, client_id: state.clientId})], {type: "application/json"}));
  });
}

async function initialize() {
  state.clientId = newClientId();
  el("clientName").value = state.clientId;
  bindEvents();
  setEditorEnabled(false);
  await refreshState();
  state.pollTimer = window.setInterval(() => refreshState(true), 5000);
  if (["starting", "running"].includes(state.data?.training.state)) startTrainingPolling();
}

initialize();
