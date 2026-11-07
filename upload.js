"use strict";

const get = id => document.getElementById(id);
const form = get("projectForm");
const plan = get("planFile");
const videos = get("videoFiles");
const status = get("status");
const submit = get("submit");
const result = get("result");
const fields = [...form.querySelectorAll("input"), submit];
let uploaded;

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Проверьте введённые данные.");
  return data;
}

function put(path, slot, file) {
  return api(path + "/files/" + slot + "?filename=" + encodeURIComponent(file.name), {
    method: "PUT", body: file,
  });
}

async function watch(path) {
  while (true) {
    const state = await api(path);
    status.textContent = state.message || "Сеанс готов";
    if (state.state === "uploading") throw new Error("Загрузка этого проекта ещё не завершена.");
    if (state.state === "failed") throw new Error(state.message);
    if (state.state === "ready") {
      result.href = state.url;
      result.hidden = false;
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

form.addEventListener("input", () => {
  uploaded = undefined;
  const files = [...videos.files];
  get("cameraCount").textContent = files.length
    ? files.map((file, i) => "Камера " + (i + 1) + " — " + file.name).join("; ")
    : "Видео пока не выбраны.";
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  const files = [...videos.files];
  const roi = get("clockRoi").value.split(",").map(Number);
  fields.forEach(field => { field.disabled = true; });
  result.hidden = true;
  try {
    if (roi.length !== 4 || !roi.every(Number.isFinite)) throw new Error("Укажите четыре числа для области часов.");
    if (!uploaded) {
      const created = await api("/api/projects", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({cameras: files.length, clock_roi: roi}),
      });
      const path = "/api/projects/" + created.project_id;
      status.textContent = "Загрузка плана";
      await put(path, "plan", plan.files[0]);
      for (const [i, file] of files.entries()) {
        status.textContent = "Загрузка видео " + (i + 1) + " из " + files.length;
        await put(path, i + 1, file);
      }
      uploaded = path;
    }
    const path = uploaded;
    await api(path + "/start", {method: "POST"});
    uploaded = undefined;
    history.replaceState(null, "", "?project=" + path.split("/").at(-1));
    await watch(path);
  } catch (error) {
    status.textContent = error.message;
  } finally {
    fields.forEach(field => { field.disabled = false; });
  }
});

const pending = new URLSearchParams(location.search).get("project");
if (/^[a-f0-9]{32}$/.test(pending ?? "")) {
  fields.forEach(field => { field.disabled = true; });
  watch("/api/projects/" + pending).catch(error => {
    status.textContent = error.message;
  }).finally(() => fields.forEach(field => { field.disabled = false; }));
}
