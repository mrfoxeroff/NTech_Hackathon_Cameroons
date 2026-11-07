"use strict";

const element = id => document.getElementById(id);
const timeline = element("timeline");
const canvas = element("mapCanvas");
const ctx = canvas.getContext("2d");
const plan = new Image();
const slots = [1, 2].map(number => ({
  number,
  video: element("video" + number),
  select: element("slot" + number + "Select"),
  title: element("slot" + number + "Title"),
  camera: null,
}));

let metadata;
let tracks = [];
let currentTime = 0;
let selectedId = null;
let playing = false;
let lastTick = 0;
let lastPaint = 0;
let cardState = "";
let animationFrame;

function formatTime(seconds) {
  const tenths = Math.round(seconds * 10);
  const minutes = String(Math.floor(tenths / 600)).padStart(2, "0");
  return minutes + ":" + ((tenths % 600) / 10).toFixed(1).padStart(4, "0");
}

function covered(slot) {
  return metadata.cameras[slot.camera].coverage.some(([start, end]) => currentTime >= start && currentTime <= end);
}

function syncSlot(slot, seek = false) {
  if (!slot.camera) return;
  const {video} = slot;
  const camera = metadata.cameras[slot.camera];
  const available = covered(slot) && !video.error;
  slot.title.textContent = slot.camera + (video.error ? " · ошибка видео" : available ? "" : " · нет кадра");
  video.style.visibility = available ? "visible" : "hidden";
  if (!available || !playing) video.pause();
  if (!available || video.readyState < 1) return;
  const target = Math.min(video.duration, currentTime - camera.start);
  if (seek || (!video.seeking && Math.abs(video.currentTime - target) > 0.15)) {
    video.currentTime = target;
  }
  if (playing && video.paused && !video.seeking) video.play().catch(() => {});
}

function setSlotCam(number, camera) {
  const slot = slots[number - 1];
  if (slot.camera === camera || !metadata.cameras[camera]) return;
  slot.camera = camera;
  slot.select.value = camera;
  slot.video.src = metadata.cameras[camera].url;
  syncSlot(slot, true);
}

function manualCamChange(number, camera) {
  element("autoSwitchCam").checked = false;
  setSlotCam(number, camera);
}

function visibleSegments(track) {
  return track.camera_segments.filter(segment => currentTime >= segment.start && currentTime <= segment.end);
}

function selectCameras(track) {
  const active = [...new Set(visibleSegments(track).map(segment => segment.cam))];
  if (active.length === 1) {
    const others = track.camera_segments.filter(segment => segment.cam !== active[0]);
    const distance = segment => Math.min(Math.abs(segment.start - currentTime), Math.abs(segment.end - currentTime));
    others.sort((a, b) => distance(a) - distance(b));
    if (others[0]) active.push(others[0].cam);
  }
  active.slice(0, 2).forEach((camera, index) => setSlotCam(index + 1, camera));
}

function renderCards(active) {
  const key = JSON.stringify([selectedId, active.map(track => track.id)]);
  if (key === cardState) return;
  cardState = key;
  const container = element("activeIdsContainer");
  container.replaceChildren();
  if (!active.length) container.textContent = "Нет активных треков";
  for (const track of active) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "id-card" + (selectedId === track.id ? " selected" : "");
    card.textContent = "ID " + track.id + " · " + track.cameras_visited.join(" → ");
    card.style.color = track.color;
    card.onclick = () => {
      selectedId = selectedId === track.id ? null : track.id;
      render();
      slots.forEach(slot => syncSlot(slot));
    };
    container.append(card);
  }
  const selected = metadata.tracks_summary[selectedId];
  element("trackPassport").textContent = selected
    ? "ID " + selected.id + " · " + formatTime(selected.start_time) + " — " + formatTime(selected.end_time)
      + " · " + selected.cameras_visited.join(" → ")
    : "Выберите ID, чтобы увидеть маршрут и привязать камеры.";
}

function drawMap(active, selected) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (plan.complete && plan.naturalWidth) ctx.drawImage(plan, 0, 0);
  for (const track of selected ? [selected] : active) {
    let previous;
    ctx.strokeStyle = ctx.fillStyle = track.color;
    ctx.lineWidth = 2;
    for (const point of track.map_path) {
      if (point.t > currentTime) break;
      if (previous) {
        ctx.setLineDash(point.gap_before || point.status === "estimated" || previous.status === "estimated" ? [5, 5] : []);
        ctx.beginPath();
        ctx.moveTo(previous.x, previous.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      }
      previous = point;
    }
    ctx.setLineDash([]);
    if (previous && visibleSegments(track).length && currentTime - previous.t <= metadata.visibility_gap_sec) {
      ctx.beginPath();
      ctx.arc(previous.x, previous.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "bold 12px sans-serif";
      ctx.fillText("ID " + track.id, previous.x + 8, previous.y + 4);
    }
  }
}

function render() {
  timeline.value = currentTime;
  element("timeLabel").textContent = "t = " + formatTime(currentTime);
  const active = tracks.filter(track => visibleSegments(track).length);
  const selected = metadata.tracks_summary[selectedId];
  renderCards(active);
  if (selected && element("autoSwitchCam").checked) selectCameras(selected);
  drawMap(active, selected);
}

function seek(time) {
  currentTime = time;
  render();
  slots.forEach(slot => syncSlot(slot, true));
}

function togglePlay() {
  if (!metadata) return;
  if (!playing && currentTime >= metadata.end_time) seek(metadata.start_time);
  playing = !playing;
  element("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
  slots.forEach(slot => syncSlot(slot));
  if (playing) {
    lastTick = performance.now();
    animationFrame = requestAnimationFrame(tick);
  } else {
    cancelAnimationFrame(animationFrame);
  }
}

function tick(now) {
  if (!playing) return;
  const elapsed = Math.min(0.2, (now - lastTick) / 1000);
  lastTick = now;
  const loading = slots.some(slot => slot.camera && covered(slot) && !slot.video.error
    && (slot.video.seeking || slot.video.readyState < 2));
  if (!loading) currentTime = Math.min(metadata.end_time, currentTime + elapsed);
  if (now - lastPaint > 65 || currentTime === metadata.end_time) {
    lastPaint = now;
    render();
    slots.forEach(slot => syncSlot(slot));
  }
  if (currentTime === metadata.end_time) togglePlay();
  else animationFrame = requestAnimationFrame(tick);
}

async function init() {
  try {
    const response = await fetch("/api/session");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Не удалось загрузить сеанс.");
    metadata = data;
    tracks = Object.values(data.tracks_summary);
    canvas.width = data.map.width;
    canvas.height = data.map.height;
    plan.onload = render;
    plan.src = data.map.url;
    element("totalTracksBadge").textContent = "Всего уникальных сквозных треков: " + data.total_unique_tracks;
    timeline.min = data.start_time;
    timeline.max = data.end_time;
    timeline.step = "any";
    currentTime = data.start_time;
    const cameras = Object.keys(data.cameras);
    for (const slot of slots) {
      for (const camera of cameras) slot.select.add(new Option(camera, camera));
      slot.video.addEventListener("loadedmetadata", () => syncSlot(slot, true));
      slot.video.addEventListener("seeked", () => syncSlot(slot));
      slot.video.addEventListener("error", () => syncSlot(slot));
      setSlotCam(slot.number, cameras[(slot.number - 1) % cameras.length]);
    }
    timeline.oninput = () => seek(Number(timeline.value));
    element("autoSwitchCam").onchange = render;
    timeline.disabled = element("playBtn").disabled = false;
    element("sessionStatus").textContent = "";
    render();
  } catch (error) {
    element("sessionStatus").textContent = error.message;
  }
}

init();
