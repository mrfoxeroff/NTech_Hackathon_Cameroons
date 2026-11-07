"use strict";

const element = id => document.getElementById(id);
const sessionPrefix = location.pathname.match(/^\/projects\/[a-f0-9]{32}(?=\/|$)/)?.[0] || "";
const sessionUrl = path => sessionPrefix + path;
const timeline = element("timeline");
const canvas = element("mapCanvas");
const ctx = canvas.getContext("2d");
const plan = new Image();
const mapView = { level: 0, scale: 1, x: 0, y: 0 };
const maxMapZoom = 6;
const slots = [1, 2].map(number => ({
  number,
  video: element("video" + number),
  select: element("slot" + number + "Select"),
  title: element("slot" + number + "Title"),
  overlay: element("overlay" + number),
  camera: null,
  mediaTime: null,
  boxes: null,
}));
const overlayCache = new Map();

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

function formatBoundary(seconds) {
  const ms = Math.round(seconds * 1000);
  return String(Math.floor(ms / 60000)).padStart(2, "0") + ":"
    + ((ms % 60000) / 1000).toFixed(3).padStart(6, "0");
}

function initTrackLog() {
  const dialog = element("trackLog");
  const table = element("trackLogRows");
  for (const track of [...tracks].sort((a, b) => a.start_time - b.start_time || a.id - b.id)) {
    const row = table.insertRow();
    const button = document.createElement("button");
    button.type = "button";
    button.className = "id-card";
    button.style.color = track.color;
    button.textContent = "ID " + track.id;
    button.onclick = () => {
      dialog.close();
      if (playing) togglePlay();
      selectedId = track.id;
      seek(Math.max(metadata.start_time, Math.min(track.start_time, metadata.end_time)));
    };
    row.insertCell().append(button);
    row.insertCell().textContent = formatBoundary(track.start_time);
    row.insertCell().textContent = formatBoundary(track.end_time);
    row.insertCell().textContent = track.cameras_visited.join(" → ");
  }
  const open = element("openTrackLog");
  open.onclick = () => dialog.showModal();
  open.disabled = false;
}

function covered(slot) {
  return metadata.cameras[slot.camera].coverage.some(([start, end]) => currentTime >= start && currentTime <= end);
}

function lowerBound(rows, value, key) {
  let lo = 0;
  let hi = rows.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (key(rows[mid]) < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function interpolateClock(anchors, value, from, to) {
  const index = Math.max(0, Math.min(anchors.length - 2, lowerBound(anchors, value, row => row[from]) - 1));
  const a = anchors[index];
  const b = anchors[index + 1];
  return a[to] + (value - a[from]) * (b[to] - a[to]) / (b[from] - a[from]);
}

function videoTime(camera, time) {
  if (!camera.overlay_url) return time - camera.start;
  const local = time - camera.time_offset_sec;
  return camera.clock_anchors.length
    ? interpolateClock(camera.clock_anchors, local, "timestamp_sec", "video_time_sec")
    : local;
}

function commonTime(camera, time) {
  if (!camera.overlay_url) return time + camera.start;
  const mapped = camera.clock_anchors.length
    ? interpolateClock(camera.clock_anchors, time, "video_time_sec", "timestamp_sec")
    : time;
  return mapped + camera.time_offset_sec;
}

function drawOverlay(slot, clear = false) {
  const layer = slot.overlay;
  const width = layer.clientWidth;
  const height = layer.clientHeight;
  if (!width || !height) return;
  if (layer.width !== width || layer.height !== height) {
    layer.width = width;
    layer.height = height;
  }
  const drawing = layer.getContext("2d");
  drawing.clearRect(0, 0, width, height);
  if (clear || !slot.boxes || slot.mediaTime === null || slot.video.seeking || !covered(slot)) return;
  const {frames} = slot.boxes;
  const index = lowerBound(frames, slot.mediaTime, frame => frame[0]);
  const before = frames[index - 1];
  const after = frames[index];
  const frame = !before || (after && after[0] - slot.mediaTime < slot.mediaTime - before[0]) ? after : before;
  if (!frame || Math.abs(frame[0] - slot.mediaTime) > .16) return;
  const camera = metadata.cameras[slot.camera];
  if (Math.abs(commonTime(camera, slot.mediaTime) - currentTime) > .20) return;
  const scale = Math.min(width / slot.boxes.width, height / slot.boxes.height);
  const dx = (width - scale * slot.boxes.width) / 2;
  const dy = (height - scale * slot.boxes.height) / 2;
  drawing.font = "bold 13px sans-serif";
  for (const [ident, x1, y1, x2, y2] of frame[1]) {
    const track = metadata.tracks_summary[ident];
    drawing.strokeStyle = drawing.fillStyle = track.color;
    drawing.lineWidth = ident === selectedId ? 3 : 1.5;
    const x = dx + x1 * scale;
    const y = dy + y1 * scale;
    drawing.strokeRect(x, y, (x2 - x1) * scale, (y2 - y1) * scale);
    drawing.fillText("ID " + ident, x + 2, Math.max(dy + 13, y - 4));
  }
}

function receiveVideoFrame(slot, mediaTime) {
  slot.mediaTime = mediaTime;
  drawOverlay(slot);
}

function syncSlot(slot, seek = false) {
  if (!slot.camera) return;
  const {video} = slot;
  const camera = metadata.cameras[slot.camera];
  const available = covered(slot) && !video.error;
  slot.title.textContent = slot.camera + (video.error ? " · ошибка видео" : available ? "" : " · нет кадра");
  video.style.visibility = available ? "visible" : "hidden";
  if (!available || !playing) video.pause();
  if (!available || video.readyState < 1) {
    drawOverlay(slot, true);
    return;
  }
  const target = Math.max(0, Math.min(video.duration, videoTime(camera, currentTime)));
  if (seek || (!video.seeking && Math.abs(video.currentTime - target) > 0.14)) {
    drawOverlay(slot, true);
    video.currentTime = target;
  }
  if (camera.overlay_url) {
    const rate = (videoTime(camera, currentTime + .04) - target) / .04;
    video.playbackRate = Math.min(2, Math.max(.5, rate));
  }
  if (playing && video.paused && !video.seeking) video.play().catch(() => {});
}

function setSlotCam(number, camera) {
  const slot = slots[number - 1];
  if (slot.camera === camera || !metadata.cameras[camera]) return;
  slot.camera = camera;
  slot.boxes = null;
  slot.mediaTime = null;
  slot.select.value = camera;
  slot.video.src = metadata.cameras[camera].url;
  drawOverlay(slot, true);
  const url = metadata.cameras[camera].overlay_url;
  if (url) {
    if (!overlayCache.has(url)) overlayCache.set(url, fetch(url).then(response => {
      if (!response.ok) throw new Error("Не удалось загрузить подписи " + camera);
      return response.json();
    }));
    overlayCache.get(url).then(data => {
      if (slot.camera !== camera) return;
      slot.boxes = data;
      drawOverlay(slot);
    }).catch(error => {
      element("sessionStatus").textContent = error.message;
      overlayCache.delete(url);
    });
  }
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
}

function positionAt(track) {
  const index = lowerBound(track.map_path, currentTime, point => point.t);
  const point = track.map_path[index]?.t === currentTime ? track.map_path[index] : track.map_path[index - 1];
  return point && currentTime - point.t <= metadata.visibility_gap_sec
    && visibleSegments(track).some(segment => segment.cam === point.cam && point.t >= segment.start - .12)
    ? point : null;
}

function mapPointer(event) {
  const rect = canvas.getBoundingClientRect();
  const fit = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  return {
    x: (event.clientX - rect.left - rect.width / 2) / fit + canvas.width / 2,
    y: (event.clientY - rect.top - rect.height / 2) / fit + canvas.height / 2,
  };
}

function drawCurrentMap() {
  mapView.x = Math.max(canvas.width * (1 - mapView.scale), Math.min(0, mapView.x));
  mapView.y = Math.max(canvas.height * (1 - mapView.scale), Math.min(0, mapView.y));
  drawMap(tracks.filter(track => visibleSegments(track).length), metadata.tracks_summary[selectedId]);
}

function zoomMap(level, anchor = { x: canvas.width / 2, y: canvas.height / 2 }) {
  level = Math.max(0, Math.min(maxMapZoom, level));
  const scale = 1.25 ** level;
  const ratio = scale / mapView.scale;
  mapView.x = anchor.x - (anchor.x - mapView.x) * ratio;
  mapView.y = anchor.y - (anchor.y - mapView.y) * ratio;
  mapView.level = level;
  mapView.scale = scale;
  canvas.classList.toggle("map-zoomed", level > 0);
  element("mapZoomOut").disabled = element("mapZoomReset").disabled = level === 0;
  element("mapZoomIn").disabled = level === maxMapZoom;
  element("mapZoomValue").value = Math.round(scale * 100) + "%";
  drawCurrentMap();
}

function initMapControls() {
  let drag;
  element("mapZoomOut").onclick = () => zoomMap(mapView.level - 1);
  element("mapZoomIn").onclick = () => zoomMap(mapView.level + 1);
  element("mapZoomReset").onclick = () => zoomMap(0);
  canvas.addEventListener("wheel", event => {
    const anchor = mapPointer(event);
    if (!event.deltaY || anchor.x < 0 || anchor.x > canvas.width
      || anchor.y < 0 || anchor.y > canvas.height) return;
    const level = mapView.level + (event.deltaY < 0 ? 1 : -1);
    if (level < 0 || level > maxMapZoom) return;
    event.preventDefault();
    zoomMap(level, anchor);
  }, { passive: false });
  canvas.addEventListener("pointerdown", event => {
    if (drag || mapView.level === 0 || event.button !== 0) return;
    drag = { id: event.pointerId, point: mapPointer(event) };
    canvas.setPointerCapture(drag.id);
    canvas.classList.add("map-dragging");
  });
  canvas.addEventListener("pointermove", event => {
    if (drag?.id !== event.pointerId) return;
    const point = mapPointer(event);
    mapView.x += point.x - drag.point.x;
    mapView.y += point.y - drag.point.y;
    drag.point = point;
    drawCurrentMap();
  });
  canvas.addEventListener("lostpointercapture", event => {
    if (drag?.id !== event.pointerId) return;
    drag = null;
    canvas.classList.remove("map-dragging");
  });
  element("mapZoomIn").disabled = false;
}

function drawMap(active, selected) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.translate(mapView.x, mapView.y);
  ctx.scale(mapView.scale, mapView.scale);
  if (plan.complete && plan.naturalWidth) ctx.drawImage(plan, 0, 0);
  for (const track of selected ? [selected] : active) {
    let previous;
    ctx.strokeStyle = ctx.fillStyle = track.color;
    ctx.lineWidth = 2;
    for (const point of track.map_path) {
      if (point.t > currentTime) break;
      if (previous && !point.gap_before) {
        ctx.setLineDash(point.status === "estimated" || previous.status === "estimated" ? [5, 5] : []);
        ctx.beginPath();
        ctx.moveTo(previous.x, previous.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      }
      previous = point;
    }
    ctx.setLineDash([]);
    const point = positionAt(track);
    if (point) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "bold 12px sans-serif";
      ctx.fillText("ID " + track.id, point.x + 8, point.y + 4);
    }
  }
  ctx.restore();
}

function render() {
  timeline.value = currentTime;
  element("timeLabel").textContent = "t = " + formatTime(currentTime) + " / " + formatTime(metadata.end_time);
  if (metadata.localization && metadata.timeline_origin) {
    const origin = Date.parse(metadata.timeline_origin + "Z");
    if (Number.isFinite(origin)) {
      const clock = element("clockLabel");
      clock.textContent = "OSD " + new Date(origin + currentTime * 1000).toISOString().slice(0, 19).replace("T", " ");
      clock.hidden = false;
    }
  }
  const active = tracks.filter(track => visibleSegments(track).length);
  const selected = metadata.tracks_summary[selectedId];
  renderCards(active);
  const passport = selected
    ? "ID " + selected.id + " · " + formatTime(selected.start_time) + " — " + formatTime(selected.end_time)
      + " · " + selected.cameras_visited.join(" → ")
      + (visibleSegments(selected).length && !positionAt(selected) ? " · Позиция на плане неизвестна" : "")
    : "Выберите ID, чтобы увидеть маршрут и привязать камеры.";
  if (element("trackPassport").textContent !== passport) element("trackPassport").textContent = passport;
  if (selected && element("autoSwitchCam").checked) selectCameras(selected);
  drawMap(active, selected);
  slots.forEach(slot => drawOverlay(slot));
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
    const response = await fetch(sessionUrl("/api/session"));
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Не удалось загрузить сеанс.");
    metadata = data;
    tracks = Object.values(data.tracks_summary);
    initTrackLog();
    canvas.width = data.map.width;
    canvas.height = data.map.height;
    plan.onload = render;
    plan.src = data.map.url;
    const unlocalized = data.localization?.unlocalized_cameras ?? [];
    if (unlocalized.length) {
      const warning = element("localizationStatus");
      warning.textContent = "Нет подтверждённой привязки к плану: " + unlocalized.join(", ");
      warning.hidden = false;
    }
    element("totalTracksBadge").textContent = "Подтверждённые треки за сеанс: " + data.total_unique_tracks;
    timeline.min = data.start_time;
    timeline.max = data.end_time;
    timeline.step = "any";
    currentTime = data.start_time;
    const cameras = Object.keys(data.cameras);
    for (const slot of slots) {
      for (const camera of cameras) slot.select.add(new Option(camera, camera));
      slot.video.addEventListener("loadedmetadata", () => syncSlot(slot, true));
      slot.video.addEventListener("seeked", () => {
        if (!slot.video.requestVideoFrameCallback) receiveVideoFrame(slot, slot.video.currentTime);
        syncSlot(slot);
        drawOverlay(slot);
      });
      slot.video.addEventListener("error", () => syncSlot(slot));
      if (slot.video.requestVideoFrameCallback) {
        const onFrame = (_, frame) => {
          receiveVideoFrame(slot, frame.mediaTime);
          slot.video.requestVideoFrameCallback(onFrame);
        };
        slot.video.requestVideoFrameCallback(onFrame);
      } else {
        slot.video.addEventListener("timeupdate", () => receiveVideoFrame(slot, slot.video.currentTime));
      }
      setSlotCam(slot.number, cameras[(slot.number - 1) % cameras.length]);
    }
    timeline.oninput = () => seek(Number(timeline.value));
    element("autoSwitchCam").onchange = render;
    timeline.disabled = element("playBtn").disabled = false;
    element("sessionStatus").textContent = "";
    initMapControls();
    render();
  } catch (error) {
    element("sessionStatus").textContent = error.message;
  }
}

init();
fetch("/api/projects/enabled").then(response => {
  element("newProject").hidden = !response.ok;
}).catch(() => {});
