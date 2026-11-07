"use strict";

const byId = (id) => document.getElementById(id);
const timeline = byId("timeline");
const playButton = byId("playBtn");
const trackSelect = byId("trackSelect");
const status = byId("status");
const autoSwitch = byId("autoSwitchCam");
const mapCanvas = byId("mapCanvas");
const mapContext = mapCanvas.getContext("2d");
const slots = [1, 2].map((id) => {
  const video = byId("video" + id);
  return {
    id, video, canvas: byId("overlay" + id), select: byId("slot" + id + "Select"),
    camera: null, presentedTime: null, pendingSeek: false, outOfRange: false,
    frameCallback: null,
  };
});
let metadata, timing, mapImage;
let tracks = [];
let cameraIds = [];
let selected = null;
let time = 0;
let playing = false;
let animation = null;
let activeKey = null;
const overlays = new Map();

async function readJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(url + " (HTTP " + response.status + ")");
  return response.json();
}

function lowerBound(items, value, key) {
  let left = 0;
  let right = items.length;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (items[middle][key] < value) left = middle + 1;
    else right = middle;
  }
  return left;
}

function convertClock(value, camera, fromVideo) {
  const {clock_anchors: anchors, time_offset_sec: offset} = timing[camera];
  if (!anchors.length) return fromVideo ? value + offset : value - offset;
  const from = fromVideo ? "video_time_sec" : "timestamp_sec";
  const to = fromVideo ? "timestamp_sec" : "video_time_sec";
  const input = fromVideo ? value : value - offset;
  const index = Math.min(anchors.length - 1, Math.max(1, lowerBound(anchors, input, from)));
  const a = anchors[index - 1];
  const b = anchors[index];
  const output = a[to] + (input - a[from]) * (b[to] - a[to]) / (b[from] - a[from]);
  return fromVideo ? output + offset : output;
}

function formatTime(seconds) {
  const tenths = Math.round(seconds * 10);
  const minutes = String(Math.floor(tenths / 600)).padStart(2, "0");
  const rest = ((tenths % 600) / 10).toFixed(1).padStart(4, "0");
  return minutes + ":" + rest;
}

function clearOverlay(slot) {
  slot.canvas.getContext("2d").clearRect(0, 0, slot.canvas.width, slot.canvas.height);
}

function drawOverlay(slot) {
  const {video, canvas, presentedTime} = slot;
  const data = overlays.get(slot.camera);
  if (!data || presentedTime === null || video.seeking || slot.pendingSeek || slot.outOfRange) {
    clearOverlay(slot);
    return;
  }
  const {frames, frame_width: width, frame_height: height} = data;
  const index = lowerBound(frames, presentedTime, "video_time_sec");
  const frame = [frames[index - 1], frames[index]].filter(Boolean).reduce((best, item) =>
    !best || Math.abs(item.video_time_sec - presentedTime) < Math.abs(best.video_time_sec - presentedTime)
      ? item : best, null);
  if (!frame || Math.abs(frame.video_time_sec - presentedTime) > 0.008) {
    clearOverlay(slot);
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(video.clientWidth * dpr);
  const h = Math.round(video.clientHeight * dpr);
  if (!w || !h) return;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const scale = Math.min(w / width, h / height);
  const left = (w - width * scale) / 2;
  const top = (h - height * scale) / 2;
  ctx.font = "bold " + 13 * dpr + "px system-ui";
  ctx.textBaseline = "top";
  for (const {global_id: id, silhouette_polygons: polygons} of frame.observations) {
    const track = metadata.tracks_summary[id];
    if (!track) continue;
    ctx.strokeStyle = track.color;
    ctx.fillStyle = track.color;
    ctx.lineWidth = (selected?.id === id ? 3 : 1.5) * dpr;
    let label = null;
    for (const polygon of polygons) {
      if (polygon.length < 3) continue;
      ctx.beginPath();
      for (const [x, y] of polygon) {
        const px = left + x * scale;
        const py = top + y * scale;
        ctx.lineTo(px, py);
        if (!label || py < label[1]) label = [px, py];
      }
      ctx.closePath();
      ctx.stroke();
    }
    if (label) {
      const text = "ID " + id;
      const y = Math.max(top, label[1] - 17 * dpr);
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(label[0], y, ctx.measureText(text).width + 6 * dpr, 16 * dpr);
      ctx.fillStyle = track.color;
      ctx.fillText(text, label[0] + 3 * dpr, y);
    }
  }
}

function watchFrames(slot) {
  const {video} = slot;
  if (!video.requestVideoFrameCallback || slot.frameCallback !== null) return;
  slot.frameCallback = video.requestVideoFrameCallback((_now, frame) => {
    slot.frameCallback = null;
    slot.presentedTime = frame.mediaTime;
    drawOverlay(slot);
    watchFrames(slot);
  });
}

function positionSlot(slot, tolerance = 0.001) {
  const {video} = slot;
  if (!slot.camera || video.readyState < 1) return;
  const target = convertClock(metadata.timeline_start_sec + time, slot.camera, false);
  const outside = target < 0 || target >= video.duration;
  if (outside !== slot.outOfRange) {
    slot.outOfRange = outside;
    video.style.visibility = outside ? "hidden" : "";
    slot.presentedTime = null;
    clearOverlay(slot);
  }
  if (outside) {
    video.pause();
  } else if (Math.abs(video.currentTime - target) > tolerance) {
    slot.presentedTime = null;
    clearOverlay(slot);
    video.currentTime = target;
  }
}

function playSlot(slot) {
  if (slot.video.paused && !slot.pendingSeek && !slot.outOfRange) {
    slot.video.play().catch((error) => {
      if (playing && error.name !== "AbortError") {
        stop();
        status.textContent = "Не удалось воспроизвести " + slot.camera + ". Проверь MP4.";
      }
    });
  }
}

function setCamera(slot, camera) {
  if (slot.camera === camera) return;
  slot.video.pause();
  slot.camera = camera;
  slot.select.value = camera;
  byId("slot" + slot.id + "Title").textContent = "Слот " + slot.id + ": " + camera;
  slot.pendingSeek = true;
  slot.outOfRange = false;
  slot.video.style.visibility = "";
  slot.presentedTime = null;
  clearOverlay(slot);
  slot.video.src = metadata.cameras[camera];
  watchFrames(slot);

  if (overlays.has(camera)) return;
  overlays.set(camera, null);
  readJSON(metadata.overlay_urls[camera]).then((data) => {
    if (data.schema_version !== 1 || data.camera_id !== camera) throw new Error("Формат оверлея");
    overlays.set(camera, data);
    if (slot.camera === camera) drawOverlay(slot);
    for (const other of slots) if (other !== slot && other.camera === camera) drawOverlay(other);
  }).catch(() => {
    overlays.delete(camera);
    status.textContent = "Не удалось загрузить контуры " + camera;
  });
}

function autoSelectCameras(track) {
  const visible = track.camera_segments.filter((s) => time >= s.start && time <= s.end);
  const next = track.camera_segments.filter((s) => s.start > time);
  const ids = [...new Set([...visible, ...next].map((s) => s.cam)
    .concat(track.cameras_visited, cameraIds))];
  slots.forEach((slot, index) => setCamera(slot, ids[index]));
}

function drawMap() {
  const ctx = mapContext;
  const {width, height} = mapCanvas;
  ctx.clearRect(0, 0, width, height);
  if (!mapImage?.complete || !mapImage.naturalWidth) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "15px system-ui";
    ctx.fillText("План офиса не загружен", 18, 28);
    return;
  }
  const scale = Math.min(width / metadata.map.width_px, height / metadata.map.height_px);
  const w = metadata.map.width_px * scale;
  const h = metadata.map.height_px * scale;
  const x = (width - w) / 2;
  const y = (height - h) / 2;
  ctx.drawImage(mapImage, x, y, w, h);
  if (!selected) return;
  const points = selected.map_path;
  ctx.strokeStyle = selected.color;
  ctx.lineWidth = 3;
  for (let i = 1; i < points.length && points[i].t <= time; i++) {
    const a = points[i - 1];
    const b = points[i];
    if (b.t - a.t > 0.5) continue;
    const observed = a.status === "observed" && b.status === "observed" &&
      selected.camera_segments.some((s) => a.cams.includes(s.cam) && b.cams.includes(s.cam) &&
        a.t >= s.start - 0.001 && b.t <= s.end + 0.001);
    ctx.setLineDash(observed ? [] : [5, 5]);
    ctx.beginPath();
    ctx.moveTo(x + a.x * scale, y + a.y * scale);
    ctx.lineTo(x + b.x * scale, y + b.y * scale);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  const current = points[lowerBound(points, time + 0.000001, "t") - 1];
  if (current && time - current.t <= 0.5) {
    ctx.beginPath();
    ctx.arc(x + current.x * scale, y + current.y * scale, 5, 0, Math.PI * 2);
    ctx.fillStyle = selected.color;
    if (current.status === "observed") ctx.fill();
    else ctx.stroke();
  }
}

function render() {
  timeline.value = time;
  byId("timeLabel").textContent = "t = " + formatTime(time);
  const active = tracks.filter((track) =>
    track.camera_segments.some((segment) => segment.start <= time && time <= segment.end));
  const key = active.map((t) => t.id).join(",") + "/" + selected?.id;
  if (key !== activeKey) {
    activeKey = key;
    byId("activeIdsContainer").replaceChildren(...active.map((track) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "id-card" + (selected === track ? " selected" : "");
      button.textContent = "ID " + track.id + " · " + track.cameras_visited.join(" → ");
      button.style.borderColor = track.color;
      button.onclick = () => selectTrack(selected === track ? null : track, false);
      return button;
    }));
  }
  if (selected && autoSwitch.checked) autoSelectCameras(selected);
  drawMap();
}

function selectTrack(track, jump) {
  selected = track;
  trackSelect.value = track?.id ?? "";
  const passport = byId("trackPassport");
  passport.replaceChildren();
  if (!track) {
    passport.textContent = "Выбери сквозной трек, чтобы посмотреть интервалы по камерам.";
  } else {
    const title = document.createElement("strong");
    title.textContent = "Трек ID " + track.id + " · " + formatTime(track.start_time) +
      " – " + formatTime(track.end_time);
    passport.append(title);
    for (const {cam, start, end} of track.camera_segments) {
      const line = document.createElement("div");
      line.textContent = cam + " · " + formatTime(start) + " – " + formatTime(end);
      passport.append(line);
    }
  }
  if (jump && track) seek(track.start_time);
  else render();
  slots.forEach(drawOverlay);
}

function seek(value) {
  time = Math.max(0, Math.min(metadata.duration_sec, value));
  render();
  slots.forEach((slot) => positionSlot(slot));
}

function stop() {
  playing = false;
  cancelAnimationFrame(animation);
  animation = null;
  slots.forEach(({video}) => video.pause());
  playButton.textContent = "▶ Play";
}

function playbackTick() {
  const master = slots.find((slot) => !slot.outOfRange);
  if (master && !master.pendingSeek && !master.video.seeking) {
    const now = convertClock(master.video.currentTime, master.camera, true) - metadata.timeline_start_sec;
    if (now >= metadata.duration_sec || master.video.ended) {
      stop();
      seek(Math.min(now, metadata.duration_sec));
      return;
    }
    time = Math.max(0, now);
    render();
    for (const slot of slots) {
      if (slot !== master && !slot.video.seeking) positionSlot(slot, 0.15);
    }
  }
  for (const slot of slots) {
    if (!slot.video.requestVideoFrameCallback) {
      slot.presentedTime = slot.video.currentTime;
      drawOverlay(slot);
    }
    if (slot !== master && (master?.pendingSeek || master?.video.readyState < 3)) slot.video.pause();
    else playSlot(slot);
  }
  animation = requestAnimationFrame(playbackTick);
}

async function init() {
  metadata = await readJSON("/static/tracks_metadata.json");
  if (metadata.schema_version !== 1) throw new Error("Неизвестная версия JSON");
  timing = (await readJSON(metadata.clock_alignment_url)).cameras;
  cameraIds = Object.keys(metadata.cameras);
  tracks = Object.values(metadata.tracks_summary).sort((a, b) => a.start_time - b.start_time);
  byId("totalTracksBadge").textContent = "Всего сквозных треков в эпизоде: " + metadata.total_unique_tracks;
  timeline.max = metadata.duration_sec;
  for (const track of tracks) trackSelect.add(new Option("ID " + track.id, track.id));
  trackSelect.onchange = () => selectTrack(metadata.tracks_summary[trackSelect.value] ?? null, true);
  autoSwitch.onchange = render;

  slots.forEach((slot, index) => {
    const {video} = slot;
    for (const camera of cameraIds) slot.select.add(new Option(camera, camera));
    slot.select.onchange = () => {
      autoSwitch.checked = false;
      setCamera(slot, slot.select.value);
    };
    video.onloadedmetadata = () => {
      positionSlot(slot);
      slot.pendingSeek = false;
    };
    video.onloadeddata = () => {
      if (!video.requestVideoFrameCallback) slot.presentedTime = video.currentTime;
      drawOverlay(slot);
    };
    video.onseeking = () => clearOverlay(slot);
    video.onseeked = () => {
      if (!video.requestVideoFrameCallback) slot.presentedTime = video.currentTime;
      drawOverlay(slot);
    };
    video.onerror = () => {
      stop();
      clearOverlay(slot);
      status.textContent = "Не удалось открыть видео " + slot.camera;
    };
    setCamera(slot, cameraIds[index]);
  });
  if (metadata.map) {
    mapImage = new Image();
    mapImage.onload = drawMap;
    mapImage.onerror = () => { status.textContent = "Не удалось загрузить план офиса"; };
    mapImage.src = metadata.map.url;
  }
  window.addEventListener("resize", () => slots.forEach(drawOverlay));
  timeline.oninput = () => seek(Number(timeline.value));
  playButton.onclick = () => {
    if (playing) {
      stop();
      return;
    }
    if (time >= metadata.duration_sec - 0.001 || slots.every((slot) => slot.video.ended)) seek(0);
    playing = true;
    playButton.textContent = "⏸ Pause";
    playbackTick();
  };
  playButton.disabled = timeline.disabled = trackSelect.disabled = false;
  selectTrack(null, false);
}

init().catch((error) => {
  status.textContent = "Не удалось загрузить данные. Выполни export_ui.py по README. " + error.message;
});
