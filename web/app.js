const videoSelect = document.querySelector("#video-select");
const runButton = document.querySelector("#run-button");
const uploadInput = document.querySelector("#video-file");
const uploadMessage = document.querySelector("#upload-message");
const runMessage = document.querySelector("#run-message");
const emptyState = document.querySelector("#empty-state");
const jobResults = document.querySelector("#job-results");
const jobState = document.querySelector("#job-state");
const modelNotice = document.querySelector("#model-notice");
const videoTrack = document.querySelector("#video-track");
const heroTitle = document.querySelector("#hero-title");
const heroSub = document.querySelector("#hero-sub");
const trackerChoice = document.querySelector("#tracker-choice");
const trackerSelect = document.querySelector("#tracker-select");
const modeText = {
  tracking: "Player tracking: detection with persistent IDs and trails.",
  detection: "Detection only: player bounding boxes in every frame.",
};
let currentJobId = null;
let pollTimer = null;

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status}).`);
  }
  return payload;
}

function setMessage(element, message, kind = "") {
  element.textContent = message;
  element.className = `inline-message ${kind}`.trim();
}

function hueFor(text) {
  let hue = 0;
  for (const char of text) hue = (hue * 31 + char.charCodeAt(0)) % 360;
  return hue;
}

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function renderVideoCards(videos) {
  videoTrack.querySelectorAll(".video-card").forEach((card) => card.remove());
  for (const video of videos) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "card video-card";
    card.dataset.id = video.id;
    card.setAttribute("role", "radio");
    const poster = document.createElement("span");
    poster.className = "poster";
    poster.style.setProperty("--h", hueFor(video.name));
    poster.innerHTML = '<span class="play"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15a.8.8 0 0 0 1.2.7l12-7.5a.8.8 0 0 0 0-1.4l-12-7.5A.8.8 0 0 0 7 4.5Z"/></svg></span>';
    const ext = document.createElement("span");
    ext.className = "ext";
    ext.textContent = (video.name.split(".").pop() || "").slice(0, 4).toUpperCase();
    poster.append(ext);
    const title = document.createElement("span");
    title.className = "card-title";
    title.textContent = video.name;
    title.title = video.name;
    card.append(poster, title);
    card.addEventListener("click", () => {
      videoSelect.value = video.id;
      syncSelection();
    });
    videoTrack.append(card);
  }
}

function syncSelection() {
  const id = videoSelect.value;
  videoTrack.querySelectorAll(".video-card").forEach((card) => {
    card.setAttribute("aria-checked", String(card.dataset.id === id));
  });
  const selected = videoSelect.selectedOptions[0];
  heroTitle.textContent = id && selected ? selected.textContent : "Choose a video";
  heroSub.textContent = id
    ? `${modeText[currentMode()]}${currentMode() === "tracking" ? ` Tracker: ${trackerSelect.value === "kmt" ? "KMT" : "ByteTrack"}.` : ""}`
    : "Pick a match video below or upload a new one to get started.";
}

async function refreshDashboard() {
  try {
    const [status, videoData] = await Promise.all([
      requestJson("/api/status"),
      requestJson("/api/videos"),
    ]);
    document.querySelector("#model-status").textContent = status.model_available ? "Ready" : "Not installed";
    document.querySelector("#model-detail").textContent = status.model_available ? "weights/best.pt found" : "Add weights/best.pt to run a pipeline";
    document.querySelector("#model-indicator").classList.toggle("ready", status.model_available);
    document.querySelector("#model-notice").hidden = status.model_available;
    document.querySelector("#homography-status").textContent = status.homography_available ? "Available" : "Not installed";
    document.querySelector("#homography-indicator").classList.toggle("ready", status.homography_available);
    document.querySelector("#video-count").textContent = videoData.videos.length.toString();

    const selectedId = videoSelect.value;
    videoSelect.replaceChildren();
    if (videoData.videos.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No videos yet — add one below";
      videoSelect.append(option);
    } else {
      for (const video of videoData.videos) {
        const option = document.createElement("option");
        option.value = video.id;
        option.textContent = video.name;
        videoSelect.append(option);
      }
      if (videoData.videos.some((video) => video.id === selectedId)) {
        videoSelect.value = selectedId;
      }
    }
    renderVideoCards(videoData.videos);
    syncSelection();
    if (status.active_job && !currentJobId) {
      currentJobId = status.active_job.id;
      showJob(status.active_job);
      beginPolling();
    }
  } catch (error) {
    setMessage(runMessage, error.message, "error");
  }
}

async function uploadVideo(file) {
  uploadInput.disabled = true;
  setMessage(uploadMessage, `Uploading ${file.name}…`);
  try {
    const uploaded = await requestJson(`/api/videos?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    await refreshDashboard();
    videoSelect.value = uploaded.id;
    syncSelection();
    videoTrack.querySelector(`[data-id="${CSS.escape(uploaded.id)}"]`)?.scrollIntoView({ inline: "nearest", block: "nearest", behavior: "smooth" });
    setMessage(uploadMessage, `${uploaded.name} is ready to use.`, "success");
  } catch (error) {
    setMessage(uploadMessage, error.message, "error");
  } finally {
    uploadInput.disabled = false;
    uploadInput.value = "";
  }
}

function renderOutputs(job) {
  const outputList = document.querySelector("#output-list");
  const previewContainer = document.querySelector("#preview-container");
  const previewVideo = document.querySelector("#preview-video");
  outputList.replaceChildren();
  const videoFile = job.files.find((file) => file.name.endsWith("_annotated.mp4") || file.name === "annotated.mp4");
  if (videoFile) {
    previewVideo.src = videoFile.url;
    previewContainer.hidden = false;
  } else {
    previewVideo.removeAttribute("src");
    previewContainer.hidden = true;
  }
  for (const file of job.files) {
    const link = document.createElement("a");
    link.className = "output-link";
    link.href = file.url;
    link.target = "_blank";
    link.rel = "noopener";
    const name = document.createElement("span");
    name.textContent = file.name;
    const action = document.createElement("span");
    action.textContent = "Open ↗";
    link.append(name, action);
    outputList.append(link);
  }
}

function showJob(job) {
  emptyState.hidden = true;
  jobResults.hidden = false;
  jobState.textContent = job.status;
  jobState.dataset.status = job.status;
  document.querySelector("#nav-results").dataset.running = String(job.status === "running");
  document.querySelector("#job-summary").textContent =
    `${job.mode === "tracking" ? `${job.tracker === "kmt" ? "KMT" : "ByteTrack"} tracking` : "Detection"} · ${job.video_name}`;
  renderOutputs(job);
  document.querySelector("#log-output").textContent = job.log || "Waiting for pipeline output…";
  runButton.disabled = job.status === "running";
  runButton.querySelector("span:first-child").textContent = job.status === "running" ? "Pipeline running…" : "Run pipeline";
}

function beginPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!currentJobId) return;
    try {
      const job = await requestJson(`/api/jobs/${encodeURIComponent(currentJobId)}`);
      showJob(job);
      if (job.status !== "running") {
        clearInterval(pollTimer);
        pollTimer = null;
        if (job.status === "failed") {
          setMessage(runMessage, `Pipeline failed (exit code ${job.return_code}). Check the process log.`, "error");
        } else if (job.status === "completed") {
          setMessage(runMessage, "Pipeline completed. Your outputs are ready.", "success");
        } else if (job.status === "cancelled") {
          setMessage(runMessage, "Pipeline was stopped.", "error");
        }
      }
    } catch (error) {
      clearInterval(pollTimer);
      setMessage(runMessage, error.message, "error");
    }
  }, 1500);
}

async function startPipeline() {
  setMessage(runMessage, "");
  if (!videoSelect.value) {
    setMessage(runMessage, "Choose or upload a video first.", "error");
    return;
  }
  runButton.disabled = true;
  try {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const tracker = trackerSelect.value;
    const job = await requestJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoSelect.value, mode, tracker }),
    });
    currentJobId = job.id;
    showJob(job);
    beginPolling();
    showView("results");
  } catch (error) {
    runButton.disabled = false;
    runButton.querySelector("span:first-child").textContent = "Run pipeline";
    setMessage(runMessage, error.message, "error");
  }
}

uploadInput.addEventListener("change", () => {
  const [file] = uploadInput.files;
  if (file) uploadVideo(file);
});
runButton.addEventListener("click", startPipeline);
document.querySelector("#refresh-videos").addEventListener("click", refreshDashboard);
document.querySelector("#toggle-log").addEventListener("click", (event) => {
  const log = document.querySelector("#log-output");
  log.hidden = !log.hidden;
  event.currentTarget.textContent = log.hidden ? "Show log" : "Hide log";
});
document.querySelectorAll('input[name="mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    document.querySelectorAll(".mode-option").forEach((option) => option.classList.remove("selected"));
    input.closest(".mode-option").classList.add("selected");
    trackerChoice.hidden = input.value !== "tracking";
    syncSelection();
  });
});
trackerSelect.addEventListener("change", syncSelection);
trackerChoice.hidden = currentMode() !== "tracking";

refreshDashboard();

const viewNames = ["home", "results"];

function showView(name) {
  if (!viewNames.includes(name)) name = "home";
  document.querySelectorAll("[data-view]").forEach((section) => {
    section.hidden = section.dataset.view !== name;
  });
  document.querySelectorAll(".nav-link").forEach((link) => {
    const active = link.dataset.nav === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
showView(location.hash.slice(1));