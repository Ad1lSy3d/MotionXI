// ============================================================================
// CourtVision AI — Football Match Analytics Dashboard
// ============================================================================

// State
let currentJobId = null;
let currentJob = null;
let currentAnalytics = null;
let selectedPlayerId = null;
let pollTimer = null;
let isSeekingVideo = false;
let isSeekingPitch = false;
let activeRosterFilter = "sustained";
let tacticalTimelineFrame = 0;
let currentPitchMode = "current"; // "current" | "avg"

// DOM Elements: Navigation & Views
const navHome = document.querySelector("#nav-home");
const navResults = document.querySelector("#nav-results");
const viewHome = document.querySelector('[data-view="home"]');
const viewResults = document.querySelector('[data-view="results"]');

// DOM Elements: Setup / Home
const videoSelect = document.querySelector("#video-select");
const videoTrack = document.querySelector("#video-track");
const runButton = document.querySelector("#run-button");
const demoButton = document.querySelector("#demo-button");
const emptyLoadDemoBtn = document.querySelector("#empty-load-demo");
const quickDemoBtn = document.querySelector("#quick-demo-btn");
const uploadInput = document.querySelector("#video-file");
const uploadMessage = document.querySelector("#upload-message");
const runMessage = document.querySelector("#run-message");
const heroTitle = document.querySelector("#hero-title");
const heroSub = document.querySelector("#hero-sub");
const trackerChoice = document.querySelector("#tracker-choice");
const trackerSelect = document.querySelector("#tracker-select");
const trackerHelpText = document.querySelector("#tracker-help-text");
const modelNotice = document.querySelector("#model-notice");

// DOM Elements: Results Page
const emptyState = document.querySelector("#empty-state");
const jobResults = document.querySelector("#job-results");
const resultsTitle = document.querySelector("#results-title");
const jobState = document.querySelector("#job-state");
const jobModeBadge = document.querySelector("#job-mode-badge");
const jobTrackerBadge = document.querySelector("#job-tracker-badge");
const jobCalibBadge = document.querySelector("#job-calib-badge");
const cancelJobBtn = document.querySelector("#cancel-job-btn");

// Video & Canvas Overlay
const videoWrapper = document.querySelector("#video-wrapper");
const previewVideo = document.querySelector("#preview-video");
const videoOverlay = document.querySelector("#video-overlay");
const videoLoading = document.querySelector("#video-loading");
const videoFallbackBanner = document.querySelector("#video-fallback-banner");
const videoSourceLabel = document.querySelector("#video-source-label");
const btnModeInteractive = document.querySelector("#btn-mode-interactive");
const btnModeAnnotated = document.querySelector("#btn-mode-annotated");
let currentVideoMode = "interactive";

const ctrlPlay = document.querySelector("#ctrl-play");
const iconPlay = ctrlPlay ? ctrlPlay.querySelector(".icon-play") : null;
const iconPause = ctrlPlay ? ctrlPlay.querySelector(".icon-pause") : null;
const videoScrubber = document.querySelector("#video-scrubber");
const timeCurrent = document.querySelector("#time-current");
const timeTotal = document.querySelector("#time-total");
const frameCounter = document.querySelector("#frame-counter");
const toggleBBoxes = document.querySelector("#toggle-bboxes");
const toggleTags = document.querySelector("#toggle-tags");
const ctrlSpeed = document.querySelector("#ctrl-speed");

// Export Dropdown
const exportBtn = document.querySelector("#export-btn");
const exportMenu = document.querySelector("#export-menu");
const exportJson = document.querySelector("#export-json");
const exportCsv = document.querySelector("#export-csv");
const exportMetricsCsv = document.querySelector("#export-metrics-csv");
const exportVideo = document.querySelector("#export-video");

// Match Overview Metrics Strip (Prioritized 5 Critical Metrics)
const mDuration = document.querySelector("#m-duration");
const mFrames = document.querySelector("#m-frames");
const mFps = document.querySelector("#m-fps");
const mLatency = document.querySelector("#m-latency");
const mPlayers = document.querySelector("#m-players");
const mPlayersSub = document.querySelector("#m-players-sub");
const mCalibTracks = document.querySelector("#m-calib-tracks");
const mCalibSub = document.querySelector("#m-calib-sub");
const mTeamDist = document.querySelector("#m-team-dist");
const mTopSpeed = document.querySelector("#m-top-speed");
const mTopSpeedSub = document.querySelector("#m-top-speed-sub");
const mSprintsCount = document.querySelector("#m-sprints-count");

// Roster & Sidebar
const playerRoster = document.querySelector("#player-roster");
const rosterCount = document.querySelector("#roster-count");
const playerSearch = document.querySelector("#player-search");
const filterChips = document.querySelectorAll(".chip-filter");
const filterAllCount = document.querySelector("#filter-all-count");
const filterSustainedCount = document.querySelector("#filter-sustained-count");
const filterCalibCount = document.querySelector("#filter-calib-count");
const filterTransientCount = document.querySelector("#filter-transient-count");
const sbTag = document.querySelector("#sb-tag");
const sbMeta = document.querySelector("#sb-meta");
const sbDist = document.querySelector("#sb-dist");
const sbSpeed = document.querySelector("#sb-speed");
const sbDot = document.querySelector("#sb-dot");

// Diagnostics
const diagOutliersTable = document.querySelector("#diag-outliers-table tbody");
const diagOutliersCount = document.querySelector("#diag-outliers-count");
const diagPeakSimul = document.querySelector("#diag-peak-simul");
const diagAvgSimul = document.querySelector("#diag-avg-simul");
const diagSustainedCount = document.querySelector("#diag-sustained-count");
const diagTransientCount = document.querySelector("#diag-transient-count");
const diagTotalIds = document.querySelector("#diag-total-ids");
const diagCrowdedTable = document.querySelector("#diag-crowded-table tbody");
const diagCrowdedCount = document.querySelector("#diag-crowded-count");
const diagIdswTable = document.querySelector("#diag-idsw-table tbody");
const diagIdswCount = document.querySelector("#diag-idsw-count");
const logOutput = document.querySelector("#log-output");
const toggleLogBtn = document.querySelector("#toggle-log");
const outputList = document.querySelector("#output-list");

// Player Detail Tab
const pdTag = document.querySelector("#pd-tag");
const pdSub = document.querySelector("#pd-sub");
const pdBadge = document.querySelector("#pd-badge");
const pdDist = document.querySelector("#pd-dist");
const pdDistKm = document.querySelector("#pd-dist-km");
const pdAvgSpeed = document.querySelector("#pd-avg-speed");
const pdTopSpeed = document.querySelector("#pd-top-speed");
const pdSprints = document.querySelector("#pd-sprints");
const pdCoverage = document.querySelector("#pd-coverage");
const pdSamples = document.querySelector("#pd-samples");
const pdOcclusions = document.querySelector("#pd-occlusions");
const pdJumpVideo = document.querySelector("#pd-jump-video");
const pdViewHeatmap = document.querySelector("#pd-view-heatmap");

// Player Comparison
const comparePlayerA = document.querySelector("#compare-player-a");
const comparePlayerB = document.querySelector("#compare-player-b");
const comparisonBars = document.querySelector("#comparison-bars");

// Heatmaps Tab
const hmSwitchTeam = document.querySelector("#hm-switch-team");
const hmSwitchPlayer = document.querySelector("#hm-switch-player");
const hmPlayerSelectWrap = document.querySelector("#hm-player-select-wrap");
const hmPlayerSelect = document.querySelector("#hm-player-select");
const heatmapImage = document.querySelector("#heatmap-image");
const heatmapPlaceholder = document.querySelector("#heatmap-placeholder");
const hmDownloadLink = document.querySelector("#hm-download-link");
const heatmapCaption = document.querySelector("#heatmap-caption");

// Tactical Pitch (2D)
const pitchCanvas = document.querySelector("#pitch-canvas");
const pitchAllTrails = document.querySelector("#pitch-all-trails");
const pitchModeCurrent = document.querySelector("#pitch-mode-current");
const pitchModeAvg = document.querySelector("#pitch-mode-avg");
const pitchTrailsToggleWrap = document.querySelector("#pitch-trails-toggle-wrap");
const pitchTimelineControl = document.querySelector("#pitch-timeline-control");
const pitchPlayBtn = document.querySelector("#pitch-play-btn");
const iconPitchPlay = pitchPlayBtn ? pitchPlayBtn.querySelector(".icon-pitch-play") : null;
const iconPitchPause = pitchPlayBtn ? pitchPlayBtn.querySelector(".icon-pitch-pause") : null;
const pitchScrubber = document.querySelector("#pitch-scrubber");
const pitchFrameLabel = document.querySelector("#pitch-frame-label");
const tacWidth = document.querySelector("#tac-width");
const tacLength = document.querySelector("#tac-length");
const tacCentroid = document.querySelector("#tac-centroid");
const tacActiveCount = document.querySelector("#tac-active-count");



// Leaderboards
const leadDistance = document.querySelector("#lead-distance");
const leadSpeed = document.querySelector("#lead-speed");
const leadSprints = document.querySelector("#lead-sprints");

// Telemetry
const tTrackerName = document.querySelector("#t-tracker-name");
const tCameraMotion = document.querySelector("#t-camera-motion");
const tDetectorModel = document.querySelector("#t-detector-model");
const tCalibScope = document.querySelector("#t-calib-scope");
const tResolution = document.querySelector("#t-resolution");
const calibWarningText = document.querySelector("#calib-warning-text");

// Helper: HSL Color for Track ID (Deterministic Golden Ratio Angle)
function colorForTrack(trackId) {
  const hue = Math.round((Number(trackId) * 137.5) % 360);
  return `hsl(${hue}, 85%, 55%)`;
}

function tagForTrack(trackId) {
  const num = parseInt(trackId, 10);
  return isNaN(num) ? `P${trackId}` : `P${String(num).padStart(2, "0")}`;
}

// HTTP API Wrapper
async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status}).`);
  }
  return payload;
}

function setMessage(element, message, kind = "") {
  if (!element) return;
  element.textContent = message;
  element.className = `inline-message ${kind}`.trim();
}

// Navigation View Routing
const viewNames = ["home", "results"];
function showView(name) {
  if (!viewNames.includes(name)) name = "home";
  document.querySelectorAll("[data-view]").forEach((sec) => {
    sec.hidden = sec.dataset.view !== name;
  });
  document.querySelectorAll(".nav-link").forEach((link) => {
    const active = link.dataset.nav === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  window.scrollTo(0, 0);

  if (name === "results" && currentAnalytics) {
    resizeOverlayCanvas();
    drawPitchCanvas();
  }
}
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

// Tab Switching
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const targetId = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b === btn);
      b.setAttribute("aria-selected", String(b === btn));
    });
    document.querySelectorAll(".tab-pane").forEach((pane) => {
      pane.hidden = pane.id !== targetId;
      pane.classList.toggle("active", pane.id === targetId);
    });
    if (targetId === "tab-tactical") {
      drawPitchCanvas();
    }
  });
});

// ============================================================================
// DASHBOARD REFRESH & VIDEO MANAGEMENT (HOME VIEW)
// ============================================================================

function currentMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "tracking";
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
    poster.innerHTML = '<span class="card-play-icon"><svg viewBox="0 0 24 24" aria-hidden="true" width="20" height="20"><path d="M7 4.5v15a.8.8 0 0 0 1.2.7l12-7.5a.8.8 0 0 0 0-1.4l-12-7.5A.8.8 0 0 0 7 4.5Z" fill="currentColor"/></svg></span>';

    const title = document.createElement("span");
    title.className = "card-title";
    title.textContent = video.name;
    title.title = video.name;

    const meta = document.createElement("span");
    meta.className = "card-meta";
    meta.textContent = (video.name.split(".").pop() || "mp4").toUpperCase();

    card.append(poster, title, meta);
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
  heroTitle.textContent = id && selected ? selected.textContent : "Football Match Analytics";
  heroSub.textContent = id
    ? `Configured for ${currentMode() === "tracking" ? `Player Tracking with ${trackerSelect.value === "kmt" ? "KMT (Kinematic Memory)" : "ByteTrack"}` : "Detection Only"}. Click "Run Pipeline" to start processing.`
    : "Select a match video below or launch the verified demonstration match to explore occlusion-aware player tracking, physical performance metrics, and tactical pitch heatmaps.";
}

async function refreshDashboard() {
  try {
    const [status, videoData] = await Promise.all([
      requestJson("/api/status"),
      requestJson("/api/videos"),
    ]);

    const modelStatusEl = document.querySelector("#model-status");
    const modelIndEl = document.querySelector("#model-indicator");
    if (modelStatusEl) modelStatusEl.textContent = status.model_available ? "Ready (YOLOv11m)" : "Not installed";
    if (modelIndEl) modelIndEl.classList.toggle("ready", status.model_available);
    if (modelNotice) modelNotice.hidden = status.model_available;

    const homographyStatusEl = document.querySelector("#homography-status");
    const homographyIndEl = document.querySelector("#homography-indicator");
    if (homographyStatusEl) homographyStatusEl.textContent = status.homography_available ? "18-Yard Box" : "Not installed";
    if (homographyIndEl) homographyIndEl.classList.toggle("ready", status.homography_available);

    const videoCountEl = document.querySelector("#video-count");
    if (videoCountEl) videoCountEl.textContent = videoData.videos.length.toString();

    const demoChip = document.querySelector("#demo-chip");
    if (demoChip) demoChip.hidden = !status.demo_available;

    const selectedId = videoSelect.value;
    videoSelect.replaceChildren();
    if (videoData.videos.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No videos yet — upload one below";
      videoSelect.append(option);
    } else {
      for (const video of videoData.videos) {
        const option = document.createElement("option");
        option.value = video.id;
        option.textContent = video.name;
        videoSelect.append(option);
      }
      if (videoData.videos.some((v) => v.id === selectedId)) {
        videoSelect.value = selectedId;
      }
    }
    renderVideoCards(videoData.videos);
    syncSelection();
    const urlParams = new URLSearchParams(window.location.search);
    const hashMatch = window.location.hash.match(/^(?:#job=|#results\?job=)([a-zA-Z0-9_-]+)/);
    const targetJobId = urlParams.get("job") || (hashMatch ? hashMatch[1] : null);

    if (targetJobId && !currentJobId) {
      currentJobId = targetJobId;
      try {
        const targetJob = await requestJson(`/api/jobs/${encodeURIComponent(targetJobId)}`);
        showJob(targetJob);
        showView("results");
      } catch (err) {
        console.warn("Could not load target job from URL:", err);
      }
    } else if (status.active_job && !currentJobId) {
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
    videoTrack.querySelector(`[data-id="${CSS.escape(uploaded.id)}"]`)?.scrollIntoView({ inline: "nearest", behavior: "smooth" });
    setMessage(uploadMessage, `${uploaded.name} uploaded successfully.`, "success");
  } catch (error) {
    setMessage(uploadMessage, error.message, "error");
  } finally {
    uploadInput.disabled = false;
    uploadInput.value = "";
  }
}

// Mode & Tracker switch events
document.querySelectorAll('input[name="mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    document.querySelectorAll(".mode-option").forEach((opt) => opt.classList.remove("selected"));
    input.closest(".mode-option").classList.add("selected");
    trackerChoice.hidden = input.value !== "tracking";
    syncSelection();
  });
});

if (trackerSelect) {
  trackerSelect.addEventListener("change", () => {
    if (trackerHelpText) {
      if (trackerSelect.value === "kmt") {
        trackerHelpText.innerHTML = "<strong>Kinematic Memory Tracker (KMT):</strong> Maintains player identity through dense crowds and physical occlusions by caching velocity vectors and gating re-identifications.";
      } else {
        trackerHelpText.innerHTML = "<strong>ByteTrack Baseline:</strong> High-speed bounding-box IoU association. Fast baseline, but prone to ID switches in dense penalty box clusters.";
      }
    }
    syncSelection();
  });
}

// ============================================================================
// PIPELINE EXECUTION & DEMO SHOWCASE LAUNCH
// ============================================================================

async function startPipeline() {
  setMessage(runMessage, "");
  if (!videoSelect.value) {
    setMessage(runMessage, "Please select or upload a match video first.", "error");
    return;
  }
  runButton.disabled = true;
  try {
    const mode = currentMode();
    const tracker = trackerSelect ? trackerSelect.value : "kmt";
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
    setMessage(runMessage, error.message, "error");
  }
}

async function loadDemoMatch() {
  currentJobId = "demo";
  try {
    const job = await requestJson("/api/jobs/demo");
    showJob(job);
    showView("results");
  } catch (error) {
    alert("Unable to load demo match: " + error.message);
  }
}

if (demoButton) demoButton.addEventListener("click", loadDemoMatch);
if (emptyLoadDemoBtn) emptyLoadDemoBtn.addEventListener("click", loadDemoMatch);
if (quickDemoBtn) quickDemoBtn.addEventListener("click", loadDemoMatch);
if (runButton) runButton.addEventListener("click", startPipeline);

if (cancelJobBtn) {
  cancelJobBtn.addEventListener("click", async () => {
    if (!currentJobId) return;
    try {
      cancelJobBtn.disabled = true;
      const job = await requestJson(`/api/jobs/${encodeURIComponent(currentJobId)}/cancel`, { method: "POST" });
      showJob(job);
    } catch (e) {
      alert("Failed to cancel job: " + e.message);
    } finally {
      cancelJobBtn.disabled = false;
    }
  });
}

function beginPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!currentJobId || currentJobId === "demo") return;
    try {
      const job = await requestJson(`/api/jobs/${encodeURIComponent(currentJobId)}`);
      showJob(job);
      if (job.status !== "running") {
        clearInterval(pollTimer);
        pollTimer = null;
        if (job.status === "failed") {
          setMessage(runMessage, `Pipeline failed (exit code ${job.return_code}). Inspect diagnostics log.`, "error");
        } else if (job.status === "completed") {
          setMessage(runMessage, "Pipeline processing completed.", "success");
        } else if (job.status === "cancelled") {
          setMessage(runMessage, "Pipeline execution was stopped.", "error");
        }
      }
    } catch (error) {
      clearInterval(pollTimer);
      setMessage(runMessage, error.message, "error");
    }
  }, 1400);
}

// ============================================================================
// RESULTS VIEW & ANALYTICS RENDERING
// ============================================================================

async function showJob(job) {
  currentJob = job;
  emptyState.hidden = true;
  jobResults.hidden = false;

  jobState.textContent = job.status;
  jobState.dataset.status = job.status;
  if (navResults) navResults.dataset.running = String(job.status === "running");

  if (jobModeBadge) jobModeBadge.textContent = job.mode === "tracking" ? "Tracking" : "Detection";
  if (jobTrackerBadge) {
    jobTrackerBadge.textContent = (job.tracker || "kmt").toUpperCase();
    jobTrackerBadge.hidden = job.mode !== "tracking";
  }

  resultsTitle.textContent = job.video_name || "Match Analytics";
  if (cancelJobBtn) cancelJobBtn.hidden = job.status !== "running";

  if (runButton) {
    runButton.disabled = job.status === "running";
    runButton.querySelector("span:first-child").textContent = job.status === "running" ? "Pipeline Running…" : "Run Pipeline";
  }

  renderOutputFiles(job);
  if (logOutput) logOutput.textContent = job.log || "No execution log recorded.";

  // Fetch Full Analytics Payload
  try {
    const analytics = await requestJson(`/api/jobs/${encodeURIComponent(currentJobId)}/analytics`);
    if (analytics.has_analytics) {
      currentAnalytics = analytics;
      renderAnalyticsDashboard(analytics);
    }
  } catch (err) {
    console.warn("Analytics not yet available for job:", err);
  }
}

function setVideoMode(mode, isFallback = false) {
  currentVideoMode = mode;
  if (btnModeInteractive) btnModeInteractive.classList.toggle("active", mode === "interactive");
  if (btnModeAnnotated) btnModeAnnotated.classList.toggle("active", mode === "annotated");

  if (videoSourceLabel) {
    videoSourceLabel.textContent = mode === "interactive" ? "Clean Match Footage (Interactive Overlay)" : "Pipeline Annotated Video (Pre-rendered)";
  }

  if (!isFallback && videoFallbackBanner) {
    videoFallbackBanner.hidden = true;
  }

  const cleanUrl = currentAnalytics?.summary?.clean_video_url;
  const annoUrl = currentAnalytics?.summary?.annotated_video_url;
  const targetUrl = mode === "interactive" ? (cleanUrl || annoUrl) : annoUrl;

  if (targetUrl && previewVideo) {
    const curTime = previewVideo.currentTime || 0;
    const isPlaying = !previewVideo.paused;
    if (!previewVideo.src.endsWith(targetUrl)) {
      previewVideo.src = targetUrl;
      previewVideo.currentTime = curTime;
      previewVideo.load();
      if (isPlaying) {
        previewVideo.play().catch(() => {});
      }
    }
  }

  resizeOverlayCanvas();
  drawVideoOverlay();
}

if (btnModeInteractive) {
  btnModeInteractive.addEventListener("click", () => setVideoMode("interactive"));
}
if (btnModeAnnotated) {
  btnModeAnnotated.addEventListener("click", () => setVideoMode("annotated"));
}

function renderOutputFiles(job) {
  if (!outputList) return;
  outputList.replaceChildren();

  const videoFile = job.files.find((f) => f.name.endsWith(".mp4") && (f.name.includes("annotated") || f.name.includes("kmt")));
  if (videoFile) {
    if (exportVideo) exportVideo.href = videoFile.url;
  }

  const tracksCsvFile = job.files.find((f) => f.name === "tracks.csv");
  if (tracksCsvFile && exportCsv) exportCsv.href = tracksCsvFile.url;

  const metricsCsvFile = job.files.find((f) => f.name === "player_physical_metrics.csv");
  if (metricsCsvFile && exportMetricsCsv) exportMetricsCsv.href = metricsCsvFile.url;

  if (exportJson) exportJson.href = `/api/jobs/${encodeURIComponent(currentJobId)}/analytics`;

  for (const file of job.files) {
    const link = document.createElement("a");
    link.className = "output-link";
    link.href = file.url;
    link.target = "_blank";
    link.rel = "noopener";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = file.name;

    const actionSpan = document.createElement("span");
    actionSpan.textContent = "Open ↗";

    link.append(nameSpan, actionSpan);
    outputList.append(link);
  }
}

// Render All Analytics Views
function renderAnalyticsDashboard(analytics) {
  const sum = analytics.summary || {};
  const players = analytics.players || [];
  const tactical = analytics.tactical || {};
  const diag = analytics.diagnostics || {};

  // 1. Match Overview Strip (Prioritized 5 Critical Metrics)
  if (mDuration) mDuration.textContent = `${sum.duration_sec || 0}s`;
  if (mFrames) mFrames.textContent = `${sum.total_frames || 0} frames · ${sum.fps || 30} FPS`;
  if (mFps) mFps.textContent = sum.effective_processing_fps ? `${sum.effective_processing_fps} FPS` : `${sum.fps || 30} FPS`;
  if (mLatency) {
    const lat = sum.avg_tracker_assoc_time_ms ? `${sum.avg_tracker_assoc_time_ms}ms latency` : "Real-time inference";
    mLatency.textContent = lat;
  }
  if (mPlayers) mPlayers.textContent = `${sum.peak_simultaneous_tracks || '—'}`;
  if (mPlayersSub) {
    mPlayersSub.textContent = `Avg Active: ${sum.avg_simultaneous_tracks || 0} tracks · ${sum.total_unique_tracks || players.length} unique IDs`;
  }
  if (mTeamDist) mTeamDist.textContent = `${sum.total_team_distance_m || 0} m`;
  if (mCalibSub) {
    mCalibSub.textContent = `18-yd penalty box (${sum.calibrated_tracks || 0} players calibrated)`;
  }
  if (mTopSpeed) mTopSpeed.textContent = sum.match_top_speed_kmh ? `${sum.match_top_speed_kmh} km/h` : "—";
  if (mTopSpeedSub) {
    const rej = sum.outliers_rejected_count || 0;
    mTopSpeedSub.textContent = `Biomechanical filter passed (${rej} speed outliers rejected)`;
  }

  // 2. Leaderboards
  renderLeaderboards(players);

  // 3. Telemetry Tab
  if (tTrackerName) tTrackerName.textContent = sum.tracker_name || "KinematicMemoryTracker";
  if (tCameraMotion) tCameraMotion.textContent = sum.camera_motion_enabled ? "Enabled (Homography translation)" : "Disabled";
  if (tDetectorModel) tDetectorModel.textContent = sum.detector_model || "YOLOv11m";
  if (tCalibScope) tCalibScope.textContent = sum.calibration_scope === "full_pitch" ? "Full Pitch" : "18-Yard Penalty Box";
  if (tResolution) tResolution.textContent = sum.resolution || "1920x1080";
  if (calibWarningText && sum.calibration_warning) calibWarningText.textContent = sum.calibration_warning;

  // 4. Tactical Strip Metrics
  if (tacWidth) tacWidth.textContent = `${tactical.avg_team_width_m || 0} m`;
  if (tacLength) tacLength.textContent = `${tactical.avg_team_length_m || 0} m`;
  if (tacCentroid && tactical.team_centroid) {
    tacCentroid.textContent = `(${tactical.team_centroid[0]}, ${tactical.team_centroid[1]})`;
  }
  if (pitchScrubber) {
    pitchScrubber.max = sum.total_frames ? sum.total_frames - 1 : 90;
  }

  // 5. Diagnostics Tables
  renderDiagnostics(diag, sum);

  // 6. Right Sidebar Player Roster & Filter Counts
  if (filterAllCount) filterAllCount.textContent = String(players.length);
  if (filterSustainedCount) {
    filterSustainedCount.textContent = String(sum.sustained_tracks_count != null ? sum.sustained_tracks_count : players.filter(p => p.is_sustained).length);
  }
  if (filterCalibCount) {
    filterCalibCount.textContent = String(sum.calibrated_tracks != null ? sum.calibrated_tracks : players.filter(p => p.has_calibrated_data).length);
  }
  if (filterTransientCount) {
    filterTransientCount.textContent = String(sum.transient_tracks_count != null ? sum.transient_tracks_count : players.filter(p => !p.is_sustained).length);
  }
  renderPlayerRoster(players);

  // 7. Populate Comparison Selectors & Heatmap Selectors
  populatePlayerSelectors(players);

  // 8. Auto-select top player if none selected
  if (!selectedPlayerId && players.length > 0) {
    selectPlayer(players[0].track_id);
  } else if (selectedPlayerId) {
    selectPlayer(selectedPlayerId);
  }

  // 9. Initial Heatmap display
  showTeamHeatmap();

  // 10. Draw 2D pitch
  drawPitchCanvas();

  // 11. Activate video mode
  setVideoMode(currentVideoMode);
}

function renderLeaderboards(players) {
  if (leadDistance) {
    leadDistance.replaceChildren();
    const sortedDist = [...players].sort((a, b) => b.distance_m - a.distance_m).slice(0, 5);
    sortedDist.forEach((p, idx) => {
      leadDistance.append(createLeaderItem(idx + 1, p, `${p.distance_m} m`));
    });
  }

  if (leadSpeed) {
    leadSpeed.replaceChildren();
    const sortedSpeed = [...players].filter((p) => p.top_speed_kmh).sort((a, b) => b.top_speed_kmh - a.top_speed_kmh).slice(0, 5);
    sortedSpeed.forEach((p, idx) => {
      leadSpeed.append(createLeaderItem(idx + 1, p, `${p.top_speed_kmh} km/h`));
    });
  }

  if (leadSprints) {
    leadSprints.replaceChildren();
    const sortedSprints = [...players].sort((a, b) => b.sprint_count - a.sprint_count).slice(0, 5);
    sortedSprints.forEach((p, idx) => {
      leadSprints.append(createLeaderItem(idx + 1, p, `${p.sprint_count} sprints`));
    });
  }
}

function createLeaderItem(rank, player, valText) {
  const li = document.createElement("li");
  li.className = "lead-item";
  li.innerHTML = `
    <span class="lead-tag">
      <span class="lead-rank">#${rank}</span>
      <span class="player-color-dot" style="background: ${player.color}"></span>
      <span>${player.tag}</span>
    </span>
    <span class="lead-val">${valText}</span>
  `;
  li.addEventListener("click", () => selectPlayer(player.track_id));
  return li;
}

// Diagnostics Rendering
function renderDiagnostics(diag, sum = {}) {
  // Speed Outlier Jump Audit Table
  if (diagOutliersTable) {
    diagOutliersTable.replaceChildren();
    const outliers = diag.speed_outliers || [];
    if (diagOutliersCount) {
      diagOutliersCount.textContent = `${sum.outliers_rejected_count || outliers.length} jumps rejected`;
    }
    if (outliers.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="6" style="text-align: center; color: var(--muted); padding: 16px;">No speed outliers detected. All coordinate jumps passed physical biomechanical validation (&le;38 km/h).</td>`;
      diagOutliersTable.append(tr);
    } else {
      outliers.forEach((ev) => {
        const tr = document.createElement("tr");
        tr.className = "outlier-row";
        const color = colorForTrack(ev.track_id);
        tr.innerHTML = `
          <td>
            <span class="lead-tag">
              <span class="player-color-dot" style="background: ${color}"></span>
              <strong>${ev.tag}</strong>
            </span>
          </td>
          <td><strong>Frame ${ev.from_frame} &rarr; ${ev.to_frame}</strong></td>
          <td>${ev.dt_s.toFixed(3)}s</td>
          <td>${ev.distance_m.toFixed(2)} m</td>
          <td><span class="outlier-spike-val">${ev.raw_speed_kmh.toFixed(1)} km/h</span></td>
          <td>
            <span class="badge-rejected">REJECTED</span>
            <span class="outlier-reason-note">${ev.reason || "Exceeds 38 km/h threshold"}</span>
          </td>
        `;
        tr.title = `Click to seek video to Frame ${ev.from_frame}`;
        tr.addEventListener("click", () => seekVideoToFrame(ev.from_frame));
        diagOutliersTable.append(tr);
      });
    }
  }

  // Tracking Identity Fragmentation Card
  const frag = diag.tracking_fragmentation || {};
  if (diagPeakSimul) diagPeakSimul.textContent = String(frag.peak_simultaneous != null ? frag.peak_simultaneous : (sum.peak_simultaneous_tracks || "—"));
  if (diagAvgSimul) diagAvgSimul.textContent = String(frag.avg_simultaneous != null ? frag.avg_simultaneous : (sum.avg_simultaneous_tracks || "—"));
  if (diagSustainedCount) diagSustainedCount.textContent = String(frag.sustained_tracks != null ? frag.sustained_tracks : (sum.sustained_tracks_count || "—"));
  if (diagTransientCount) diagTransientCount.textContent = String(frag.transient_tracks != null ? frag.transient_tracks : (sum.transient_tracks_count || "—"));
  if (diagTotalIds) diagTotalIds.textContent = String(frag.total_tracker_ids != null ? frag.total_tracker_ids : (sum.total_unique_tracks || "—"));

  if (diagCrowdedTable) {
    diagCrowdedTable.replaceChildren();
    const crowded = diag.crowded_frames || [];
    if (diagCrowdedCount) diagCrowdedCount.textContent = `${crowded.length} frames`;
    crowded.forEach((c) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${c.frame}</strong></td>
        <td>${c.timestamp.toFixed(2)}s</td>
        <td>${c.player_count}</td>
        <td>${c.max_pairwise_iou.toFixed(3)}</td>
        <td>${c.overlapping_pairs}</td>
      `;
      tr.addEventListener("click", () => seekVideoToFrame(c.frame));
      diagCrowdedTable.append(tr);
    });
  }

  if (diagIdswTable) {
    diagIdswTable.replaceChildren();
    const idsw = diag.id_switch_candidates || [];
    if (diagIdswCount) diagIdswCount.textContent = `${idsw.length} events`;
    idsw.forEach((ev) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${ev.frame}</strong></td>
        <td>${ev.candidate_type}</td>
        <td>${ev.track_id_1} &rarr; ${ev.track_id_2}</td>
        <td>${ev.description}</td>
      `;
      tr.addEventListener("click", () => seekVideoToFrame(ev.frame));
      diagIdswTable.append(tr);
    });
  }
}

// ============================================================================
// PLAYER IDENTITY SYSTEM & ROSTER
// ============================================================================

function renderPlayerRoster(players) {
  if (!playerRoster) return;
  playerRoster.replaceChildren();

  // Sort sustained tracks first, then calibrated tracks, then by distance and frame counts
  const sorted = [...players].sort((a, b) => {
    if (a.is_sustained !== b.is_sustained) return b.is_sustained ? 1 : -1;
    if (a.has_calibrated_data !== b.has_calibrated_data) return b.has_calibrated_data ? 1 : -1;
    return (b.distance_m || 0) - (a.distance_m || 0) || (b.total_samples || 0) - (a.total_samples || 0);
  });

  const query = (playerSearch ? playerSearch.value : "").trim().toLowerCase();
  const filtered = sorted.filter((p) => {
    if (activeRosterFilter === "sustained" && !p.is_sustained) return false;
    if (activeRosterFilter === "calibrated" && !p.has_calibrated_data) return false;
    if (activeRosterFilter === "transient" && p.is_sustained) return false;
    if (!query) return true;
    return p.tag.toLowerCase().includes(query) || String(p.track_id).includes(query);
  });

  if (rosterCount) rosterCount.textContent = filtered.length.toString();

  filtered.forEach((p) => {
    const item = document.createElement("div");
    item.className = "roster-item";
    item.dataset.trackId = p.track_id;
    if (p.track_id === selectedPlayerId) item.classList.add("selected");

    const statusBadge = p.is_sustained
      ? '<span class="pill-sustained" title="Track duration ≥15 frames">Sustained</span>'
      : '<span class="pill-transient" title="Transient track <15 frames">Transient</span>';
    const calibBadge = p.has_calibrated_data
      ? '<span class="calib-pill" title="Calibrated in 18-yard box">18-yd</span>'
      : '';

    item.innerHTML = `
      <div class="roster-left">
        <span class="player-color-dot" style="background: ${p.color}"></span>
        <div>
          <div class="roster-tag-row">
            <span class="roster-tag">${p.tag}</span>
            ${statusBadge}
          </div>
          <div class="roster-meta">${p.total_samples} frames ${calibBadge}</div>
        </div>
      </div>
      <div class="roster-right">
        <div class="roster-dist">${p.distance_m} m</div>
        <div class="roster-speed">${p.top_speed_kmh ? `${p.top_speed_kmh} km/h` : '—'}</div>
      </div>
    `;

    item.addEventListener("click", () => selectPlayer(p.track_id));
    playerRoster.append(item);
  });
}

// Search and filter chip events
if (playerSearch) {
  playerSearch.addEventListener("input", () => {
    if (currentAnalytics) renderPlayerRoster(currentAnalytics.players || []);
  });
}

filterChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    filterChips.forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    activeRosterFilter = chip.dataset.filter;
    if (currentAnalytics) renderPlayerRoster(currentAnalytics.players || []);
  });
});

// Select a Player Across All Components
function selectPlayer(trackId) {
  selectedPlayerId = Number(trackId);
  if (!currentAnalytics) return;
  const player = currentAnalytics.players.find((p) => p.track_id === selectedPlayerId);
  if (!player) return;

  // 1. Highlight in roster
  document.querySelectorAll(".roster-item").forEach((el) => {
    el.classList.toggle("selected", Number(el.dataset.trackId) === selectedPlayerId);
  });

  // 2. Update Sidebar Selected Player Snapshot
  if (sbTag) sbTag.textContent = player.tag;
  if (sbMeta) sbMeta.textContent = `ID ${player.track_id} · ${player.total_samples} frames`;
  if (sbDist) sbDist.textContent = `${player.distance_m} m`;
  if (sbSpeed) sbSpeed.textContent = player.top_speed_kmh ? `${player.top_speed_kmh} km/h` : "—";
  if (sbDot) sbDot.style.background = player.color;

  // 3. Update Player Detail Tab
  if (pdTag) pdTag.textContent = player.tag;
  if (pdSub) pdSub.textContent = `Tracker ID: ${player.track_id} · Span: ${player.time_span_s || 0}s`;
  if (pdBadge) pdBadge.style.background = player.color;
  if (pdDist) pdDist.textContent = `${player.distance_m} m`;
  if (pdDistKm) pdDistKm.textContent = `${player.distance_km} km`;
  if (pdAvgSpeed) pdAvgSpeed.textContent = player.avg_speed_kmh ? `${player.avg_speed_kmh} km/h` : "—";
  if (pdTopSpeed) pdTopSpeed.textContent = player.top_speed_kmh ? `${player.top_speed_kmh} km/h` : "—";
  if (pdSprints) pdSprints.textContent = player.sprint_count.toString();
  if (pdCoverage) pdCoverage.textContent = `${player.coverage_ratio}%`;
  if (pdSamples) pdSamples.textContent = `${player.valid_samples} calibrated / ${player.total_samples} total`;
  if (pdOcclusions) pdOcclusions.textContent = player.occlusion_gaps.toString();

  // 4. Update Quick Action Handlers
  if (pdJumpVideo) {
    pdJumpVideo.onclick = () => seekVideoToFrame(player.first_frame || 0);
  }
  if (pdViewHeatmap) {
    pdViewHeatmap.onclick = () => {
      document.querySelector('[data-tab="tab-heatmaps"]')?.click();
      showPlayerHeatmap(player.track_id);
    };
  }

  // 5. Update Comparison selector
  if (comparePlayerA && comparePlayerA.value !== String(player.track_id)) {
    comparePlayerA.value = String(player.track_id);
    updateComparison();
  }

  // 6. Redraw canvases
  drawVideoOverlay();
  drawPitchCanvas();
}

function populatePlayerSelectors(players) {
  if (comparePlayerA && comparePlayerB) {
    comparePlayerA.replaceChildren();
    comparePlayerB.replaceChildren();

    players.forEach((p, idx) => {
      const optA = document.createElement("option");
      optA.value = String(p.track_id);
      optA.textContent = `${p.tag} (${p.distance_m}m)`;

      const optB = document.createElement("option");
      optB.value = String(p.track_id);
      optB.textContent = `${p.tag} (${p.distance_m}m)`;

      comparePlayerA.append(optA);
      comparePlayerB.append(optB);
    });

    if (players.length >= 2) {
      comparePlayerA.value = String(players[0].track_id);
      comparePlayerB.value = String(players[1].track_id);
    }
    updateComparison();
  }

  if (hmPlayerSelect) {
    hmPlayerSelect.replaceChildren();
    players.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = String(p.track_id);
      opt.textContent = `${p.tag} (ID ${p.track_id})`;
      hmPlayerSelect.append(opt);
    });
  }
}

// Head-to-Head Comparison
if (comparePlayerA) comparePlayerA.addEventListener("change", updateComparison);
if (comparePlayerB) comparePlayerB.addEventListener("change", updateComparison);

function updateComparison() {
  if (!comparisonBars || !currentAnalytics) return;
  const idA = Number(comparePlayerA.value);
  const idB = Number(comparePlayerB.value);
  const pA = currentAnalytics.players.find((p) => p.track_id === idA);
  const pB = currentAnalytics.players.find((p) => p.track_id === idB);
  if (!pA || !pB) return;

  comparisonBars.replaceChildren();

  const metrics = [
    { label: "Distance Covered", valA: pA.distance_m, valB: pB.distance_m, unit: "m" },
    { label: "Top Sprint Speed", valA: pA.top_speed_kmh || 0, valB: pB.top_speed_kmh || 0, unit: "km/h" },
    { label: "Average Speed", valA: pA.avg_speed_kmh || 0, valB: pB.avg_speed_kmh || 0, unit: "km/h" },
    { label: "Sprint Count", valA: pA.sprint_count, valB: pB.sprint_count, unit: "" },
    { label: "Tracking Coverage", valA: pA.coverage_ratio, valB: pB.coverage_ratio, unit: "%" },
  ];

  metrics.forEach((m) => {
    const maxVal = Math.max(m.valA, m.valB, 1e-4);
    const pctA = Math.round((m.valA / (m.valA + m.valB || 1)) * 100);
    const pctB = 100 - pctA;

    const row = document.createElement("div");
    row.className = "comp-row";
    row.innerHTML = `
      <div class="comp-labels">
        <span style="color: ${pA.color}"><strong>${pA.tag}:</strong> ${m.valA} ${m.unit}</span>
        <span class="comp-metric-name">${m.label}</span>
        <span style="color: ${pB.color}"><strong>${pB.tag}:</strong> ${m.valB} ${m.unit}</span>
      </div>
      <div class="comp-bar-track">
        <div class="comp-bar-a" style="width: ${pctA}%; background: ${pA.color}"></div>
        <div class="comp-bar-b" style="width: ${pctB}%; background: ${pB.color}"></div>
      </div>
    `;
    comparisonBars.append(row);
  });
}

// ============================================================================
// HEATMAP VIEWER
// ============================================================================

function showTeamHeatmap() {
  if (hmSwitchTeam) hmSwitchTeam.classList.add("active");
  if (hmSwitchPlayer) hmSwitchPlayer.classList.remove("active");
  if (hmPlayerSelectWrap) hmPlayerSelectWrap.hidden = true;

  const url = currentAnalytics?.tactical?.team_heatmap_url;
  if (url) {
    heatmapImage.src = url;
    heatmapImage.hidden = false;
    if (heatmapPlaceholder) heatmapPlaceholder.hidden = true;
    if (hmDownloadLink) {
      hmDownloadLink.href = url;
      hmDownloadLink.download = "team_heatmap.png";
    }
  } else {
    heatmapImage.hidden = true;
    if (heatmapPlaceholder) heatmapPlaceholder.hidden = false;
  }
  if (heatmapCaption) heatmapCaption.textContent = "Combined spatial player density across calibrated pitch coordinates.";
}

function showPlayerHeatmap(trackId) {
  if (hmSwitchTeam) hmSwitchTeam.classList.remove("active");
  if (hmSwitchPlayer) hmSwitchPlayer.classList.add("active");
  if (hmPlayerSelectWrap) hmPlayerSelectWrap.hidden = false;
  if (hmPlayerSelect) hmPlayerSelect.value = String(trackId);

  const player = currentAnalytics?.players.find((p) => p.track_id === Number(trackId));
  if (player && player.heatmap_url) {
    heatmapImage.src = player.heatmap_url;
    heatmapImage.hidden = false;
    if (heatmapPlaceholder) heatmapPlaceholder.hidden = true;
    if (hmDownloadLink) {
      hmDownloadLink.href = player.heatmap_url;
      hmDownloadLink.download = `player_${player.track_id}_heatmap.png`;
    }
    if (heatmapCaption) heatmapCaption.textContent = `Spatial position density for ${player.tag} (Track ID ${player.track_id}). Total samples: ${player.total_samples}.`;
  } else {
    heatmapImage.hidden = true;
    if (heatmapPlaceholder) heatmapPlaceholder.hidden = false;
    if (heatmapCaption) heatmapCaption.textContent = "No valid calibrated observations recorded for this player.";
  }
}

if (hmSwitchTeam) hmSwitchTeam.addEventListener("click", showTeamHeatmap);
if (hmSwitchPlayer) {
  hmSwitchPlayer.addEventListener("click", () => {
    if (selectedPlayerId) showPlayerHeatmap(selectedPlayerId);
    else if (currentAnalytics?.players.length > 0) showPlayerHeatmap(currentAnalytics.players[0].track_id);
  });
}
if (hmPlayerSelect) {
  hmPlayerSelect.addEventListener("change", () => showPlayerHeatmap(hmPlayerSelect.value));
}

// ============================================================================
// VIDEO SYNCHRONIZATION & INTERACTIVE CANVAS OVERLAY
// ============================================================================

function resizeOverlayCanvas() {
  if (!videoOverlay || !previewVideo || !videoWrapper) return;
  const vWidth = previewVideo.videoWidth || 1920;
  const vHeight = previewVideo.videoHeight || 1080;
  videoOverlay.width = vWidth;
  videoOverlay.height = vHeight;

  // Accurately position canvas over the video's actual rendered box inside videoWrapper
  const rect = previewVideo.getBoundingClientRect();
  const wrapRect = videoWrapper.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  const videoRatio = vWidth / (vHeight || 1);
  const elemRatio = rect.width / (rect.height || 1);
  let renderW = rect.width;
  let renderH = rect.height;
  let offsetX = 0;
  let offsetY = 0;

  if (elemRatio > videoRatio) {
    renderW = rect.height * videoRatio;
    offsetX = (rect.width - renderW) / 2;
  } else {
    renderH = rect.width / videoRatio;
    offsetY = (rect.height - renderH) / 2;
  }

  videoOverlay.style.left = `${(rect.left - wrapRect.left) + offsetX}px`;
  videoOverlay.style.top = `${(rect.top - wrapRect.top) + offsetY}px`;
  videoOverlay.style.width = `${renderW}px`;
  videoOverlay.style.height = `${renderH}px`;
}

function getCurrentFrameIndex() {
  if (!previewVideo || !currentAnalytics) return 0;
  const fps = currentAnalytics.summary?.fps || 30.0;
  return Math.floor(previewVideo.currentTime * fps);
}

function seekVideoToFrame(frameIndex) {
  if (!previewVideo || !currentAnalytics) return;
  const fps = currentAnalytics.summary?.fps || 30.0;
  previewVideo.currentTime = frameIndex / fps;
  drawVideoOverlay();
}

function formatTime(sec) {
  if (isNaN(sec)) return "00:00.0";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 10);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${ms}`;
}

// Video Playback Controls
if (ctrlPlay && previewVideo) {
  ctrlPlay.addEventListener("click", () => {
    if (previewVideo.paused) previewVideo.play();
    else previewVideo.pause();
  });
}

if (previewVideo) {
  previewVideo.addEventListener("play", () => {
    if (iconPlay) iconPlay.hidden = true;
    if (iconPause) iconPause.hidden = false;
    if (iconPitchPlay) iconPitchPlay.hidden = true;
    if (iconPitchPause) iconPitchPause.hidden = false;
  });
  previewVideo.addEventListener("pause", () => {
    if (iconPlay) iconPlay.hidden = false;
    if (iconPause) iconPause.hidden = true;
    if (iconPitchPlay) iconPitchPlay.hidden = false;
    if (iconPitchPause) iconPitchPause.hidden = true;
  });
  previewVideo.addEventListener("timeupdate", () => {
    if (isSeekingVideo) return;
    const cur = previewVideo.currentTime;
    const dur = previewVideo.duration || 1;
    if (videoScrubber) videoScrubber.value = ((cur / dur) * 100).toFixed(1);
    if (timeCurrent) timeCurrent.textContent = formatTime(cur);
    if (timeTotal) timeTotal.textContent = formatTime(dur);
    const f = getCurrentFrameIndex();
    if (frameCounter) {
      const totalF = currentAnalytics?.summary?.total_frames || Math.floor(dur * 30);
      frameCounter.textContent = `(Frame ${f}/${totalF})`;
    }
    drawVideoOverlay();

    // Synchronize tactical pitch timeline
    if (!isSeekingPitch) {
      tacticalTimelineFrame = f;
      if (pitchScrubber) pitchScrubber.value = f;
      if (pitchFrameLabel) {
        pitchFrameLabel.textContent = `Frame ${f} · ${formatTime(cur)}`;
      }
      drawPitchCanvas();
    }
  });
  previewVideo.addEventListener("loadedmetadata", () => {
    resizeOverlayCanvas();
    if (timeTotal) timeTotal.textContent = formatTime(previewVideo.duration);
  });
  previewVideo.addEventListener("error", (e) => {
    console.warn("previewVideo load error, attempting fallback:", e);
    const annoUrl = currentAnalytics?.summary?.annotated_video_url;
    if (annoUrl && !previewVideo.src.endsWith(annoUrl)) {
      setVideoMode("annotated", true);
      if (videoFallbackBanner) videoFallbackBanner.hidden = false;
    }
  });
}

if (pitchPlayBtn && previewVideo) {
  pitchPlayBtn.addEventListener("click", () => {
    if (previewVideo.paused) previewVideo.play();
    else previewVideo.pause();
  });
}

if (videoScrubber && previewVideo) {
  videoScrubber.addEventListener("input", () => {
    isSeekingVideo = true;
    const pct = parseFloat(videoScrubber.value) / 100;
    const curTime = pct * (previewVideo.duration || 1);
    previewVideo.currentTime = curTime;
    if (timeCurrent) timeCurrent.textContent = formatTime(curTime);
    const f = getCurrentFrameIndex();
    tacticalTimelineFrame = f;
    if (pitchScrubber) pitchScrubber.value = f;
    if (pitchFrameLabel) pitchFrameLabel.textContent = `Frame ${f} · ${formatTime(curTime)}`;
    drawVideoOverlay();
    drawPitchCanvas();
  });
  videoScrubber.addEventListener("change", () => {
    isSeekingVideo = false;
  });
}

if (ctrlSpeed && previewVideo) {
  ctrlSpeed.addEventListener("change", () => {
    previewVideo.playbackRate = parseFloat(ctrlSpeed.value);
  });
}

if (toggleBBoxes) toggleBBoxes.addEventListener("change", drawVideoOverlay);
if (toggleTags) toggleTags.addEventListener("change", drawVideoOverlay);
window.addEventListener("resize", resizeOverlayCanvas);

// Canvas Overlay Drawing
function drawVideoOverlay() {
  if (!videoOverlay || !currentAnalytics || !currentAnalytics.frames) return;
  const ctx = videoOverlay.getContext("2d");
  if (!ctx) return;

  const fIdx = getCurrentFrameIndex();
  ctx.clearRect(0, 0, videoOverlay.width, videoOverlay.height);

  const isAnnotatedMode = currentVideoMode === "annotated";
  const frameData = currentAnalytics.frames[fIdx] || currentAnalytics.frames[String(fIdx)] || [];
  const showBoxes = toggleBBoxes ? toggleBBoxes.checked : true;
  const showTags = toggleTags ? toggleTags.checked : true;

  frameData.forEach((item) => {
    const isSelected = item.id === selectedPlayerId;
    const [x1, y1, x2, y2] = item.bbox;
    const color = item.color || colorForTrack(item.id);

    // In annotated video mode, boxes and tags are pre-rendered into the video stream.
    // Only render the interactive highlight selection box if selected.
    if (isAnnotatedMode) {
      if (isSelected) {
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#f97316";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = "rgba(249, 115, 22, 0.22)";
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }
      return;
    }

    if (showBoxes) {
      ctx.lineWidth = isSelected ? 4 : 2;
      ctx.strokeStyle = isSelected ? "#f97316" : color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      if (isSelected) {
        ctx.fillStyle = "rgba(249, 115, 22, 0.2)";
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }
    }

    if (showTags) {
      const tagText = `${item.tag || tagForTrack(item.id)}`;
      ctx.font = isSelected ? "bold 13px sans-serif" : "bold 11px sans-serif";
      const txtWidth = ctx.measureText(tagText).width;
      const pillH = isSelected ? 22 : 18;
      const pillW = txtWidth + 14;
      const pillX = x1;
      const pillY = Math.max(0, y1 - pillH - 4);

      // Pill Background
      ctx.fillStyle = isSelected ? "#f97316" : "rgba(11, 13, 17, 0.88)";
      ctx.beginPath();
      ctx.roundRect(pillX, pillY, pillW, pillH, 4);
      ctx.fill();

      // Border
      ctx.strokeStyle = isSelected ? "#fff" : color;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Text
      ctx.fillStyle = isSelected ? "#0b0d11" : "#ffffff";
      ctx.fillText(tagText, pillX + 7, pillY + (isSelected ? 16 : 13));
    }
  });
}

// Click Canvas to Select Player
if (videoOverlay) {
  videoOverlay.addEventListener("click", (e) => {
    if (!currentAnalytics || !currentAnalytics.frames) return;
    const rect = videoOverlay.getBoundingClientRect();
    const scaleX = videoOverlay.width / rect.width;
    const scaleY = videoOverlay.height / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    const fIdx = getCurrentFrameIndex();
    const frameData = currentAnalytics.frames[fIdx] || currentAnalytics.frames[String(fIdx)] || [];

    for (let i = frameData.length - 1; i >= 0; i--) {
      const item = frameData[i];
      const [x1, y1, x2, y2] = item.bbox;
      if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
        selectPlayer(item.id);
        break;
      }
    }
  });
}

// ============================================================================
// TACTICAL 2D PITCH CANVAS (FIFA 105m × 68m)
// ============================================================================

function drawPitchCanvas() {
  if (!pitchCanvas) return;
  const ctx = pitchCanvas.getContext("2d");
  if (!ctx) return;

  const w = pitchCanvas.width;
  const h = pitchCanvas.height;

  // FIFA Pitch Dimensions: 105m x 68m
  const PITCH_L = 105.0;
  const PITCH_W = 68.0;

  // Scale Functions: pitch (x, y) -> canvas (cx, cy)
  function toX(x) { return (x / PITCH_L) * w; }
  function toY(y) { return (y / PITCH_W) * h; }

  // 1. Grass Background
  ctx.fillStyle = "#122a1b";
  ctx.fillRect(0, 0, w, h);

  // Subtle Mowed Stripes
  ctx.fillStyle = "rgba(255, 255, 255, 0.02)";
  const stripeW = w / 10;
  for (let i = 0; i < 10; i += 2) {
    ctx.fillRect(i * stripeW, 0, stripeW, h);
  }

  // 2. Highlight 18-Yard Calibrated Penalty Box Zone
  ctx.fillStyle = "rgba(34, 197, 94, 0.14)";
  ctx.fillRect(toX(0), toY(13.84), toX(16.5), toY(54.16 - 13.84));
  ctx.strokeStyle = "rgba(34, 197, 94, 0.5)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(toX(0), toY(13.84), toX(16.5), toY(54.16 - 13.84));

  ctx.fillStyle = "rgba(34, 197, 94, 0.75)";
  ctx.font = "bold 10px monospace";
  ctx.fillText("18-YD CALIBRATED BOX", toX(1.0), toY(16.5));

  // 3. Pitch White Markings
  ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
  ctx.lineWidth = 2;

  // Boundary
  ctx.strokeRect(toX(0), toY(0), toX(PITCH_L), toY(PITCH_W));

  // Halfway Line
  ctx.beginPath();
  ctx.moveTo(toX(PITCH_L / 2), toY(0));
  ctx.lineTo(toX(PITCH_L / 2), toY(PITCH_W));
  ctx.stroke();

  // Center Circle (Radius 9.15m)
  ctx.beginPath();
  ctx.arc(toX(PITCH_L / 2), toY(PITCH_W / 2), toX(9.15), 0, Math.PI * 2);
  ctx.stroke();

  // Center Spot
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(toX(PITCH_L / 2), toY(PITCH_W / 2), 3, 0, Math.PI * 2);
  ctx.fill();

  // Left Penalty Box (16.5m from goal line, width 40.32m: y from 13.84 to 54.16)
  ctx.strokeRect(toX(0), toY(13.84), toX(16.5), toY(40.32));
  // Left Goal Area (5.5m from goal line, width 18.32m: y from 24.84 to 43.16)
  ctx.strokeRect(toX(0), toY(24.84), toX(5.5), toY(18.32));
  // Left Penalty Spot (11m)
  ctx.beginPath();
  ctx.arc(toX(11.0), toY(34.0), 3, 0, Math.PI * 2);
  ctx.fill();

  // Right Penalty Box
  ctx.strokeRect(toX(PITCH_L - 16.5), toY(13.84), toX(16.5), toY(40.32));
  // Right Goal Area
  ctx.strokeRect(toX(PITCH_L - 5.5), toY(24.84), toX(5.5), toY(18.32));
  // Right Penalty Spot
  ctx.beginPath();
  ctx.arc(toX(PITCH_L - 11.0), toY(34.0), 3, 0, Math.PI * 2);
  ctx.fill();

  if (!currentAnalytics) return;

  // PRIORITY 3: Strict Separation of Modes
  if (currentPitchMode === "avg") {
    // ========================================================================
    // MODE: AVERAGE POSITIONS (Aggregated historical mean positions)
    // ========================================================================
    let avgCount = 0;
    currentAnalytics.players.forEach((p) => {
      if (!p.avg_pitch_pos) return;
      const [ax, ay] = p.avg_pitch_pos;
      if (ax < 0.0 || ax > 105.0 || ay < 0.0 || ay > 68.0) return;
      avgCount++;
      const isSelected = p.track_id === selectedPlayerId;
      const cx = toX(ax);
      const cy = toY(ay);

      if (isSelected) {
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(249, 115, 22, 0.35)";
        ctx.fill();
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }

      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? 8 : 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.font = isSelected ? "bold 11px sans-serif" : "10px sans-serif";
      ctx.fillStyle = isSelected ? "#f97316" : "#ffffff";
      ctx.fillText(p.tag, cx + 8, cy + 3);
    });

    if (tacActiveCount) tacActiveCount.textContent = `${avgCount} players (Match Average)`;
    return;
  }

  // ==========================================================================
  // MODE: CURRENT FRAME (Dynamic, Frame-Indexed Tracking Observations)
  // ==========================================================================
  const curF = tacticalTimelineFrame;
  const activeFrameData = currentAnalytics.frames[curF] || currentAnalytics.frames[String(curF)] || [];
  const showAll = pitchAllTrails ? pitchAllTrails.checked : false;

  // 1. Draw dynamic trajectory trails up to the selected frame only
  currentAnalytics.players.forEach((p) => {
    const isSelected = p.track_id === selectedPlayerId;
    if (!isSelected && !showAll) return;
    const rawTraj = p.trajectory || [];
    // Only observations up to selected frame and within valid pitch coordinates
    const traj = rawTraj.filter((pt) => pt.f <= curF && pt.x >= 0.0 && pt.x <= 105.0 && pt.y >= 0.0 && pt.y <= 68.0);
    if (traj.length < 2) return;

    ctx.strokeStyle = isSelected ? p.color : "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = isSelected ? 3 : 1.5;
    ctx.beginPath();
    traj.forEach((pt, idx) => {
      const cx = toX(pt.x);
      const cy = toY(pt.y);
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
  });

  // 2. Draw active player observations for the selected frame
  let onPitchCount = 0;
  let calibCount = 0;
  const frameXs = [];
  const frameYs = [];

  activeFrameData.forEach((item) => {
    const tid = item.tracker_id != null ? item.tracker_id : item.id;
    const px = item.pitch_x != null ? item.pitch_x : (item.pitch ? item.pitch[0] : null);
    const py = item.pitch_y != null ? item.pitch_y : (item.pitch ? item.pitch[1] : null);
    if (px == null || py == null || !isFinite(px) || !isFinite(py)) return;
    // Exclude invalid coordinates outside FIFA pitch boundaries
    if (px < 0.0 || px > 105.0 || py < 0.0 || py > 68.0) return;

    onPitchCount++;
    const isCalib = Boolean(item.calibrated !== undefined ? item.calibrated : item.valid);
    if (isCalib) calibCount++;
    frameXs.push(px);
    frameYs.push(py);

    const isSelected = tid === selectedPlayerId;
    const cx = toX(px);
    const cy = toY(py);
    const color = item.color || colorForTrack(tid);
    const tag = item.player_tag || item.tag || tagForTrack(tid);

    // Selected Player Halo
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(cx, cy, 15, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(249, 115, 22, 0.35)";
      ctx.fill();
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }

    if (isCalib) {
      // Solid calibrated marker
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? 8 : 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = isSelected ? 2 : 1.5;
      ctx.stroke();
    } else {
      // Uncalibrated / outside 18-yard box: distinct hollow/dashed style
      ctx.save();
      ctx.fillStyle = "rgba(148, 163, 184, 0.25)";
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? 8 : 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // Player tag
    ctx.font = isSelected ? "bold 11px sans-serif" : "10px sans-serif";
    ctx.fillStyle = isSelected ? "#f97316" : "#ffffff";
    ctx.fillText(tag, cx + 8, cy + 3);
  });

  if (tacActiveCount) {
    tacActiveCount.textContent = `${onPitchCount} active (${calibCount} calibrated)`;
  }

  // Update tactical spatial metrics for current frame if >=2 players
  if (frameXs.length >= 2) {
    const wSpan = Math.max(...frameYs) - Math.min(...frameYs);
    const lSpan = Math.max(...frameXs) - Math.min(...frameXs);
    const cX = frameXs.reduce((a, b) => a + b, 0) / frameXs.length;
    const cY = frameYs.reduce((a, b) => a + b, 0) / frameYs.length;
    if (tacWidth) tacWidth.textContent = `${wSpan.toFixed(1)} m`;
    if (tacLength) tacLength.textContent = `${lSpan.toFixed(1)} m`;
    if (tacCentroid) tacCentroid.textContent = `(${cX.toFixed(1)}, ${cY.toFixed(1)})`;
  }
}

if (pitchAllTrails) pitchAllTrails.addEventListener("change", drawPitchCanvas);

if (pitchModeCurrent && pitchModeAvg) {
  pitchModeCurrent.addEventListener("click", () => {
    currentPitchMode = "current";
    pitchModeCurrent.classList.add("active");
    pitchModeAvg.classList.remove("active");
    if (pitchTimelineControl) {
      pitchTimelineControl.style.opacity = "1";
      pitchTimelineControl.style.pointerEvents = "auto";
    }
    if (pitchTrailsToggleWrap) pitchTrailsToggleWrap.hidden = false;
    drawPitchCanvas();
  });
  pitchModeAvg.addEventListener("click", () => {
    currentPitchMode = "avg";
    pitchModeAvg.classList.add("active");
    pitchModeCurrent.classList.remove("active");
    if (pitchTimelineControl) {
      pitchTimelineControl.style.opacity = "0.35";
      pitchTimelineControl.style.pointerEvents = "none";
    }
    if (pitchTrailsToggleWrap) pitchTrailsToggleWrap.hidden = true;
    drawPitchCanvas();
  });
}

if (pitchScrubber) {
  pitchScrubber.addEventListener("input", () => {
    isSeekingPitch = true;
    tacticalTimelineFrame = parseInt(pitchScrubber.value, 10);
    const fps = currentAnalytics?.summary?.fps || 30.0;
    const curTime = tacticalTimelineFrame / fps;
    if (previewVideo) {
      previewVideo.currentTime = curTime;
    }
    if (videoScrubber && previewVideo.duration) {
      videoScrubber.value = ((curTime / previewVideo.duration) * 100).toFixed(1);
    }
    if (timeCurrent) timeCurrent.textContent = formatTime(curTime);
    if (pitchFrameLabel) {
      pitchFrameLabel.textContent = `Frame ${tacticalTimelineFrame} · ${formatTime(curTime)}`;
    }
    drawVideoOverlay();
    drawPitchCanvas();
  });
  pitchScrubber.addEventListener("change", () => {
    isSeekingPitch = false;
  });
}

// Export Dropdown Toggle
if (exportBtn && exportMenu) {
  exportBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    exportMenu.hidden = !exportMenu.hidden;
  });
  document.addEventListener("click", () => {
    exportMenu.hidden = true;
  });
}

// Log Viewer Toggle
if (toggleLogBtn && logOutput) {
  toggleLogBtn.addEventListener("click", () => {
    const isHidden = logOutput.hidden;
    logOutput.hidden = !isHidden;
    toggleLogBtn.textContent = isHidden ? "Hide Log" : "Show Log";
  });
}

// Keyboard Shortcut: Spacebar for Play/Pause
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT" && previewVideo) {
    e.preventDefault();
    if (previewVideo.paused) previewVideo.play();
    else previewVideo.pause();
  }
});

// Initialization
refreshDashboard();
showView(location.hash.slice(1));