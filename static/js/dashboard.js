/**
 * SENTINEL-CV | Surveillance Command Center Frontend Logic
 * WebSocket client, Real-time biometrics rendering, Web Audio siren,
 * incident management and evidence viewer.
 */

let socket = null;
let audioAlertsEnabled = true;
let audioCtx = null;
let currentSource = "demo";
let allIncidents = [];

// Initialize Dashboard on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initWebSocket();
});

// =====================================================================
// Real-time Clock
// =====================================================================
function initClock() {
    function updateClock() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
        const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).toUpperCase();
        
        const timeEl = document.getElementById("clock-time");
        const dateEl = document.getElementById("clock-date");
        if (timeEl) timeEl.innerText = timeStr;
        if (dateEl) dateEl.innerText = dateStr;
    }
    updateClock();
    setInterval(updateClock, 1000);
}

// =====================================================================
// WebSocket Client
// =====================================================================
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(`[WS] Connecting to ${wsUrl}...`);
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("[WS] Connected to surveillance stream server.");
        updateSystemBadge("ARMED", "pulse-green");
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        } catch (e) {
            console.error("[WS] Message parsing error:", e);
        }
    };

    socket.onclose = () => {
        console.warn("[WS] Connection lost. Reconnecting in 3s...");
        updateSystemBadge("DISCONNECTED", "pulse-red");
        setTimeout(initWebSocket, 3000);
    };

    socket.onerror = (err) => {
        console.error("[WS] Error encountered:", err);
    };
}

function handleServerMessage(data) {
    if (data.type === "INIT") {
        if (data.incidents) {
            allIncidents = data.incidents;
            renderIncidentTable();
            renderSnapshotGallery();
        }
        if (data.config) {
            applyInitialConfig(data.config);
        }
    } else if (data.type === "TELEMETRY") {
        updateTelemetryView(data.telemetry);
    } else if (data.type === "NEW_ALERT") {
        handleNewIncident(data.incident);
    }
}

function updateSystemBadge(text, dotClass) {
    const label = document.getElementById("status-text");
    const dot = document.querySelector(".status-dot");
    if (label) label.innerText = `SYSTEM ${text}`;
    if (dot) {
        dot.className = `status-dot ${dotClass}`;
    }
}

function applyInitialConfig(cfg) {
    if (cfg.show_skeleton !== undefined) document.getElementById("toggle-skeleton").checked = cfg.show_skeleton;
    if (cfg.show_zones !== undefined) document.getElementById("toggle-zones").checked = cfg.show_zones;
    if (cfg.show_trajectory !== undefined) document.getElementById("toggle-trajectory").checked = cfg.show_trajectory;
    if (cfg.show_hud !== undefined) document.getElementById("toggle-hud").checked = cfg.show_hud;
    if (cfg.detection_conf !== undefined) {
        document.getElementById("conf-slider").value = cfg.detection_conf;
        document.getElementById("conf-val").innerText = Math.round(cfg.detection_conf * 100) + '%';
    }
    if (cfg.source) {
        setSourceActivePill(cfg.source);
    }
}

// =====================================================================
// Telemetry & Biometrics Rendering
// =====================================================================
function updateTelemetryView(t) {
    // 1. Meta Badges
    const fpsBadge = document.getElementById("fps-badge");
    if (fpsBadge) fpsBadge.innerText = `FPS: ${t.fps}`;

    const threatBadge = document.getElementById("threat-global-badge");
    if (threatBadge) {
        threatBadge.innerText = `THREAT: ${t.max_threat}%`;
        if (t.max_threat >= 65) {
            threatBadge.className = "meta-badge danger";
        } else if (t.max_threat >= 35) {
            threatBadge.className = "meta-badge warning";
        } else {
            threatBadge.className = "meta-badge";
        }
    }

    const hudTarget = document.getElementById("hud-target-count");
    if (hudTarget) hudTarget.innerText = `TRACKED TARGETS: ${t.active_targets}`;

    const suspectCount = document.getElementById("active-suspect-count");
    if (suspectCount) suspectCount.innerText = `${t.active_targets} ACTIVE`;

    // 2. Alert Curtain Flash
    const flashEl = document.getElementById("alert-flash");
    if (flashEl) {
        if (t.max_threat >= 65) {
            flashEl.classList.add("active");
        } else {
            flashEl.classList.remove("active");
        }
    }

    // 3. Render Suspect Biometric Cards
    renderSuspectCards(t.persons || []);
}

function renderSuspectCards(persons) {
    const container = document.getElementById("suspect-cards-list");
    if (!container) return;

    if (persons.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i data-lucide="radar" class="spin-icon"></i>
                <p>Scanning camera field for persons...</p>
            </div>
        `;
        if (window.lucide) lucide.createIcons();
        return;
    }

    container.innerHTML = persons.map(p => {
        let cardClass = "";
        let pillClass = "normal";
        if (p.threat_level === "ALERT") {
            cardClass = "alert";
            pillClass = "alert";
        } else if (p.threat_level === "SUSPICIOUS") {
            cardClass = "warning";
            pillClass = "suspicious";
        }

        const pocketFillClass = p.pocket_conf >= 60 ? "fill-red" : (p.pocket_conf >= 35 ? "fill-amber" : "fill-emerald");
        const seqFillClass = p.sequence_conf >= 60 ? "fill-red" : (p.sequence_conf >= 35 ? "fill-amber" : "fill-cyan");
        const chestFillClass = p.chest_conf >= 60 ? "fill-red" : (p.chest_conf >= 35 ? "fill-amber" : "fill-cyan");

        return `
            <div class="suspect-card ${cardClass}">
                <div class="suspect-card-header">
                    <span class="suspect-id-tag">SUSPECT #${p.id}</span>
                    <span class="threat-pill ${pillClass}">
                        ${p.threat_level} (${p.threat_score}%)
                    </span>
                </div>
                
                <!-- Pocket Concealment Dwell Meter -->
                <div class="metric-row">
                    <div class="metric-labels">
                        <span>Pocket Dwell Time</span>
                        <strong>${p.dwell_sec}s (${p.pocket_conf}%)</strong>
                    </div>
                    <div class="metric-bar-bg">
                        <div class="metric-bar-fill ${pocketFillClass}" style="width: ${p.pocket_conf}%;"></div>
                    </div>
                </div>

                <!-- Reach to Pocket Sequence Meter -->
                <div class="metric-row">
                    <div class="metric-labels">
                        <span>Shelf Reach -> Pocket Sequence</span>
                        <strong>${p.sequence_conf}%</strong>
                    </div>
                    <div class="metric-bar-bg">
                        <div class="metric-bar-fill ${seqFillClass}" style="width: ${p.sequence_conf}%;"></div>
                    </div>
                </div>

                <!-- Jacket / Torso Stash Meter -->
                <div class="metric-row">
                    <div class="metric-labels">
                        <span>Jacket / Torso Concealment</span>
                        <strong>${p.chest_conf}%</strong>
                    </div>
                    <div class="metric-bar-bg">
                        <div class="metric-bar-fill ${chestFillClass}" style="width: ${p.chest_conf}%;"></div>
                    </div>
                </div>
            </div>
        `;
    }).join("");
}

// =====================================================================
// Incident Alerts & Evidence Snapshots
// =====================================================================
function handleNewIncident(inc) {
    allIncidents.unshift(inc);
    if (allIncidents.length > 50) allIncidents.pop();

    renderIncidentTable();
    renderSnapshotGallery();

    // Trigger audible alarm if enabled
    if (audioAlertsEnabled) {
        playSecurityAlertSiren();
    }
}

function renderIncidentTable() {
    const tbody = document.getElementById("incident-table-body");
    if (!tbody) return;

    if (allIncidents.length === 0) {
        tbody.innerHTML = `
            <tr class="no-records">
                <td colspan="8">No shoplifting incidents flagged yet. Monitoring active streams.</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = allIncidents.map(inc => {
        const reasonsStr = inc.reasons.join(", ");
        return `
            <tr>
                <td>${inc.timestamp.split(" ")[1]}</td>
                <td>#${inc.person_id}</td>
                <td><span class="badge-risk high">${inc.threat_score}% [${inc.threat_level}]</span></td>
                <td>${inc.pocket_conf}%</td>
                <td>${inc.chest_conf}%</td>
                <td>${inc.sequence_conf}%</td>
                <td style="max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${reasonsStr}</td>
                <td>
                    <button class="btn-view-evidence" onclick='openModal(${JSON.stringify(inc)})'>
                        View Evidence
                    </button>
                </td>
            </tr>
        `;
    }).join("");
}

function renderSnapshotGallery() {
    const container = document.getElementById("snapshot-strip");
    if (!container) return;

    container.innerHTML = allIncidents.map(inc => {
        return `
            <div class="snapshot-card" onclick='openModal(${JSON.stringify(inc)})'>
                <img src="${inc.snapshot_url}" alt="Incident Frame" loading="lazy">
                <div class="snapshot-info">
                    <span>#${inc.person_id}</span>
                    <span class="score">${inc.threat_score}%</span>
                </div>
            </div>
        `;
    }).join("");
}

// =====================================================================
// Synthesized Web Audio Siren
// =====================================================================
function playSecurityAlertSiren() {
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = "sawtooth";
        // Frequency sweep from 880Hz down to 520Hz
        const now = audioCtx.currentTime;
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.exponentialRampToValueAtTime(440, now + 0.35);

        gain.gain.setValueAtTime(0.18, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start(now);
        osc.stop(now + 0.36);
    } catch (e) {
        console.warn("Audio Context playback prevented:", e);
    }
}

function toggleAudioAlerts() {
    audioAlertsEnabled = !audioAlertsEnabled;
    const label = document.getElementById("audio-label");
    const icon = document.getElementById("audio-icon");
    if (label) label.innerText = `Sound Alerts: ${audioAlertsEnabled ? 'ON' : 'OFF'}`;
    if (icon) {
        icon.setAttribute("data-lucide", audioAlertsEnabled ? "volume-2" : "volume-x");
        if (window.lucide) lucide.createIcons();
    }
}

// =====================================================================
// Controls & API Actions
// =====================================================================
function switchVideoSource(source) {
    currentSource = source;
    setSourceActivePill(source);
    
    const formData = new FormData();
    formData.append("source", source);
    fetch("/api/switch_source", {
        method: "POST",
        body: formData
    }).then(res => res.json())
      .then(data => {
          console.log("[Source] Switched to:", data);
          document.getElementById("stream-source-label").innerText = `SRC: ${source.toUpperCase()}`;
          // Reload MJPEG img element cleanly
          const stream = document.getElementById("surveillance-stream");
          if (stream) stream.src = `/video_feed?t=${Date.now()}`;
      });
}

function setSourceActivePill(source) {
    document.querySelectorAll(".source-pill").forEach(el => el.classList.remove("active"));
    if (source === "demo") document.getElementById("btn-src-demo")?.classList.add("active");
    if (source === "webcam") document.getElementById("btn-src-webcam")?.classList.add("active");
    if (source === "file") document.getElementById("btn-src-upload")?.classList.add("active");
}

function handleVideoUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const uploadPill = document.getElementById("btn-src-upload");
    if (uploadPill) uploadPill.querySelector("span").innerText = "Uploading...";

    fetch("/api/upload", {
        method: "POST",
        body: formData
    }).then(res => res.json())
      .then(data => {
          if (uploadPill) {
              uploadPill.querySelector("span").innerText = "Upload Video";
              setSourceActivePill("file");
          }
          document.getElementById("stream-source-label").innerText = `SRC: ${file.name}`;
          const stream = document.getElementById("surveillance-stream");
          if (stream) stream.src = `/video_feed?t=${Date.now()}`;
      }).catch(err => {
          console.error("Upload failed:", err);
          if (uploadPill) uploadPill.querySelector("span").innerText = "Upload Video";
      });
}

function toggleFeature(feature, val) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            action: "TOGGLE_FEATURE",
            feature: feature,
            value: val
        }));
    }
}

function updateConfidence(val) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            action: "SET_CONFIDENCE",
            value: parseFloat(val)
        }));
    }
}

function exportIncidents(format) {
    window.open(`/api/export?format=${format}`, '_blank');
}

// =====================================================================
// Modal Evidence Viewer
// =====================================================================
function openModal(inc) {
    const modal = document.getElementById("evidence-modal");
    const img = document.getElementById("modal-img");
    const title = document.getElementById("modal-title");
    const meta = document.getElementById("modal-meta");
    const details = document.getElementById("modal-details");

    if (!modal) return;

    img.src = inc.snapshot_url;
    title.innerText = `Incident Flag: Suspect #${inc.person_id}`;
    meta.innerText = `${inc.timestamp} | Threat Score: ${inc.threat_score}% [${inc.threat_level}]`;

    details.innerHTML = `
        <div><strong>Primary Flags:</strong> ${inc.reasons.join(", ")}</div>
        <div><strong>Biometric Confidence:</strong> Pocket Dwell: ${inc.pocket_conf}% | Jacket Stash: ${inc.chest_conf}% | Reach Seq: ${inc.sequence_conf}%</div>
        <div><strong>Snapshot ID:</strong> <code>${inc.id}</code></div>
    `;

    modal.classList.add("open");
}

function closeModal() {
    const modal = document.getElementById("evidence-modal");
    if (modal) modal.classList.remove("open");
}
