"use strict";
/* ═══════════════════════════════════════════════════════════════
   SolarScan Dashboard  –  app.js
   ═══════════════════════════════════════════════════════════════ */

// ── Constants ───────────────────────────────────────────────
const SEV_CSS = {
  "No Damage":"sev-none","Minimal":"sev-minimal","Low":"sev-low",
  "Moderate":"sev-moderate","High":"sev-high","Critical":"sev-critical"
};
const CLASS_COLORS = {
  "Bird-drop":"#F97316","Clean":"#10B981","Dusty":"#EAB308",
  "Electrical-damage":"#EF4444","Physical-Damage":"#3B82F6","Snow-Covered":"#06B6D4"
};
const DONUT_COLORS = ["#7C3AED","#10B981","#EAB308","#EF4444","#3B82F6","#06B6D4","#F97316"];

function sevCss(s){ return SEV_CSS[s]||"sev-low"; }
function clsColor(c){ return CLASS_COLORS[c]||"#7C3AED"; }
function dmgColor(p){ return p===0?"#10B981":p<40?"#EAB308":p<70?"#F97316":"#EF4444"; }

// ── Session state ────────────────────────────────────────────
let scanHistory = [];   // [{time, file, cls, conf, damage, severity, diagnosis, img}]
let lastResult  = null;
let selectedFile = null;
let classData   = [];

// ── DOM ──────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Navigation ───────────────────────────────────────────────
document.querySelectorAll(".sb-link[data-page]").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    navigateTo(link.dataset.page);
  });
});

function navigateTo(page) {
  document.querySelectorAll(".sb-link").forEach(l => l.classList.remove("active"));
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  const link = document.querySelector(`.sb-link[data-page="${page}"]`);
  const pg   = $(`page-${page}`);
  if (link) link.classList.add("active");
  if (pg)   pg.classList.add("active");
  if (page === "dashboard") refreshDashboard();
  if (page === "history")   renderHistoryPage();
  if (page === "classes")   renderClassesPage();
}

// ── Health check ─────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${CONFIG.API_BASE}/api/health`);
    const d = await r.json();
    if (d.status === "ok") {
      $("sbDot").className = "sb-status-dot online";
      $("sbStatusText").textContent = d.gpu_name
        ? d.gpu_name.replace("NVIDIA ","").substring(0,22)
        : "CPU Mode";
      $("gpuLabel").textContent = d.gpu_name ? "GPU Ready" : "CPU Mode";
      $("deviceLabel").textContent = d.device.toUpperCase();
      $("settingDevice").textContent = d.device.toUpperCase();
      $("settingGpu").textContent = d.gpu_name || "N/A";
    }
  } catch {
    $("sbDot").className = "sb-status-dot offline";
    $("sbStatusText").textContent = "Backend offline";
    $("gpuLabel").textContent = "Offline";
  }
}

// ── Load classes ─────────────────────────────────────────────
async function loadClasses() {
  try {
    const r = await fetch(`${CONFIG.API_BASE}/api/classes`);
    const d = await r.json();
    classData = d.classes;
  } catch { classData = []; }
}

function renderClassesPage() {
  const grid = $("classesGrid");
  if (!classData.length) {
    grid.innerHTML = '<p style="color:#9CA3AF;text-align:center;padding:40px">Could not load class data.</p>';
    return;
  }
  grid.innerHTML = classData.map(c => `
    <div class="class-card" style="--cls-color:${c.color}">
      <div class="cls-icon-wrap" style="background:${c.color}18">
        <span class="cls-emoji">${c.icon}</span>
      </div>
      <h4 class="cls-name">${c.name}</h4>
      <p class="cls-desc">${c.what_happened}</p>
      <span class="cls-badge" style="background:${c.color}18;color:${c.color}">
        Up to ${c.max_damage}% damage
      </span>
    </div>
  `).join("");
}

// ── File handling ─────────────────────────────────────────────
$("browseBtn").addEventListener("click", () => $("fileInput").click());
$("dropzone").addEventListener("click",  () => $("fileInput").click());
$("fileInput").addEventListener("change", e => { if(e.target.files[0]) handleFile(e.target.files[0]); });

$("dropzone").addEventListener("dragover", e => { e.preventDefault(); $("dropzone").classList.add("drag-over"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag-over"));
$("dropzone").addEventListener("drop", e => {
  e.preventDefault(); $("dropzone").classList.remove("drag-over");
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) { alert("Please upload an image file."); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    $("previewImg").src = e.target.result;
    $("dropzone").style.display    = "none";
    $("previewArea").style.display = "block";
  };
  reader.readAsDataURL(file);
}

$("removeBtn").addEventListener("click", resetUpload);
function resetUpload() {
  selectedFile = null;
  $("fileInput").value = "";
  $("previewImg").src  = "";
  $("dropzone").style.display    = "block";
  $("previewArea").style.display = "none";
}

$("confSlider").addEventListener("input", () => { $("confVal").textContent = $("confSlider").value + "%"; });
$("globalConf").addEventListener("input", () => { $("globalConfVal").textContent = $("globalConf").value + "%"; });

// ── Analyse ───────────────────────────────────────────────────
$("analyseBtn").addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;
  $("analyseBtn").disabled = true;
  $("loadingOverlay").style.display = "flex";

  const form = new FormData();
  form.append("image", selectedFile);
  form.append("conf", (parseInt($("confSlider").value)/100).toFixed(2));

  try {
    const res  = await fetch(`${CONFIG.API_BASE}/api/predict`, { method:"POST", body:form });
    const data = await res.json();

    // ── Not a solar panel ─────────────────────────────────────────────
    if (res.status === 422 && data.error === "not_solar_panel") {
      showNotPanelError(data.message, data.reason, data.suggestion);
      return;
    }

    if (!res.ok || data.error) { alert("Error: " + (data.error||"Unknown")); return; }

    lastResult = { ...data, filename: selectedFile.name };
    addToHistory(data, selectedFile.name);
    renderResults(data);
    navigateTo("results");
    $("notifBadge").style.display = "flex";

  } catch(err) {
    if (typeof addNotification === "function")
      addNotification("error", "Request failed: " + err.message);
    console.error("Request failed:", err.message);
  } finally {
    $("loadingOverlay").style.display = "none";
    $("analyseBtn").disabled = false;
  }
}

// ── History ───────────────────────────────────────────────────
function addToHistory(data, filename) {
  const entry = {
    time:     new Date().toLocaleTimeString(),
    file:     filename,
    cls:      data.primary_class,
    conf:     data.detections[0]?.confidence || 0,
    damage:   data.avg_damage,
    severity: data.severity,
    color:    data.primary_color,
    img:      data.annotated_image,
    diagnosis:data.diagnosis,
  };
  scanHistory.unshift(entry);
  $("sessionCount").textContent = scanHistory.length;
  renderHistoryTable($("historyTbody"));
  renderHistoryTable($("historyTbody2"));
  refreshDashboard();
}

function renderHistoryTable(tbody) {
  if (!scanHistory.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="6">No scans yet — upload a panel image to get started</td></tr>';
    return;
  }
  tbody.innerHTML = scanHistory.map((e, idx) => `
    <tr class="history-row" data-idx="${idx}" title="Click to view full details">
      <td style="color:var(--text-3);font-size:.78rem">${e.time}</td>
      <td style="font-weight:600;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.file}</td>
      <td><span class="cls-pill" style="background:${e.color}18;color:${e.color}">${e.cls}</span></td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-bar-track"><div class="conf-bar-fill" style="width:${e.conf}%;background:${e.color}"></div></div>
          <span style="color:${e.color};font-weight:700;min-width:38px;font-size:.78rem">${e.conf}%</span>
        </div>
      </td>
      <td style="font-weight:700;color:${dmgColor(e.damage)}">${e.damage}%</td>
      <td><span class="sev-badge ${sevCss(e.severity)}">${e.severity}</span></td>
    </tr>
  `).join("");

  // Make every row clickable
  tbody.querySelectorAll(".history-row").forEach(row => {
    row.addEventListener("click", () => openHistoryDetail(parseInt(row.dataset.idx)));
  });
}

function renderHistoryPage() {
  renderHistoryTable($("historyTbody2"));
}

// ── Render results ────────────────────────────────────────────
function renderResults(data) {
  const { annotated_image, primary_class, primary_color,
          avg_damage, severity, detection_count, detections, diagnosis } = data;

  // Show/hide
  $("noResults").style.display     = "none";
  $("resultsContent").style.display = "block";

  // Summary
  $("rClass").textContent  = primary_class;
  $("rClass").style.color  = primary_color;
  $("rDamage").textContent = avg_damage + "%";
  $("rDamage").style.color = dmgColor(avg_damage);
  $("rCount").textContent  = detection_count;
  const sevEl = $("rSeverity");
  sevEl.textContent = severity;
  sevEl.className   = "sev-badge " + sevCss(severity);

  // Meter
  const pct = Math.min(Math.max(avg_damage,0),100);
  $("meterThumb").style.left = pct + "%";
  $("meterPct").textContent  = avg_damage + "%";
  $("meterPct").style.color  = dmgColor(avg_damage);

  // Image
  const src = "data:image/jpeg;base64," + annotated_image;
  $("resultImg").src = src;

  // Download button → generates full PDF report
  const dlBtn = $("downloadBtn");
  dlBtn.onclick = async function(e) {
    e.preventDefault();
    dlBtn.textContent = "⏳ Generating PDF…";
    dlBtn.style.pointerEvents = "none";
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/report`, {
        method:      "POST",
        headers:     { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          annotated_image,
          primary_class,
          primary_color,
          avg_damage,
          severity,
          detections,
          diagnosis,
          filename: document.getElementById("resultFileName")?.textContent || "panel.jpg",
        }),
      });
      if (!res.ok) { alert("Failed to generate report."); return; }
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `SolarScan_${primary_class}_${avg_damage}pct.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch(err) {
      alert("Report generation failed: " + err.message);
    } finally {
      dlBtn.innerHTML = '↓ Download Report';
      dlBtn.style.pointerEvents = "";
    }
  };

  // Diagnosis
  $("diagWhat").textContent = diagnosis.what_happened;
  $("diagImpact").innerHTML = diagnosis.impact.map(i=>`<li>${i}</li>`).join("");
  $("diagSteps").innerHTML  = diagnosis.suggestions.map((s,i)=>
    `<li><span class="step-badge">${i+1}</span><span>${s.replace(/^\d+\.\s*/,"")}</span></li>`
  ).join("");

  // Table
  $("detTbody").innerHTML = detections.map((d,i)=>`
    <tr>
      <td style="color:var(--text-3);font-weight:600">${i+1}</td>
      <td>
        ${d.grid_label
          ? `<span class="grid-badge" style="background:${d.color}18;color:${d.color};border:1px solid ${d.color}44">${d.grid_label}</span>`
          : `<span style="color:var(--text-3)">—</span>`}
      </td>
      <td><span class="cls-pill" style="background:${d.color}18;color:${d.color}">${d.class}</span></td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-bar-track"><div class="conf-bar-fill" style="width:${d.confidence}%;background:${d.color}"></div></div>
          <span style="color:${d.color};font-weight:700;min-width:42px">${d.confidence}%</span>
        </div>
      </td>
      <td style="font-weight:700;color:${dmgColor(d.damage_pct)}">${d.damage_pct}%</td>
      <td><span class="sev-badge ${sevCss(d.severity)}">${d.severity}</span></td>
      <td style="font-size:.75rem;color:var(--text-3);font-family:monospace">[${d.bbox.join(", ")}]</td>
    </tr>
  `).join("");

  // ── Farm grid map ─────────────────────────────────────────────────────
  const farmCard = $("farmGridCard");
  if (data.farm_mode && data.total_panels > 1) {
    farmCard.style.display = "block";
    $("farmStats").textContent =
      `${data.total_panels} panels detected · ${data.affected_panels} affected`;
    $("gridMapImg").src = "data:image/png;base64," + data.grid_map_b64;

    // Panel crops
    const cropsGrid = $("panelCropsGrid");
    if (data.panel_crops && data.panel_crops.length) {
      cropsGrid.innerHTML = data.panel_crops.map(pc => `
        <div class="panel-crop-card" style="border-top:3px solid ${pc.color}">
          <div class="panel-crop-header">
            <span class="grid-badge" style="background:${pc.color}18;color:${pc.color};border:1px solid ${pc.color}44">${pc.grid_label}</span>
            <span class="cls-pill" style="background:${pc.color}18;color:${pc.color};font-size:.7rem">${pc.class}</span>
          </div>
          <img src="data:image/jpeg;base64,${pc.crop_b64}" class="panel-crop-img" alt="Panel ${pc.grid_label}"/>
          <div class="panel-crop-footer">
            <span style="font-weight:700;color:${dmgColor(pc.damage_pct)}">${pc.damage_pct}%</span>
            <span class="sev-badge ${sevCss(pc.severity)}" style="font-size:.65rem">${pc.severity}</span>
          </div>
        </div>
      `).join("");
    } else {
      cropsGrid.innerHTML = '<p style="color:var(--text-3);font-size:.85rem">No defective panels found</p>';
    }
  } else {
    farmCard.style.display = "none";
  }
}

// ── History detail modal ──────────────────────────────────────
function openHistoryDetail(idx) {
  const e = scanHistory[idx];
  if (!e) return;

  // Build modal HTML
  const impactHtml = e.diagnosis.impact
    .map(i => `<li>${i}</li>`).join("");
  const stepsHtml = e.diagnosis.suggestions
    .map((s, i) => `<li><span class="step-badge">${i+1}</span><span>${s.replace(/^\d+\.\s*/,"")}</span></li>`)
    .join("");
  const imgHtml = e.img
    ? `<img src="data:image/jpeg;base64,${e.img}" class="modal-result-img" alt="Annotated"/>`
    : `<div class="modal-no-img">No image available</div>`;

  const pct = Math.min(Math.max(e.damage, 0), 100);
  const meterFill = `width:${pct}%`;

  const modal = document.createElement("div");
  modal.className = "hd-overlay";
  modal.id = "historyDetailModal";
  modal.innerHTML = `
    <div class="hd-modal">

      <!-- Header -->
      <div class="hd-header" style="border-left:4px solid ${e.color}">
        <div class="hd-header-left">
          <span class="cls-pill" style="background:${e.color}18;color:${e.color};font-size:.85rem;padding:4px 12px">${e.cls}</span>
          <span class="hd-filename">${e.file}</span>
          <span style="color:var(--text-3);font-size:.78rem">${e.time}</span>
        </div>
        <button class="hd-close" id="hdClose">✕</button>
      </div>

      <!-- Summary strip -->
      <div class="hd-summary">
        <div class="hd-sum-item">
          <span class="hd-sum-label">Damage</span>
          <span class="hd-sum-val" style="color:${dmgColor(e.damage)}">${e.damage}%</span>
        </div>
        <div class="hd-sum-sep"></div>
        <div class="hd-sum-item">
          <span class="hd-sum-label">Severity</span>
          <span class="sev-badge ${sevCss(e.severity)}">${e.severity}</span>
        </div>
        <div class="hd-sum-sep"></div>
        <div class="hd-sum-item">
          <span class="hd-sum-label">Confidence</span>
          <span class="hd-sum-val" style="color:${e.color}">${e.conf}%</span>
        </div>
        <div class="hd-sum-sep"></div>
        <div class="hd-sum-item">
          <span class="hd-sum-label">Scanned</span>
          <span class="hd-sum-val">${e.time}</span>
        </div>
      </div>

      <!-- Damage meter -->
      <div class="hd-meter-wrap">
        <div class="hd-meter-labels">
          <span class="card-title">Damage Severity</span>
          <span style="font-weight:800;color:${dmgColor(e.damage)}">${e.damage}%</span>
        </div>
        <div class="meter-track" style="margin:8px 0">
          <div class="meter-gradient"></div>
          <div class="meter-thumb" style="left:${pct}%"></div>
        </div>
        <div class="meter-labels">
          <span class="ml green">No Damage</span>
          <span class="ml yellow">Low</span>
          <span class="ml orange">Moderate</span>
          <span class="ml red">Critical</span>
        </div>
      </div>

      <!-- Body: image + diagnosis -->
      <div class="hd-body">

        <!-- Annotated image -->
        <div class="hd-img-col">
          <div class="hd-section-title">Annotated Image</div>
          ${imgHtml}
          ${e.img ? `<a href="data:image/jpeg;base64,${e.img}" download="solar_${e.cls}_${e.damage}pct.jpg" class="btn-accent sm" style="margin-top:10px;display:inline-flex;align-items:center;gap:6px">↓ Download</a>` : ""}
        </div>

        <!-- Diagnosis -->
        <div class="hd-diag-col">
          <div class="hd-section-title">Diagnosis Report</div>

          <div class="diag-block">
            <div class="diag-block-title"><span class="diag-dot orange-dot"></span>What Happened</div>
            <p class="diag-text">${e.diagnosis.what_happened}</p>
          </div>

          <div class="diag-block" style="margin-top:14px">
            <div class="diag-block-title"><span class="diag-dot red-dot"></span>Impact on Panel</div>
            <ul class="impact-list">${impactHtml}</ul>
          </div>

          <div class="diag-block suggest-block" style="margin-top:14px">
            <div class="diag-block-title"><span class="diag-dot green-dot"></span>How to Improve</div>
            <ol class="steps-list">${stepsHtml}</ol>
          </div>
        </div>

      </div>

      <!-- Footer -->
      <div class="hd-footer">
        <button class="btn-accent sm" id="hdViewResults">View in Results Page</button>
        <button class="pill-btn" id="hdCloseBtn">Close</button>
      </div>

    </div>
  `;

  document.body.appendChild(modal);

  // Close handlers
  const close = () => modal.remove();
  document.getElementById("hdClose").addEventListener("click", close);
  document.getElementById("hdCloseBtn").addEventListener("click", close);
  modal.addEventListener("click", ev => { if (ev.target === modal) close(); });

  // View in Results page
  document.getElementById("hdViewResults").addEventListener("click", () => {
    close();
    // Reconstruct a result-like object from history entry
    renderResults({
      annotated_image:  e.img || "",
      primary_class:    e.cls,
      primary_color:    e.color,
      avg_damage:       e.damage,
      severity:         e.severity,
      detection_count:  1,
      detections: [{
        class:      e.cls,
        confidence: e.conf,
        damage_pct: e.damage,
        severity:   e.severity,
        color:      e.color,
        bbox:       [0, 0, 0, 0],
      }],
      diagnosis: e.diagnosis,
    });
    navigateTo("results");
  });
}

// ── Dashboard stats & charts ──────────────────────────────────
function refreshDashboard() {
  const total    = scanHistory.length;
  const defects  = scanHistory.filter(e => e.cls !== "Clean").length;
  const avgDmg   = total ? Math.round(scanHistory.reduce((s,e)=>s+e.damage,0)/total) : 0;
  const critical = scanHistory.filter(e => e.severity === "Critical" || e.severity === "High").length;

  $("statTotal").textContent         = total;
  $("statTotalChange").textContent   = total ? `${total} panel${total>1?"s":""} scanned` : "— panels";
  $("statDefects").textContent       = defects;
  $("statDefectsChange").textContent = defects ? `${defects} defect${defects>1?"s":""} found` : "— this session";
  $("statAvgDmg").textContent        = avgDmg + "%";
  $("statAvgChange").textContent     = total ? "across all scans" : "— across scans";
  $("statCritical").textContent      = critical;
  $("statCriticalChange").textContent= critical ? `${critical} need attention` : "— need attention";

  // Colour stat bigs
  $("statDefects").style.color  = defects  ? "#EF4444" : "var(--text)";
  $("statCritical").style.color = critical ? "#EF4444" : "var(--text)";
  $("statAvgDmg").style.color   = dmgColor(avgDmg);

  renderHistoryTable($("historyTbody"));
  drawBarChart();
  drawDonutChart();
}

// ── Bar chart (canvas) ────────────────────────────────────────
function drawBarChart() {
  const canvas = $("barChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.parentElement.clientWidth - 40;
  canvas.width  = W;
  canvas.height = 180;
  ctx.clearRect(0, 0, W, 180);

  const classes = ["Bird-drop","Clean","Dusty","Electrical-damage","Physical-Damage","Snow-Covered"];
  const counts  = classes.map(c => scanHistory.filter(e=>e.cls===c).length);
  const avgs    = classes.map(c => {
    const items = scanHistory.filter(e=>e.cls===c);
    return items.length ? Math.round(items.reduce((s,e)=>s+e.damage,0)/items.length) : 0;
  });
  const maxVal  = Math.max(...counts, 1);
  const labels  = ["Bird","Clean","Dusty","Elec","Phys","Snow"];

  const padL=30, padR=10, padT=10, padB=40;
  const chartW = W - padL - padR;
  const chartH = 180 - padT - padB;
  const groupW = chartW / classes.length;
  const barW   = Math.min(groupW * 0.35, 22);

  // Grid lines
  ctx.strokeStyle = "#EDE9FE"; ctx.lineWidth = 1;
  for (let i=0; i<=4; i++) {
    const y = padT + chartH - (i/4)*chartH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W-padR, y); ctx.stroke();
    ctx.fillStyle="#9CA3AF"; ctx.font="10px Inter";
    ctx.fillText(Math.round((i/4)*maxVal), 2, y+4);
  }

  classes.forEach((cls, i) => {
    const x = padL + i*groupW + groupW/2;
    const color = clsColor(cls);

    // Count bar
    const h1 = counts[i] ? (counts[i]/maxVal)*chartH : 0;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x - barW - 2, padT + chartH - h1, barW, h1, [4,4,0,0]);
    ctx.fill();

    // Avg% bar (lighter)
    const h2 = avgs[i] ? (avgs[i]/100)*chartH : 0;
    ctx.fillStyle = color + "55";
    ctx.beginPath();
    ctx.roundRect(x + 2, padT + chartH - h2, barW, h2, [4,4,0,0]);
    ctx.fill();

    // Label
    ctx.fillStyle = "#6B7280"; ctx.font = "10px Inter"; ctx.textAlign = "center";
    ctx.fillText(labels[i], x, padT + chartH + 16);

    // Value on top
    if (counts[i]) {
      ctx.fillStyle = color; ctx.font = "bold 10px Inter";
      ctx.fillText(counts[i], x - barW/2 - 2, padT + chartH - h1 - 4);
    }
  });
  ctx.textAlign = "left";
}

// ── Donut chart (canvas) ──────────────────────────────────────
function drawDonutChart() {
  const canvas = $("donutChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, 160, 160);

  const sevGroups = ["No Damage","Minimal","Low","Moderate","High","Critical"];
  const counts    = sevGroups.map(s => scanHistory.filter(e=>e.severity===s).length);
  const total     = counts.reduce((a,b)=>a+b, 0);
  $("donutTotal").textContent = total;

  if (!total) {
    // Empty state ring
    ctx.beginPath();
    ctx.arc(80, 80, 60, 0, Math.PI*2);
    ctx.strokeStyle = "#EDE9FE"; ctx.lineWidth = 18;
    ctx.stroke();
    return;
  }

  let startAngle = -Math.PI/2;
  const legend = $("donutLegend");
  legend.innerHTML = "";

  sevGroups.forEach((sev, i) => {
    if (!counts[i]) return;
    const slice = (counts[i]/total) * Math.PI*2;
    ctx.beginPath();
    ctx.moveTo(80, 80);
    ctx.arc(80, 80, 60, startAngle, startAngle + slice);
    ctx.closePath();
    ctx.fillStyle = DONUT_COLORS[i];
    ctx.fill();

    // Inner hole
    ctx.beginPath();
    ctx.arc(80, 80, 42, 0, Math.PI*2);
    ctx.fillStyle = "#fff";
    ctx.fill();

    const li = document.createElement("li");
    li.innerHTML = `
      <span class="dl-dot" style="background:${DONUT_COLORS[i]}"></span>
      <span>${sev}</span>
      <span class="dl-val">${counts[i]}</span>
    `;
    legend.appendChild(li);
    startAngle += slice;
  });
}

// ── Not-a-panel error modal ───────────────────────────────────
function showNotPanelError(message, reason, suggestion) {
  // Remove any existing modal
  const existing = document.getElementById("notPanelModal");
  if (existing) existing.remove();

  const modal = document.createElement("div");
  modal.id = "notPanelModal";
  modal.style.cssText = `
    position:fixed;inset:0;z-index:400;
    background:rgba(30,27,75,.5);backdrop-filter:blur(6px);
    display:flex;align-items:center;justify-content:center;padding:20px;
  `;
  modal.innerHTML = `
    <div style="background:#fff;border-radius:20px;max-width:480px;width:100%;
                box-shadow:0 24px 80px rgba(0,0,0,.22);overflow:hidden;">
      <!-- Red header -->
      <div style="background:#EF4444;padding:20px 24px;display:flex;align-items:center;gap:12px;">
        <div style="width:44px;height:44px;background:rgba(255,255,255,.2);border-radius:50%;
                    display:flex;align-items:center;justify-content:center;font-size:22px;">⚠️</div>
        <div>
          <div style="color:#fff;font-weight:800;font-size:1.1rem;">Not a Solar Panel Image</div>
          <div style="color:#FEE2E2;font-size:.82rem;">Upload rejected</div>
        </div>
      </div>
      <!-- Body -->
      <div style="padding:24px;">
        <p style="font-weight:700;color:#1E1B4B;margin-bottom:8px;font-size:.95rem;">${message}</p>
        <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;
                    padding:12px 14px;margin-bottom:16px;">
          <div style="font-size:.75rem;font-weight:700;color:#991B1B;text-transform:uppercase;
                      letter-spacing:.05em;margin-bottom:4px;">Detection Reason</div>
          <p style="font-size:.85rem;color:#7F1D1D;">${reason || "Image does not match solar panel characteristics."}</p>
        </div>
        <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;
                    padding:12px 14px;margin-bottom:20px;">
          <div style="font-size:.75rem;font-weight:700;color:#065F46;text-transform:uppercase;
                      letter-spacing:.05em;margin-bottom:4px;">💡 What to Upload</div>
          <p style="font-size:.85rem;color:#064E3B;">${suggestion || "Please upload a solar panel photograph."}</p>
          <ul style="font-size:.82rem;color:#065F46;margin-top:8px;padding-left:16px;line-height:1.8;">
            <li>Close-up of a single solar panel</li>
            <li>Aerial / drone view of a solar farm</li>
            <li>Panel showing dust, cracks, bird droppings, or snow</li>
          </ul>
        </div>
        <div style="display:flex;gap:10px;">
          <button onclick="document.getElementById('notPanelModal').remove();resetUpload();"
                  style="flex:1;padding:11px;background:#7C3AED;color:#fff;border:none;
                         border-radius:10px;font-weight:700;cursor:pointer;font-size:.9rem;">
            Try Another Image
          </button>
          <button onclick="document.getElementById('notPanelModal').remove();"
                  style="padding:11px 18px;background:#F3F4F6;color:#374151;border:none;
                         border-radius:10px;font-weight:600;cursor:pointer;font-size:.9rem;">
            Cancel
          </button>
        </div>
      </div>
    </div>
  `;
  // Close on backdrop click
  modal.addEventListener("click", e => { if (e.target === modal) modal.remove(); });
  document.body.appendChild(modal);
}


// ── Misc buttons ──────────────────────────────────────────────
$("goAnalyseBtn").addEventListener("click",  () => navigateTo("analyse"));
$("goAnalyseBtn2").addEventListener("click", () => navigateTo("analyse"));
$("newScanBtn").addEventListener("click", () => {
  resetUpload();
  navigateTo("analyse");
  $("notifBadge").style.display = "none";
});
$("clearHistoryBtn").addEventListener("click", () => {
  scanHistory = [];
  lastResult  = null;
  $("sessionCount").textContent = "0";
  refreshDashboard();
});

// ── Stat card arrows ───────────────────────────────────────────
// Total Scanned → History
$("arrowTotal").addEventListener("click", () => navigateTo("history"));

// Make the whole stat card clickable too
document.querySelectorAll(".stat-card").forEach((card, i) => {
  card.addEventListener("click", (e) => {
    // Don't double-fire if arrow button was clicked
    if (e.target.classList.contains("stat-arrow")) return;
    const pages = ["history", "history", "results", "history"];
    if (i === 2 && lastResult) navigateTo("results");
    else navigateTo("history");
  });
});

// Defects Found → History (filtered view — just navigate for now)
$("arrowDefects").addEventListener("click", () => {
  navigateTo("history");
  // Highlight defect rows after render
  setTimeout(() => {
    document.querySelectorAll(".history-row").forEach(row => {
      const clsCell = row.querySelector(".cls-pill");
      if (clsCell && clsCell.textContent.trim() !== "Clean") {
        row.style.background = "var(--purple-lt)";
        setTimeout(() => row.style.background = "", 1800);
      }
    });
  }, 100);
});

// Avg Damage → Results (show last result) or History
$("arrowAvgDmg").addEventListener("click", () => {
  if (lastResult) {
    navigateTo("results");
  } else {
    navigateTo("history");
  }
});

// Critical Panels → History (highlight critical rows)
$("arrowCritical").addEventListener("click", () => {
  navigateTo("history");
  setTimeout(() => {
    document.querySelectorAll(".history-row").forEach(row => {
      const sevCell = row.querySelector(".sev-badge");
      if (sevCell && (sevCell.textContent === "Critical" || sevCell.textContent === "High")) {
        row.style.background = "#FEE2E2";
        setTimeout(() => row.style.background = "", 1800);
      }
    });
  }, 100);
});

// Donut chart arrow → History
$("arrowDonut") && $("arrowDonut").addEventListener("click", () => navigateTo("history"));

// ── Init ──────────────────────────────────────────────────────
(async function init() {
  // Check auth first — redirect to login if not authenticated
  try {
    const r = await fetch(`${CONFIG.API_BASE}/api/auth/me`, { credentials: "include" });
    if (r.status === 401) { window.location.href = "/login.html"; return; }
    const user = await r.json();
    document.querySelector(".user-name").textContent = user.name || "SolarScan";
    document.querySelector(".user-role").textContent = user.email || "";
    // Show Google avatar if available
    if (user.avatar) {
      const avatarEl = document.querySelector(".user-avatar");
      avatarEl.innerHTML = `<img src="${user.avatar}" alt="${user.name}" style="width:36px;height:36px;border-radius:50%;object-fit:cover"/>`;
    }
  } catch {
    // Backend might be loading — continue anyway
  }

  await Promise.all([checkHealth(), loadClasses()]);
  refreshDashboard();
  window.addEventListener("resize", () => {
    if (document.getElementById("page-dashboard").classList.contains("active")) drawBarChart();
  });

  // Logout
  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      await fetch(`${CONFIG.API_BASE}/api/auth/logout`, { method: "POST", credentials: "include" });
      window.location.href = "/login.html";
    });
  }
})();


// ══════════════════════════════════════════════════════════════
//  SEARCH SYSTEM
// ══════════════════════════════════════════════════════════════

const SEARCH_INDEX = [
  // Pages
  { type:"page", icon:"📊", title:"Dashboard",      sub:"Stats, charts, recent scans",    page:"dashboard",  tags:["home","overview","stats","charts"] },
  { type:"page", icon:"🔍", title:"Analyse Panel",  sub:"Upload image for detection",     page:"analyse",    tags:["upload","detect","analyse","scan","image"] },
  { type:"page", icon:"📋", title:"Results",        sub:"Latest detection results",       page:"results",    tags:["results","output","detection","damage"] },
  { type:"page", icon:"🕐", title:"History",        sub:"All past scans",                 page:"history",    tags:["history","past","scans","log"] },
  { type:"page", icon:"🏷️", title:"Defect Classes", sub:"6 solar panel defect types",    page:"classes",    tags:["classes","defects","types","categories"] },
  { type:"page", icon:"⚙️", title:"Settings",       sub:"Model and detection config",     page:"settings",   tags:["settings","config","model","gpu"] },
  // Defect classes
  { type:"class", icon:"🐦", title:"Bird-drop",         sub:"Bird droppings on panel surface",  color:"#F97316", tags:["bird","droppings","organic"] },
  { type:"class", icon:"✅", title:"Clean",              sub:"No defects detected",              color:"#10B981", tags:["clean","ok","good","healthy"] },
  { type:"class", icon:"🌫️", title:"Dusty",             sub:"Dust accumulation on surface",     color:"#EAB308", tags:["dust","dirty","accumulation"] },
  { type:"class", icon:"⚡", title:"Electrical-damage", sub:"Burn marks or arc damage",         color:"#EF4444", tags:["electrical","burn","arc","fire"] },
  { type:"class", icon:"💥", title:"Physical-Damage",   sub:"Cracks, chips or impact marks",    color:"#3B82F6", tags:["physical","crack","chip","broken"] },
  { type:"class", icon:"❄️", title:"Snow-Covered",      sub:"Snow or ice accumulation",         color:"#06B6D4", tags:["snow","ice","covered","winter"] },
  // Actions
  { type:"action", icon:"⬆️", title:"Upload Image",     sub:"Analyse a new solar panel",        action:()=>{ closeSearch(); navigateTo("analyse"); } },
  { type:"action", icon:"↺",  title:"New Scan",         sub:"Start a fresh analysis",           action:()=>{ closeSearch(); resetUpload(); navigateTo("analyse"); } },
  { type:"action", icon:"↓",  title:"Download Report",  sub:"Download last PDF report",         action:()=>{ closeSearch(); const b=$("downloadBtn"); if(b&&b.onclick) b.onclick(new Event("click")); } },
  { type:"action", icon:"🚪", title:"Log Out",          sub:"Sign out of SolarScan",            action:async()=>{ closeSearch(); await fetch(`${CONFIG.API_BASE}/api/auth/logout`,{method:"POST",credentials:"include"}); window.location.href="/login.html"; } },
];

let searchActive = false;
let searchIdx    = -1;
let searchItems  = [];

function openSearch() {
  searchActive = true;
  $("searchOverlay").style.display = "flex";
  $("searchInput").value = "";
  $("searchInput").focus();
  renderSearchResults("");
}

function closeSearch() {
  searchActive = false;
  $("searchOverlay").style.display = "none";
  searchIdx = -1;
}

function renderSearchResults(query) {
  const q = query.toLowerCase().trim();
  const container = $("searchResults");

  if (!q) {
    // Show quick links when empty
    container.innerHTML = `
      <div class="search-group-label">Quick Navigation</div>
      ${SEARCH_INDEX.filter(i=>i.type==="page").map((item,i)=>`
        <div class="search-item" data-idx="${i}" onclick="handleSearchSelect(${SEARCH_INDEX.indexOf(item)})">
          <div class="search-item-icon" style="background:var(--purple-lt)">${item.icon}</div>
          <div class="search-item-text">
            <div class="search-item-title">${item.title}</div>
            <div class="search-item-sub">${item.sub}</div>
          </div>
          <span class="search-item-badge">Page</span>
        </div>
      `).join("")}
      <div class="search-group-label">Actions</div>
      ${SEARCH_INDEX.filter(i=>i.type==="action").map((item,i)=>`
        <div class="search-item" onclick="handleSearchSelect(${SEARCH_INDEX.indexOf(item)})">
          <div class="search-item-icon" style="background:var(--bg)">${item.icon}</div>
          <div class="search-item-text">
            <div class="search-item-title">${item.title}</div>
            <div class="search-item-sub">${item.sub}</div>
          </div>
        </div>
      `).join("")}
    `;
    searchItems = [];
    return;
  }

  // Filter by query
  const results = SEARCH_INDEX.filter(item => {
    const haystack = [item.title, item.sub, ...(item.tags||[])].join(" ").toLowerCase();
    return haystack.includes(q);
  });

  searchItems = results;
  searchIdx   = results.length ? 0 : -1;

  if (!results.length) {
    container.innerHTML = `<div class="search-no-results">No results for "<strong>${query}</strong>"</div>`;
    return;
  }

  // Group by type
  const groups = { page:"Pages", class:"Defect Classes", action:"Actions" };
  let html = "";
  for (const [type, label] of Object.entries(groups)) {
    const group = results.filter(r=>r.type===type);
    if (!group.length) continue;
    html += `<div class="search-group-label">${label}</div>`;
    group.forEach((item, gi) => {
      const globalIdx = results.indexOf(item);
      const active    = globalIdx === searchIdx ? " active" : "";
      const badge     = item.type==="page" ? "Page" : item.type==="class" ? "Class" : "Action";
      const iconBg    = item.color ? `background:${item.color}18` : "background:var(--purple-lt)";
      html += `
        <div class="search-item${active}" data-idx="${globalIdx}"
             onclick="handleSearchSelect(${SEARCH_INDEX.indexOf(item)})">
          <div class="search-item-icon" style="${iconBg}">${item.icon}</div>
          <div class="search-item-text">
            <div class="search-item-title">${highlight(item.title, q)}</div>
            <div class="search-item-sub">${item.sub}</div>
          </div>
          <span class="search-item-badge">${badge}</span>
        </div>`;
    });
  }
  container.innerHTML = html;
}

function highlight(text, q) {
  if (!q) return text;
  const re = new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")})`, "gi");
  return text.replace(re, '<mark style="background:#EDE9FE;color:#7C3AED;border-radius:2px">$1</mark>');
}

function handleSearchSelect(globalIdx) {
  const item = SEARCH_INDEX[globalIdx];
  if (!item) return;
  if (item.page)   { closeSearch(); navigateTo(item.page); }
  if (item.action) { item.action(); }
  if (item.type === "class") { closeSearch(); navigateTo("classes"); }
}

// Search input events
$("searchInput").addEventListener("input", e => {
  searchIdx = 0;
  renderSearchResults(e.target.value);
});

$("searchInput").addEventListener("keydown", e => {
  const items = $("searchResults").querySelectorAll(".search-item");
  if (e.key === "ArrowDown") {
    e.preventDefault();
    searchIdx = Math.min(searchIdx + 1, items.length - 1);
    items.forEach((el,i) => el.classList.toggle("active", i===searchIdx));
    items[searchIdx]?.scrollIntoView({block:"nearest"});
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    searchIdx = Math.max(searchIdx - 1, 0);
    items.forEach((el,i) => el.classList.toggle("active", i===searchIdx));
    items[searchIdx]?.scrollIntoView({block:"nearest"});
  } else if (e.key === "Enter") {
    const active = $("searchResults").querySelector(".search-item.active");
    if (active) active.click();
  } else if (e.key === "Escape") {
    closeSearch();
  }
});

$("searchBtn").addEventListener("click", openSearch);
$("searchEsc").addEventListener("click",  closeSearch);
$("searchOverlay").addEventListener("click", e => { if(e.target===$("searchOverlay")) closeSearch(); });

// Ctrl+K / Cmd+K shortcut
document.addEventListener("keydown", e => {
  if ((e.ctrlKey||e.metaKey) && e.key==="k") { e.preventDefault(); openSearch(); }
  if (e.key==="Escape" && searchActive) closeSearch();
});


// ══════════════════════════════════════════════════════════════
//  NOTIFICATION SYSTEM
// ══════════════════════════════════════════════════════════════

let notifications = [];
let notifUnread   = 0;
let notifOpen     = false;

// Create toast container
const toastContainer = document.createElement("div");
toastContainer.className = "toast-container";
toastContainer.id = "toastContainer";
document.body.appendChild(toastContainer);

const NOTIF_ICONS = { error:"🔴", warning:"🟡", success:"🟢", info:"🔵" };
const NOTIF_TITLES = { error:"Error", warning:"Warning", success:"Success", info:"Info" };

/**
 * Add a notification.
 * type: "error" | "warning" | "success" | "info"
 * msg:  message string
 * showToast: also show a toast popup (default true)
 */
function addNotification(type, msg, showToast=true) {
  const now  = new Date();
  const time = now.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
  const notif = { id: Date.now(), type, msg, time, read: false };
  notifications.unshift(notif);
  notifUnread++;
  updateNotifBadge();
  renderNotifList();
  if (showToast) showToastPopup(type, msg);
}

function updateNotifBadge() {
  const badge = $("notifBadge");
  if (notifUnread > 0) {
    badge.style.display = "flex";
    badge.textContent   = notifUnread > 9 ? "9+" : notifUnread;
  } else {
    badge.style.display = "none";
  }
}

function renderNotifList() {
  const list = $("notifList");
  if (!notifications.length) {
    list.innerHTML = '<div class="notif-empty">No notifications yet</div>';
    return;
  }
  list.innerHTML = notifications.slice(0, 20).map(n => `
    <div class="notif-item ${n.read ? "" : "unread"}">
      <span class="notif-dot ${n.type}"></span>
      <div class="notif-body">
        <div class="notif-msg">${n.msg}</div>
        <div class="notif-time">${NOTIF_TITLES[n.type]} · ${n.time}</div>
      </div>
    </div>
  `).join("");
}

function showToastPopup(type, msg, duration=5000) {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${NOTIF_ICONS[type]}</span>
    <div class="toast-body">
      <div class="toast-title">${NOTIF_TITLES[type]}</div>
      <div class="toast-msg">${msg}</div>
    </div>
    <button class="toast-close" onclick="this.closest('.toast').remove()">✕</button>
  `;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("removing");
    setTimeout(() => toast.remove(), 200);
  }, duration);
}

// Notification panel toggle
$("notifBtn").addEventListener("click", e => {
  e.stopPropagation();
  notifOpen = !notifOpen;
  $("notifPanel").style.display = notifOpen ? "block" : "none";
  if (notifOpen) {
    // Mark all as read
    notifications.forEach(n => n.read = true);
    notifUnread = 0;
    updateNotifBadge();
    renderNotifList();
  }
});

// Close panel on outside click
document.addEventListener("click", e => {
  if (notifOpen && !$("notifPanel").contains(e.target) && e.target !== $("notifBtn")) {
    notifOpen = false;
    $("notifPanel").style.display = "none";
  }
});

$("notifClear").addEventListener("click", () => {
  notifications = [];
  notifUnread   = 0;
  updateNotifBadge();
  renderNotifList();
});

// ── Hook notifications into existing events ────────────────────

// Override runAnalysis to add notifications
const _origRunAnalysis = runAnalysis;
window.runAnalysis = async function() {
  // Will be called from analyseBtn — patch the fetch
  return _origRunAnalysis.apply(this, arguments);
};

// Patch the fetch inside runAnalysis by wrapping the analyseBtn handler
document.getElementById("analyseBtn").removeEventListener("click", runAnalysis);
document.getElementById("analyseBtn").addEventListener("click", async function() {
  if (!selectedFile) return;
  $("analyseBtn").disabled = true;
  $("loadingOverlay").style.display = "flex";

  const form = new FormData();
  form.append("image", selectedFile);
  form.append("conf", (parseInt($("confSlider").value)/100).toFixed(2));

  try {
    const res  = await fetch(`${CONFIG.API_BASE}/api/predict`, { method:"POST", body:form });
    const data = await res.json();

    // ── Not a solar panel ─────────────────────────────────────
    if (res.status === 422 && data.error === "not_solar_panel") {
      showNotPanelError(data.message, data.reason, data.suggestion);
      addNotification("warning", `Upload rejected: ${data.message}`);
      return;
    }

    if (!res.ok || data.error) {
      addNotification("error", `Detection failed: ${data.error || "Unknown error"}`);
      alert("Error: " + (data.error||"Unknown"));
      return;
    }

    // Success notification
    const sev = data.severity;
    const type = sev === "Critical" || sev === "High" ? "error"
               : sev === "Moderate" ? "warning"
               : sev === "No Damage" ? "success" : "info";
    addNotification(type,
      `${data.primary_class} detected — Damage: ${data.avg_damage}% [${sev}]`
    );

    lastResult = { ...data, filename: selectedFile.name };
    addToHistory(data, selectedFile.name);
    renderResults(data);
    navigateTo("results");

  } catch(err) {
    addNotification("error", `Connection failed: ${err.message}`);
  } finally {
    $("loadingOverlay").style.display = "none";
    $("analyseBtn").disabled = false;
  }
});

// Notify on health check failure
const _origCheckHealth = checkHealth;
window.checkHealth = async function() {
  try {
    const r = await fetch(`${CONFIG.API_BASE}/api/health`);
    const d = await r.json();
    if (d.status === "ok") {
      $("sbDot").className = "sb-status-dot online";
      $("sbStatusText").textContent = d.gpu_name
        ? d.gpu_name.replace("NVIDIA ","").substring(0,22)
        : "CPU Mode";
      $("gpuLabel").textContent = d.gpu_name ? "GPU Ready" : "CPU Mode";
      $("deviceLabel").textContent = d.device.toUpperCase();
      $("settingDevice").textContent = d.device.toUpperCase();
      $("settingGpu").textContent = d.gpu_name || "N/A";
    }
  } catch {
    $("sbDot").className = "sb-status-dot offline";
    $("sbStatusText").textContent = "Backend offline";
    $("gpuLabel").textContent = "Offline";
    addNotification("error", "Backend server is offline. Please restart python backend/app.py");
  }
};

// Startup notification
setTimeout(() => {
  addNotification("success", "SolarScan loaded successfully. Ready to analyse panels.", false);
  renderNotifList();
  updateNotifBadge();
}, 2000);


// ══════════════════════════════════════════════════════════════
//  LIVE STREAM PAGE
// ══════════════════════════════════════════════════════════════

let liveMode        = "image";
let liveRunning     = false;
let liveTimer       = null;
let liveFrameCount  = 0;
let liveAlertCount  = 0;
let cameraStream    = null;
let videoFile       = null;

// ── Tab switching ─────────────────────────────────────────────
document.querySelectorAll(".live-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".live-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    liveMode = tab.dataset.mode;
    ["image","video","live"].forEach(m => {
      const el = document.getElementById(`liveMode-${m}`);
      if (el) el.style.display = m === liveMode ? "block" : "none";
    });
    if (liveMode === "live") loadCameras();
  });
});

// ── Image mode ────────────────────────────────────────────────
$("liveBrowseBtn").addEventListener("click", () => $("liveFileInput").click());
$("liveDropzone").addEventListener("click", () => $("liveFileInput").click());
$("liveDropzone").addEventListener("dragover", e => { e.preventDefault(); $("liveDropzone").classList.add("drag-over"); });
$("liveDropzone").addEventListener("dragleave", () => $("liveDropzone").classList.remove("drag-over"));
$("liveDropzone").addEventListener("drop", e => {
  e.preventDefault(); $("liveDropzone").classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleLiveImageFile(e.dataTransfer.files[0]);
});
$("liveFileInput").addEventListener("change", e => { if(e.target.files[0]) handleLiveImageFile(e.target.files[0]); });

function handleLiveImageFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    $("livePreviewImg").src = e.target.result;
    $("liveDropzone").style.display    = "none";
    $("liveImagePreview").style.display = "block";
  };
  reader.readAsDataURL(file);
  // Store file for analysis
  $("liveFileInput")._file = file;
}

$("liveAnalyseBtn").addEventListener("click", async () => {
  const file = $("liveFileInput")._file || $("liveFileInput").files[0];
  if (!file) return;
  $("liveAnalyseBtn").disabled = true;
  $("liveAnalyseBtn").textContent = "Analysing…";
  await runLiveAnalysis(file);
  $("liveAnalyseBtn").disabled = false;
  $("liveAnalyseBtn").textContent = "Analyse Image";
});

// ── Video mode ────────────────────────────────────────────────
$("videoBrowseBtn").addEventListener("click", () => $("videoFileInput").click());
$("videoDropzone").addEventListener("click", () => $("videoFileInput").click());
$("videoFileInput").addEventListener("change", e => {
  if (!e.target.files[0]) return;
  videoFile = e.target.files[0];
  const url = URL.createObjectURL(videoFile);
  $("videoPreview").src = url;
  $("videoDropzone").style.display   = "none";
  $("videoPreviewWrap").style.display = "block";
});
$("videoInterval").addEventListener("input", () => {
  $("videoIntervalVal").textContent = $("videoInterval").value + "s";
});

$("videoStartBtn").addEventListener("click", () => {
  if (!videoFile) return;
  liveRunning = true;
  $("videoStartBtn").disabled = true;
  $("videoStopBtn").disabled  = false;
  setLiveStatus("running", "Running");
  $("liveDot").style.display = "inline-block";
  const video = $("videoPreview");
  video.play();
  const interval = parseInt($("videoInterval").value) * 1000;
  liveTimer = setInterval(async () => {
    if (video.paused || video.ended) { stopLive(); return; }
    const canvas = $("cameraCanvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 360;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(async blob => { await runLiveAnalysis(blob, "frame.jpg"); }, "image/jpeg", 0.85);
  }, interval);
});

$("videoStopBtn").addEventListener("click", stopLive);

// ── Camera mode ───────────────────────────────────────────────
async function loadCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === "videoinput");
    const sel = $("cameraSelect");
    sel.innerHTML = cameras.length
      ? cameras.map((c,i) => `<option value="${c.deviceId}">${c.label || "Camera " + (i+1)}</option>`).join("")
      : "<option>No cameras found</option>";
  } catch {
    $("cameraSelect").innerHTML = "<option>Camera access denied</option>";
  }
}

$("liveInterval").addEventListener("input", () => {
  $("liveIntervalVal").textContent = $("liveInterval").value + "s";
});

$("liveStartBtn").addEventListener("click", async () => {
  const deviceId = $("cameraSelect").value;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : true
    });
    $("cameraFeed").srcObject = cameraStream;
    await $("cameraFeed").play();
    liveRunning = true;
    $("liveStartBtn").disabled = true;
    $("liveStopBtn").disabled  = false;
    setLiveStatus("running", "Live");
    $("liveDot").style.display = "inline-block";
    const interval = parseInt($("liveInterval").value) * 1000;
    liveTimer = setInterval(async () => {
      const video  = $("cameraFeed");
      const canvas = $("cameraCanvas");
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 360;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.toBlob(async blob => {
        $("liveCamOverlay").style.display = "flex";
        $("liveCamLabel").textContent = "Analysing…";
        await runLiveAnalysis(blob, "live_frame.jpg");
        $("liveCamOverlay").style.display = "none";
      }, "image/jpeg", 0.85);
    }, interval);
  } catch(err) {
    addNotification("error", "Camera access failed: " + err.message);
  }
});

$("liveStopBtn").addEventListener("click", stopLive);

function stopLive() {
  liveRunning = false;
  clearInterval(liveTimer);
  liveTimer = null;
  $("liveStartBtn").disabled  = false;
  $("liveStopBtn").disabled   = true;
  $("videoStartBtn").disabled = false;
  $("videoStopBtn").disabled  = true;
  $("liveDot").style.display  = "none";
  setLiveStatus("idle", "Idle");
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
}

function setLiveStatus(cls, text) {
  const badge = $("liveStatusBadge");
  badge.className = "live-status-badge " + cls;
  badge.textContent = text;
}

// ── Core analysis function ────────────────────────────────────
async function runLiveAnalysis(fileOrBlob, filename) {
  const form = new FormData();
  form.append("image", fileOrBlob, filename || "image.jpg");
  form.append("conf", "0.20");

  try {
    const res  = await fetch(`${CONFIG.API_BASE}/api/predict`, { method:"POST", body:form });
    const data = await res.json();

    if (res.status === 422 && data.error === "not_solar_panel") {
      appendLiveLog("—", "Not a panel", 0, "—", "#9CA3AF");
      return;
    }
    if (!res.ok || data.error) return;

    liveFrameCount++;
    $("liveFrameCount").textContent = liveFrameCount;
    $("liveFrameSub").textContent   = `frame${liveFrameCount>1?"s":""}`;

    const cls    = data.primary_class;
    const dmg    = data.avg_damage;
    const sev    = data.severity;
    const color  = data.primary_color;
    const conf   = data.detections[0]?.confidence || 0;

    $("liveLastClass").textContent  = cls;
    $("liveLastClass").style.color  = color;
    $("liveLastConf").textContent   = conf + "% conf";
    $("liveLastDamage").textContent = dmg + "%";
    $("liveLastDamage").style.color = dmgColor(dmg);
    $("liveLastSev").textContent    = sev;
    $("liveFrameTime").textContent  = new Date().toLocaleTimeString();

    // Update last frame image
    if (data.annotated_image) {
      $("liveLastFrame").src = "data:image/jpeg;base64," + data.annotated_image;
    }

    // Count alerts
    if (cls !== "Clean") {
      liveAlertCount++;
      $("liveAlertCount").textContent = liveAlertCount;
      const type = sev === "Critical" || sev === "High" ? "error" : "warning";
      addNotification(type, `Live: ${cls} detected — ${dmg}% damage [${sev}]`);
    }

    appendLiveLog(new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"}),
                  cls, dmg, sev, color);

  } catch(err) {
    // Silent fail for live frames
  }
}

function appendLiveLog(time, cls, dmg, sev, color) {
  const log = $("liveLog");
  const empty = log.querySelector(".live-log-empty");
  if (empty) empty.remove();

  const entry = document.createElement("div");
  entry.className = "live-log-entry";
  entry.innerHTML = `
    <span class="live-log-time">${time}</span>
    <span class="live-log-class" style="color:${color}">${cls}</span>
    <span class="live-log-dmg" style="color:${dmgColor(typeof dmg==='number'?dmg:0)}">${dmg}%</span>
    <span class="sev-badge ${sevCss(sev)}" style="font-size:.65rem">${sev}</span>
  `;
  log.insertBefore(entry, log.firstChild);

  // Keep max 50 entries
  while (log.children.length > 50) log.removeChild(log.lastChild);
}

$("clearLiveLog").addEventListener("click", () => {
  $("liveLog").innerHTML = '<div class="live-log-empty">Start streaming to see detections…</div>';
  liveFrameCount = liveAlertCount = 0;
  $("liveFrameCount").textContent = "0";
  $("liveAlertCount").textContent = "0";
  $("liveLastClass").textContent  = "—";
  $("liveLastDamage").textContent = "—";
  $("liveLastSev").textContent    = "—";
});

// Add live page to search index
SEARCH_INDEX.push(
  { type:"page", icon:"📹", title:"Live Stream", sub:"Real-time camera detection", page:"live", tags:["live","camera","stream","video","realtime"] }
);


// ══════════════════════════════════════════════════════════════
//  ANALYSE PANEL – Image/Video/Live tabs
// ══════════════════════════════════════════════════════════════

let analyseMode2   = "image";
let liveRunning2   = false;
let liveTimer2     = null;
let liveFrames2    = 0;
let liveAlerts2    = 0;
let cameraStream2  = null;
let videoFile2     = null;

// ── Tab switching ─────────────────────────────────────────────
document.querySelectorAll(".live-tab[data-amode]").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".live-tab[data-amode]").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    analyseMode2 = tab.dataset.amode;
    ["image","video","live"].forEach(m => {
      const el = document.getElementById(`amode-${m}`);
      if (el) el.style.display = m === analyseMode2 ? "block" : "none";
    });
    if (analyseMode2 === "live") loadCameras2();
    // Stop any running stream when switching tabs
    if (analyseMode2 !== "video" && analyseMode2 !== "live") stopLive2();
  });
});

// ── Video mode ────────────────────────────────────────────────
$("videoBrowseBtn2").addEventListener("click", () => $("videoFileInput2").click());
$("videoDropzone2").addEventListener("click",  () => $("videoFileInput2").click());
$("videoFileInput2").addEventListener("change", e => {
  if (!e.target.files[0]) return;
  videoFile2 = e.target.files[0];
  $("videoPreview2").src = URL.createObjectURL(videoFile2);
  $("videoDropzone2").style.display    = "none";
  $("videoPreviewWrap2").style.display = "block";
});
$("videoInterval2").addEventListener("input", () => {
  $("videoIntervalVal2").textContent = $("videoInterval2").value + "s";
});

$("videoStartBtn2").addEventListener("click", () => {
  if (!videoFile2) return;
  liveRunning2 = true;
  $("videoStartBtn2").disabled = true;
  $("videoStopBtn2").disabled  = false;
  setAnalyseStatus("running", "Running");
  showLivePanels();
  const video    = $("videoPreview2");
  const interval = parseInt($("videoInterval2").value) * 1000;
  video.play();
  liveTimer2 = setInterval(async () => {
    if (video.paused || video.ended) { stopLive2(); return; }
    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 360;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(async blob => { await runLiveAnalysis2(blob, "frame.jpg"); }, "image/jpeg", 0.85);
  }, interval);
});
$("videoStopBtn2").addEventListener("click", stopLive2);

// ── Camera mode ───────────────────────────────────────────────
async function loadCameras2() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams    = devices.filter(d => d.kind === "videoinput");
    const sel     = $("cameraSelect2");
    sel.innerHTML = cams.length
      ? cams.map((c,i) => `<option value="${c.deviceId}">${c.label || "Camera "+(i+1)}</option>`).join("")
      : "<option>No cameras found</option>";
  } catch {
    $("cameraSelect2").innerHTML = "<option>Camera access denied</option>";
  }
}

$("liveInterval2").addEventListener("input", () => {
  $("liveIntervalVal2").textContent = $("liveInterval2").value + "s";
});

$("liveStartBtn2").addEventListener("click", async () => {
  const deviceId = $("cameraSelect2").value;
  try {
    cameraStream2 = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : true
    });
    $("cameraFeed2").srcObject = cameraStream2;
    await $("cameraFeed2").play();
    liveRunning2 = true;
    $("liveStartBtn2").disabled = true;
    $("liveStopBtn2").disabled  = false;
    setAnalyseStatus("running", "Live");
    showLivePanels();
    const interval = parseInt($("liveInterval2").value) * 1000;
    liveTimer2 = setInterval(async () => {
      const video  = $("cameraFeed2");
      const canvas = $("cameraCanvas2");
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 360;
      canvas.getContext("2d").drawImage(video, 0, 0);
      $("liveCamOverlay2").style.display = "flex";
      $("liveCamLabel2").textContent = "Analysing…";
      canvas.toBlob(async blob => {
        await runLiveAnalysis2(blob, "live.jpg");
        $("liveCamOverlay2").style.display = "none";
      }, "image/jpeg", 0.85);
    }, interval);
  } catch(err) {
    addNotification("error", "Camera access failed: " + err.message);
  }
});
$("liveStopBtn2").addEventListener("click", stopLive2);

function stopLive2() {
  liveRunning2 = false;
  clearInterval(liveTimer2); liveTimer2 = null;
  $("liveStartBtn2").disabled  = false;
  $("liveStopBtn2").disabled   = true;
  $("videoStartBtn2").disabled = false;
  $("videoStopBtn2").disabled  = true;
  setAnalyseStatus("idle", "Ready");
  if (cameraStream2) { cameraStream2.getTracks().forEach(t=>t.stop()); cameraStream2=null; }
}

function setAnalyseStatus(cls, text) {
  const b = $("analyseStatusBadge");
  if (b) { b.className = "live-status-badge " + cls; b.textContent = text; }
}

function showLivePanels() {
  $("liveResultsMini").style.display    = "block";
  $("liveLastFrameCard").style.display  = "block";
  liveFrames2 = liveAlerts2 = 0;
  $("miniFrames").textContent = "0";
  $("miniAlerts").textContent = "0";
  $("miniClass").textContent  = "—";
  $("miniDamage").textContent = "—";
}

// ── Core live analysis ────────────────────────────────────────
async function runLiveAnalysis2(blob, filename) {
  const form = new FormData();
  form.append("image", blob, filename);
  form.append("conf", "0.20");
  try {
    const res  = await fetch(`${CONFIG.API_BASE}/api/predict`, { method:"POST", body:form });
    const data = await res.json();
    if (res.status === 422 || !res.ok || data.error) return;

    liveFrames2++;
    const cls   = data.primary_class;
    const dmg   = data.avg_damage;
    const sev   = data.severity;
    const color = data.primary_color;
    const conf  = data.detections[0]?.confidence || 0;

    // Update mini stats
    $("miniFrames").textContent = liveFrames2;
    $("miniClass").textContent  = cls;
    $("miniClass").style.color  = color;
    $("miniDamage").textContent = dmg + "%";
    $("miniDamage").style.color = dmgColor(dmg);

    // Update last frame
    $("liveFrameTime2").textContent = new Date().toLocaleTimeString();
    if (data.annotated_image) {
      $("liveLastFrame2").src = "data:image/jpeg;base64," + data.annotated_image;
    }

    // Alert on defect
    if (cls !== "Clean") {
      liveAlerts2++;
      $("miniAlerts").textContent = liveAlerts2;
      const type = sev==="Critical"||sev==="High" ? "error" : "warning";
      addNotification(type, `Live: ${cls} — ${dmg}% [${sev}]`);
    }

    // Log entry
    const log   = $("liveLog2");
    const empty = log.querySelector(".live-log-empty");
    if (empty) empty.remove();
    const entry = document.createElement("div");
    entry.className = "live-log-entry";
    entry.innerHTML = `
      <span class="live-log-time">${new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}</span>
      <span class="live-log-class" style="color:${color}">${cls}</span>
      <span class="live-log-dmg" style="color:${dmgColor(dmg)}">${dmg}%</span>
      <span class="sev-badge ${sevCss(sev)}" style="font-size:.65rem">${sev}</span>
    `;
    log.insertBefore(entry, log.firstChild);
    while (log.children.length > 30) log.removeChild(log.lastChild);

  } catch { /* silent */ }
}


// ══════════════════════════════════════════════════════════════
//  ANALYSE PANEL TABS  (Image / Video / Live)
// ══════════════════════════════════════════════════════════════
(function() {
  "use strict";

  let aMode      = "image";
  let aRunning   = false;
  let aTimer     = null;
  let aFrames    = 0;
  let aAlerts    = 0;
  let aCamStream = null;
  let aVideoFile = null;

  function el(id) { return document.getElementById(id); }

  // ── Tab switching ───────────────────────────────────────────
  function initTabs() {
    document.querySelectorAll(".live-tab[data-amode]").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".live-tab[data-amode]").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        aMode = tab.dataset.amode;
        ["image","video","live"].forEach(m => {
          const p = el("amode-" + m);
          if (p) p.style.display = (m === aMode) ? "block" : "none";
        });
        if (aMode === "live") loadCams();
        if (aMode === "image") stopStream();
      });
    });
  }

  // ── Video mode ──────────────────────────────────────────────
  function initVideo() {
    const browse = el("videoBrowseBtn2");
    const drop   = el("videoDropzone2");
    const input  = el("videoFileInput2");
    if (!browse || !drop || !input) return;

    browse.addEventListener("click", () => input.click());
    drop.addEventListener("click",   () => input.click());
    drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("drag-over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
    drop.addEventListener("drop", e => {
      e.preventDefault(); drop.classList.remove("drag-over");
      if (e.dataTransfer.files[0]) loadVideoFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", e => { if (e.target.files[0]) loadVideoFile(e.target.files[0]); });

    el("videoInterval2").addEventListener("input", () => {
      el("videoIntervalVal2").textContent = el("videoInterval2").value + "s";
    });

    el("videoStartBtn2").addEventListener("click", startVideo);
    el("videoStopBtn2").addEventListener("click",  stopStream);
  }

  function loadVideoFile(file) {
    aVideoFile = file;
    el("videoPreview2").src = URL.createObjectURL(file);
    el("videoDropzone2").style.display    = "none";
    el("videoPreviewWrap2").style.display = "block";
  }

  function startVideo() {
    if (!aVideoFile) return;
    aRunning = true;
    el("videoStartBtn2").disabled = true;
    el("videoStopBtn2").disabled  = false;
    setStatus("running", "Running");
    showLiveUI();
    const video    = el("videoPreview2");
    const interval = parseInt(el("videoInterval2").value) * 1000;
    video.play();
    aTimer = setInterval(() => {
      if (video.paused || video.ended) { stopStream(); return; }
      const c = document.createElement("canvas");
      c.width = video.videoWidth || 640; c.height = video.videoHeight || 360;
      c.getContext("2d").drawImage(video, 0, 0);
      c.toBlob(blob => analyseBlob(blob, "frame.jpg"), "image/jpeg", 0.85);
    }, interval);
  }

  // ── Live camera mode ────────────────────────────────────────
  async function loadCams() {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const cams = devs.filter(d => d.kind === "videoinput");
      const sel  = el("cameraSelect2");
      if (!sel) return;
      sel.innerHTML = cams.length
        ? cams.map((c,i) => `<option value="${c.deviceId}">${c.label || "Camera "+(i+1)}</option>`).join("")
        : "<option>No cameras found</option>";
    } catch { if (el("cameraSelect2")) el("cameraSelect2").innerHTML = "<option>Access denied</option>"; }
  }

  function initLive() {
    const intSlider = el("liveInterval2");
    if (intSlider) intSlider.addEventListener("input", () => {
      el("liveIntervalVal2").textContent = intSlider.value + "s";
    });
    const startBtn = el("liveStartBtn2");
    const stopBtn  = el("liveStopBtn2");
    if (startBtn) startBtn.addEventListener("click", startLive);
    if (stopBtn)  stopBtn.addEventListener("click",  stopStream);
  }

  async function startLive() {
    const deviceId = el("cameraSelect2")?.value;
    try {
      aCamStream = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId } } : true
      });
      const feed = el("cameraFeed2");
      feed.srcObject = aCamStream;
      await feed.play();
      aRunning = true;
      el("liveStartBtn2").disabled = true;
      el("liveStopBtn2").disabled  = false;
      setStatus("running", "Live");
      showLiveUI();
      const interval = parseInt(el("liveInterval2").value) * 1000;
      aTimer = setInterval(() => {
        const canvas = el("cameraCanvas2");
        canvas.width  = feed.videoWidth  || 640;
        canvas.height = feed.videoHeight || 360;
        canvas.getContext("2d").drawImage(feed, 0, 0);
        const overlay = el("liveCamOverlay2");
        if (overlay) { overlay.style.display = "flex"; el("liveCamLabel2").textContent = "Analysing…"; }
        canvas.toBlob(blob => {
          analyseBlob(blob, "live.jpg").then(() => {
            if (overlay) overlay.style.display = "none";
          });
        }, "image/jpeg", 0.85);
      }, interval);
    } catch(err) {
      if (typeof addNotification === "function")
        addNotification("error", "Camera access failed: " + err.message);
      else alert("Camera access failed: " + err.message);
    }
  }

  function stopStream() {
    aRunning = false;
    clearInterval(aTimer); aTimer = null;
    if (el("liveStartBtn2"))  el("liveStartBtn2").disabled  = false;
    if (el("liveStopBtn2"))   el("liveStopBtn2").disabled   = true;
    if (el("videoStartBtn2")) el("videoStartBtn2").disabled = false;
    if (el("videoStopBtn2"))  el("videoStopBtn2").disabled  = true;
    setStatus("idle", "Ready");
    if (aCamStream) { aCamStream.getTracks().forEach(t => t.stop()); aCamStream = null; }
  }

  function setStatus(cls, text) {
    const b = el("analyseStatusBadge");
    if (b) { b.className = "live-status-badge " + cls; b.textContent = text; }
  }

  function showLiveUI() {
    const mini = el("liveResultsMini");
    const card = el("liveLastFrameCard");
    if (mini) mini.style.display = "block";
    if (card) card.style.display = "block";
    aFrames = aAlerts = 0;
    ["miniFrames","miniAlerts"].forEach(id => { if (el(id)) el(id).textContent = "0"; });
    ["miniClass","miniDamage"].forEach(id => { if (el(id)) el(id).textContent = "—"; });
  }

  // ── Core: send blob to /api/predict ────────────────────────
  async function analyseBlob(blob, filename) {
    const form = new FormData();
    form.append("image", blob, filename);
    form.append("conf", "0.20");
    try {
      const res  = await fetch((window.CONFIG?.API_BASE || "") + "/api/predict",
                               { method: "POST", body: form });
      const data = await res.json();
      if (res.status === 422 || !res.ok || data.error) {
        appendALog("—", "Not a panel", 0, "—", "#9CA3AF");
        return;
      }
      aFrames++;
      const cls   = data.primary_class;
      const dmg   = data.avg_damage;
      const sev   = data.severity;
      const color = data.primary_color;
      const conf  = data.detections?.[0]?.confidence || 0;

      if (el("miniFrames"))  el("miniFrames").textContent = aFrames;
      if (el("miniClass"))   { el("miniClass").textContent = cls; el("miniClass").style.color = color; }
      if (el("miniDamage"))  { el("miniDamage").textContent = dmg + "%"; el("miniDamage").style.color = dmgColor(dmg); }
      if (el("liveFrameTime2")) el("liveFrameTime2").textContent = new Date().toLocaleTimeString();
      if (data.annotated_image && el("liveLastFrame2"))
        el("liveLastFrame2").src = "data:image/jpeg;base64," + data.annotated_image;

      if (cls !== "Clean") {
        aAlerts++;
        if (el("miniAlerts")) el("miniAlerts").textContent = aAlerts;
        const type = (sev === "Critical" || sev === "High") ? "error" : "warning";
        if (typeof addNotification === "function")
          addNotification(type, `Live: ${cls} — ${dmg}% [${sev}]`);
      }
      appendALog(new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"}),
                 cls, dmg, sev, color);
    } catch { /* silent */ }
  }

  function appendALog(time, cls, dmg, sev, color) {
    const log = el("liveLog2");
    if (!log) return;
    const empty = log.querySelector(".live-log-empty");
    if (empty) empty.remove();
    const entry = document.createElement("div");
    entry.className = "live-log-entry";
    const sevCssClass = (typeof sevCss === "function") ? sevCss(sev) : "sev-low";
    const dmgC = (typeof dmgColor === "function") ? dmgColor(typeof dmg === "number" ? dmg : 0) : "#374151";
    entry.innerHTML = `
      <span class="live-log-time">${time}</span>
      <span class="live-log-class" style="color:${color}">${cls}</span>
      <span class="live-log-dmg" style="color:${dmgC}">${dmg}%</span>
      <span class="sev-badge ${sevCssClass}" style="font-size:.65rem">${sev}</span>
    `;
    log.insertBefore(entry, log.firstChild);
    while (log.children.length > 30) log.removeChild(log.lastChild);
  }

  // ── Init on DOM ready ───────────────────────────────────────
  function init() {
    initTabs();
    initVideo();
    initLive();
    const clearBtn = el("clearLiveLog2");
    if (clearBtn) clearBtn.addEventListener("click", () => {
      const log = el("liveLog2");
      if (log) log.innerHTML = '<div class="live-log-empty">Waiting for frames…</div>';
      aFrames = aAlerts = 0;
      ["miniFrames","miniAlerts"].forEach(id => { if (el(id)) el(id).textContent = "0"; });
    });
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else
    init();
})();
