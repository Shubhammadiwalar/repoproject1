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
    if (!res.ok || data.error) { alert("Error: " + (data.error||"Unknown")); return; }

    lastResult = { ...data, filename: selectedFile.name };
    addToHistory(data, selectedFile.name);
    renderResults(data);
    navigateTo("results");
    $("notifBadge").style.display = "flex";

  } catch(err) {
    alert("Request failed. Is the backend running?\n" + err.message);
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
  tbody.innerHTML = scanHistory.map(e => `
    <tr>
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
  $("downloadBtn").href     = src;
  $("downloadBtn").download = `solar_${primary_class}_${avg_damage}pct.jpg`;

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

// ── Init ──────────────────────────────────────────────────────
(async function init() {
  // Check auth first — redirect to login if not authenticated
  try {
    const r = await fetch(`${CONFIG.API_BASE}/api/auth/me`, { credentials: "include" });
    if (r.status === 401) { window.location.href = "/login.html"; return; }
    const user = await r.json();
    // Show user name in topbar
    const nameEl = document.getElementById("topbar-username");
    if (nameEl) nameEl.textContent = user.name || "User";
    document.querySelector(".user-name").textContent = user.name || "SolarScan";
    document.querySelector(".user-role").textContent = user.email || "";
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
