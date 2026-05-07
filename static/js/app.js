/* ═══════════════════════════════════════════════════════════════
   SolarScan  –  Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

// ── Severity helpers ────────────────────────────────────────────
const SEV_CLASS = {
  "No Damage": "sev-no-damage",
  "Minimal":   "sev-minimal",
  "Low":       "sev-low",
  "Moderate":  "sev-moderate",
  "High":      "sev-high",
  "Critical":  "sev-critical",
};

const CLASS_COLORS = {
  "Bird-drop":          "#F97316",
  "Clean":              "#22C55E",
  "Dusty":              "#EAB308",
  "Electrical-damage":  "#DC2626",
  "Physical-Damage":    "#3B82F6",
  "Snow-Covered":       "#06B6D4",
};

function sevClass(sev) {
  return SEV_CLASS[sev] || "sev-low";
}

function classColor(cls) {
  return CLASS_COLORS[cls] || "#94A3B8";
}

// ── DOM refs ────────────────────────────────────────────────────
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const browseBtn      = document.getElementById("browseBtn");
const uploadControls = document.getElementById("uploadControls");
const previewImg     = document.getElementById("previewImg");
const removeBtn      = document.getElementById("removeBtn");
const confSlider     = document.getElementById("confSlider");
const confValue      = document.getElementById("confValue");
const analyseBtn     = document.getElementById("analyseBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const resultsSection = document.getElementById("resultsSection");
const newScanBtn     = document.getElementById("newScanBtn");

let selectedFile = null;

// ── File selection ──────────────────────────────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file (JPG, PNG).");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropZone.style.display = "none";
    uploadControls.style.display = "block";
    resultsSection.style.display = "none";
  };
  reader.readAsDataURL(file);
}

removeBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  dropZone.style.display = "block";
  uploadControls.style.display = "none";
  resultsSection.style.display = "none";
});

// ── Confidence slider ───────────────────────────────────────────
confSlider.addEventListener("input", () => {
  confValue.textContent = confSlider.value + "%";
});

// ── Analyse ─────────────────────────────────────────────────────
analyseBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;

  analyseBtn.disabled = true;
  loadingOverlay.style.display = "flex";
  resultsSection.style.display = "none";

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("conf", (parseInt(confSlider.value) / 100).toFixed(2));

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    renderResults(data);
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    alert("Request failed: " + err.message);
  } finally {
    loadingOverlay.style.display = "none";
    analyseBtn.disabled = false;
  }
}

// ── Render results ──────────────────────────────────────────────
function renderResults(data) {
  const { annotated_image, primary_class, avg_damage, severity, detections, diagnosis } = data;

  // ── Summary bar ──────────────────────────────────────────────
  document.getElementById("resClass").textContent   = primary_class;
  document.getElementById("resDamage").textContent  = avg_damage + "%";
  document.getElementById("resCount").textContent   = detections.length;

  const sevBadge = document.getElementById("resSeverity");
  sevBadge.textContent  = severity;
  sevBadge.className    = "summary-badge " + sevClass(severity);

  // ── Damage meter ─────────────────────────────────────────────
  const fill  = document.getElementById("damageMeterFill");
  const thumb = document.getElementById("damageMeterThumb");
  // fill covers the right portion (white overlay from right)
  const pct   = Math.min(Math.max(avg_damage, 0), 100);
  fill.style.width  = (100 - pct) + "%";
  fill.style.left   = "auto";
  fill.style.right  = "0";
  thumb.style.left  = pct + "%";

  // ── Annotated image ───────────────────────────────────────────
  const imgSrc = "data:image/jpeg;base64," + annotated_image;
  document.getElementById("resultImg").src = imgSrc;
  const dlBtn = document.getElementById("downloadBtn");
  dlBtn.href = imgSrc;
  dlBtn.download = `solar_analysis_${primary_class.replace(/\s+/g, "_")}.jpg`;

  // ── Diagnosis ─────────────────────────────────────────────────
  document.getElementById("diagWhat").textContent = diagnosis.what_happened;

  const impactList = document.getElementById("diagImpact");
  impactList.innerHTML = "";
  diagnosis.impact.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    impactList.appendChild(li);
  });

  const stepsList = document.getElementById("diagSteps");
  stepsList.innerHTML = "";
  diagnosis.suggestions.forEach((step, i) => {
    const li   = document.createElement("li");
    const badge = document.createElement("span");
    badge.className   = "step-num-badge";
    badge.textContent = i + 1;
    const text = document.createElement("span");
    // Strip leading "1. " etc from step text
    text.textContent = step.replace(/^\d+\.\s*/, "");
    li.appendChild(badge);
    li.appendChild(text);
    stepsList.appendChild(li);
  });

  // ── Detection table ───────────────────────────────────────────
  const tbody = document.getElementById("detTableBody");
  tbody.innerHTML = "";
  detections.forEach((det, i) => {
    const color = classColor(det.class);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="color:var(--text-3);font-weight:600">${i + 1}</td>
      <td>
        <span class="class-pill" style="background:${color}22;color:${color}">
          ${det.class}
        </span>
      </td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:${det.confidence}%;background:${color}"></div>
          </div>
          <span style="font-weight:700;color:${color};min-width:42px">${det.confidence}%</span>
        </div>
      </td>
      <td style="font-weight:700;color:${damageColor(det.damage_pct)}">${det.damage_pct}%</td>
      <td><span class="summary-badge ${sevClass(det.severity)}">${det.severity}</span></td>
      <td style="font-size:.8rem;color:var(--text-3);font-family:monospace">
        [${det.bbox[0]}, ${det.bbox[1]}, ${det.bbox[2]}, ${det.bbox[3]}]
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function damageColor(pct) {
  if (pct === 0)   return "#16A34A";
  if (pct < 40)    return "#CA8A04";
  if (pct < 70)    return "#EA580C";
  return "#DC2626";
}

// ── New scan ────────────────────────────────────────────────────
newScanBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  dropZone.style.display = "block";
  uploadControls.style.display = "none";
  resultsSection.style.display = "none";
  document.getElementById("upload-section").scrollIntoView({ behavior: "smooth" });
});

// ── Nav active link ─────────────────────────────────────────────
const navLinks = document.querySelectorAll(".nav-link");
const sections = document.querySelectorAll("section[id]");

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(l => l.classList.remove("active"));
      const active = document.querySelector(`.nav-link[href="#${entry.target.id}"]`);
      if (active) active.classList.add("active");
    }
  });
}, { threshold: 0.4 });

sections.forEach(s => observer.observe(s));
