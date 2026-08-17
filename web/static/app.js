// Gravimem V0 Workbench Frontend Logic

let experimentData = null;
let viewMode = 'active'; // 'active', 'original', 'vectors'
let selectedParticle = null;
let hoveredParticle = null;

// Canvas state
const canvas = document.getElementById('particleCanvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('canvasTooltip');

let zoom = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;

// Coordinate bounds
let bounds = { minX: -5, maxX: 5, minY: -5, maxY: 5 };

// Init Canvas Size
function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  renderCanvas();
}

window.addEventListener('resize', resizeCanvas);

// Fetch Experiment Results
async function loadResults() {
  try {
    const res = await fetch('/api/results');
    if (!res.ok) {
      console.warn("No results yet, waiting or running...");
      return;
    }
    experimentData = await res.json();
    populateUI();
    computeBounds();
    resetView();
    renderCanvas();
  } catch (err) {
    console.error("Failed to load experiment results:", err);
  }
}

function computeBounds() {
  if (!experimentData || !experimentData.particles.length) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

  experimentData.particles.forEach(p => {
    const [x0, y0] = p.x0_2d;
    const [x, y] = p.x_2d;
    minX = Math.min(minX, x0, x);
    maxX = Math.max(maxX, x0, x);
    minY = Math.min(minY, y0, y);
    maxY = Math.max(maxY, y0, y);
  });

  const padX = (maxX - minX) * 0.1 || 1.0;
  const padY = (maxY - minY) * 0.1 || 1.0;
  bounds = {
    minX: minX - padX,
    maxX: maxX + padX,
    minY: minY - padY,
    maxY: maxY + padY
  };
}

function resetView() {
  const rect = canvas.parentElement.getBoundingClientRect();
  zoom = Math.min(
    rect.width / (bounds.maxX - bounds.minX),
    rect.height / (bounds.maxY - bounds.minY)
  ) * 0.85;

  panX = rect.width / 2 - ((bounds.minX + bounds.maxX) / 2) * zoom;
  panY = rect.height / 2 - ((bounds.minY + bounds.maxY) / 2) * zoom;
  renderCanvas();
}

function worldToScreen(wx, wy) {
  return {
    sx: wx * zoom + panX,
    sy: wy * zoom + panY
  };
}

function screenToWorld(sx, sy) {
  return {
    wx: (sx - panX) / zoom,
    wy: (sy - panY) / zoom
  };
}

// Canvas Rendering
function renderCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  if (!experimentData || !experimentData.particles) {
    ctx.fillStyle = '#64748B';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('No particle data loaded. Run experiment first.', w / 2, h / 2);
    return;
  }

  // Draw displacement vectors if enabled
  if (viewMode === 'vectors' || viewMode === 'active') {
    experimentData.particles.forEach(p => {
      const p0 = worldToScreen(p.x0_2d[0], p.x0_2d[1]);
      const p1 = worldToScreen(p.x_2d[0], p.x_2d[1]);
      
      const dx = p1.sx - p0.sx;
      const dy = p1.sy - p0.sy;
      const dist = Math.hypot(dx, dy);

      if (dist > 1.5) {
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.35)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(p0.sx, p0.sy);
        ctx.lineTo(p1.sx, p1.sy);
        ctx.stroke();

        // Draw original anchor point as faint ghost
        ctx.fillStyle = 'rgba(100, 116, 139, 0.4)';
        ctx.beginPath();
        ctx.arc(p0.sx, p0.sy, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  // Draw particles
  experimentData.particles.forEach(p => {
    const coords = viewMode === 'original' ? p.x0_2d : p.x_2d;
    const pt = worldToScreen(coords[0], coords[1]);

    const isSelected = selectedParticle && selectedParticle.id === p.id;
    const isHovered = hoveredParticle && hoveredParticle.id === p.id;

    // Radius scales with mass
    const baseRadius = 3.5;
    const radius = Math.min(18, baseRadius + Math.sqrt(p.mass - 1.0) * 3);

    // Color based on mass
    if (p.mass > 1.5) {
      ctx.fillStyle = isSelected ? '#FBBF24' : '#F59E0B';
      ctx.shadowColor = 'rgba(245, 158, 11, 0.6)';
      ctx.shadowBlur = p.mass > 3 ? 12 : 6;
    } else {
      ctx.fillStyle = isSelected ? '#FFFFFF' : '#38BDF8';
      ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
      ctx.shadowBlur = 4;
    }

    ctx.beginPath();
    ctx.arc(pt.sx, pt.sy, isSelected ? radius + 2 : radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0; // reset

    // Draw label for high mass or hovered/selected
    if (p.mass > 2.0 || isSelected || isHovered || zoom > 80) {
      ctx.fillStyle = isSelected ? '#FFFFFF' : (p.mass > 2.0 ? '#FDE68A' : '#94A3B8');
      ctx.font = `${isSelected ? '600 12px' : '11px'} Inter`;
      ctx.textAlign = 'center';
      ctx.fillText(p.label, pt.sx, pt.sy - radius - 5);
    }
  });
}

// Populate UI Elements
function populateUI() {
  const meta = experimentData.metadata;
  document.getElementById('statParticles').textContent = meta.num_particles;
  document.getElementById('statEvents').textContent = `${meta.train_events_count} train / ${meta.test_events_count} test`;

  // Compute avg displacement
  let totalDisp = 0, maxM = 0;
  experimentData.particles.forEach(p => {
    totalDisp += p.displacement;
    maxM = Math.max(maxM, p.mass);
  });
  document.getElementById('statDisplacement').textContent = (totalDisp / experimentData.particles.length).toFixed(4);
  document.getElementById('statMaxMass').textContent = maxM.toFixed(2);

  // Populate Test Queries Dropdown
  const select = document.getElementById('selectTestQuery');
  select.innerHTML = '';
  experimentData.all_query_evaluations.forEach((item, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = `[#${idx+1}] "${item.test_query}" (${item.timestamp || 'Test Query'})`;
    select.appendChild(opt);
  });

  select.addEventListener('change', () => {
    displayQueryComparison(parseInt(select.value, 10));
  });

  if (experimentData.all_query_evaluations.length > 0) {
    displayQueryComparison(0);
  }

  // Populate Shifts Table
  const tableBody = document.getElementById('shiftsTableBody');
  tableBody.innerHTML = '';
  (experimentData.top_similarity_shifts || []).forEach(shift => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td class="pair-name" title="${shift.concept_1} ↔ ${shift.concept_2}">
        <strong>${shift.concept_1}</strong><br><span style="color:var(--text-muted)">${shift.concept_2}</span>
      </td>
      <td>${shift.orig_sim.toFixed(3)}</td>
      <td>${shift.active_sim.toFixed(3)}</td>
      <td class="delta-pos">+${shift.delta_sim.toFixed(3)}</td>
    `;
    tableBody.appendChild(row);
  });
}

function displayQueryComparison(index) {
  const item = experimentData.all_query_evaluations[index];
  if (!item) return;

  const staticList = document.getElementById('staticResultsList');
  const dynamicList = document.getElementById('dynamicResultsList');

  staticList.innerHTML = item.static_top_k.map(res => `
    <div class="result-item">
      <span class="item-text" title="${res.label}">#${res.rank} ${res.label}</span>
      <div class="item-meta">
        <span class="score-tag">${res.score.toFixed(3)}</span>
      </div>
    </div>
  `).join('');

  dynamicList.innerHTML = item.dynamic_top_k.map(res => `
    <div class="result-item">
      <span class="item-text" title="${res.label}">#${res.rank} ${res.label}</span>
      <div class="item-meta">
        <span class="mass-tag">m=${res.mass.toFixed(1)}</span>
        <span class="score-tag">${res.score.toFixed(3)}</span>
      </div>
    </div>
  `).join('');
}

// Particle Interaction Events
canvas.addEventListener('mousedown', e => {
  isDragging = true;
  startX = e.clientX - panX;
  startY = e.clientY - panY;
});

window.addEventListener('mouseup', () => { isDragging = false; });

canvas.addEventListener('mousemove', e => {
  if (isDragging) {
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    renderCanvas();
    return;
  }

  // Hover detection
  if (!experimentData) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  let found = null;
  for (const p of experimentData.particles) {
    const coords = viewMode === 'original' ? p.x0_2d : p.x_2d;
    const pt = worldToScreen(coords[0], coords[1]);
    const d = Math.hypot(pt.sx - mx, pt.sy - my);
    if (d < 12) {
      found = p;
      break;
    }
  }

  hoveredParticle = found;
  if (found) {
    tooltip.style.display = 'block';
    tooltip.style.left = `${mx + 15}px`;
    tooltip.style.top = `${my + 15}px`;
    tooltip.innerHTML = `
      <strong>${found.label}</strong><br>
      <span style="color:var(--accent-amber)">Mass: ${found.mass.toFixed(2)}</span> · Visits: ${found.visits}<br>
      <span style="color:var(--accent-rose)">Displacement: ${found.displacement.toFixed(4)}</span>
    `;
    canvas.style.cursor = 'pointer';
  } else {
    tooltip.style.display = 'none';
    canvas.style.cursor = 'grab';
  }

  renderCanvas();
});

canvas.addEventListener('click', e => {
  if (hoveredParticle) {
    selectedParticle = hoveredParticle;
    inspectConcept(selectedParticle);
    renderCanvas();
  }
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
  
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  panX = mx - (mx - panX) * zoomFactor;
  panY = my - (my - panY) * zoomFactor;
  zoom *= zoomFactor;

  renderCanvas();
}, { passive: false });

function inspectConcept(p) {
  document.getElementById('inspectLabel').textContent = p.label;
  document.getElementById('conceptMetrics').style.display = 'flex';
  document.getElementById('inspectMass').textContent = p.mass.toFixed(3);
  document.getElementById('inspectVisits').textContent = p.visits;
  document.getElementById('inspectDisp').textContent = p.displacement.toFixed(5);
  document.getElementById('inspectX0').textContent = `[${p.x0_2d[0].toFixed(2)}, ${p.x0_2d[1].toFixed(2)}]`;
  document.getElementById('inspectX').textContent = `[${p.x_2d[0].toFixed(2)}, ${p.x_2d[1].toFixed(2)}]`;

  // Switch to inspector tab
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  const inspectTab = document.querySelector('[data-tab="tab-inspect"]');
  if (inspectTab) inspectTab.classList.add('active');
  document.getElementById('tab-inspect').classList.add('active');
}

// Mode Buttons
document.getElementById('btnModeActive').addEventListener('click', function() {
  document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
  this.classList.add('active');
  viewMode = 'active';
  renderCanvas();
});

document.getElementById('btnModeOriginal').addEventListener('click', function() {
  document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
  this.classList.add('active');
  viewMode = 'original';
  renderCanvas();
});

document.getElementById('btnModeVectors').addEventListener('click', function() {
  document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
  this.classList.add('active');
  viewMode = 'vectors';
  renderCanvas();
});

document.getElementById('btnResetView').addEventListener('click', resetView);

// Tabs Navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// Slider Value Bindings
['G', 'K', 'Beta', 'R', 'Alpha'].forEach(key => {
  const input = document.getElementById(`input${key}`);
  const display = document.getElementById(`val${key}`);
  if (input && display) {
    input.addEventListener('input', () => { display.textContent = input.value; });
  }
});

// Re-run Button
document.getElementById('btnRerunSim').addEventListener('click', async () => {
  const btn = document.getElementById('btnRerunSim');
  btn.textContent = 'Simulating Physics...';
  btn.disabled = true;

  try {
    const payload = {
      events: parseInt(document.getElementById('inputEvents').value, 10),
      G: parseFloat(document.getElementById('inputG').value),
      k: parseFloat(document.getElementById('inputK').value),
      beta: parseFloat(document.getElementById('inputBeta').value),
      R: parseFloat(document.getElementById('inputR').value),
      alpha: parseFloat(document.getElementById('inputAlpha').value)
    };

    const res = await fetch('/api/rerun', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      await loadResults();
    }
  } catch (err) {
    console.error("Rerun failed:", err);
  } finally {
    btn.textContent = 'Re-Run Particle Simulation';
    btn.disabled = false;
  }
});

// Initial load
window.addEventListener('DOMContentLoaded', () => {
  resizeCanvas();
  loadResults();
});
