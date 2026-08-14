'use strict';

/* ============================================================
   AMASE-P Observation Scheduler — frontend logic
   Single page, vanilla JS, Plotly.js from CDN.
   ============================================================ */

/* ---------------- tiny helpers ---------------- */

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

function esc(v) {
  return String(v).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function parseNum(v) {
  if (v === null || v === undefined) return null;
  const t = String(v).trim();
  if (t === '') return null;
  const n = Number(t);
  return Number.isNaN(n) ? null : n;
}

const pad2 = (n) => String(n).padStart(2, '0');

/* Parse a UTC-ish timestamp ("2027-04-01T13:47:35.452", or with a space,
   with or without seconds / trailing Z) into a Date using UTC interpretation.
   All date math below only ever uses *differences* between two such Dates,
   and formatting is done with getUTC*(), so timezone handling stays consistent. */
function parseUtc(s) {
  if (!s) return null;
  const t = String(s).trim();
  const m = t.match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?(Z)?$/);
  if (m) {
    const [, y, mo, d, h, mi, se, fr] = m;
    let ms = 0;
    if (fr) ms = Math.round(parseFloat('0.' + fr) * 1000);
    return new Date(Date.UTC(+y, +mo - 1, +d, +h, +mi, +(se || 0), ms));
  }
  const dt = new Date(t);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

function fmtHM(d) {
  if (!d || Number.isNaN(d.getTime())) return '';
  return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
}

/* Nanshan Observatory local time — FIXED UTC+8 offset, not the browser's
   timezone. Only the Per-night tab uses these; elsewhere UTC is kept. */
const LOCAL_OFFSET_MS = 8 * 3600000;

function toLocal(d) {
  return new Date(d.getTime() + LOCAL_OFFSET_MS);
}

function fmtLocalHM(d) {
  if (!d || Number.isNaN(d.getTime())) return '';
  const t = toLocal(d);
  return `${pad2(t.getUTCHours())}:${pad2(t.getUTCMinutes())}`;
}

function fmtLocal(d, withSec) {
  if (!d || Number.isNaN(d.getTime())) return '—';
  const t = toLocal(d);
  const base = `${t.getUTCFullYear()}-${pad2(t.getUTCMonth() + 1)}-${pad2(t.getUTCDate())} ` +
               `${pad2(t.getUTCHours())}:${pad2(t.getUTCMinutes())}`;
  return withSec ? `${base}:${pad2(t.getUTCSeconds())}` : base;
}

function toISODate(d) {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

function truncate(s, n) {
  s = String(s == null ? '' : s);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

const msToHours = (ms) => ms / 3.6e6;

function diffDays(a, b) {
  return Math.round((new Date(b) - new Date(a)) / 86400000);
}

function sumNightHours(night) {
  return (night.blocks || []).reduce((sum, b) => {
    const s = parseUtc(b.start_utc), e = parseUtc(b.end_utc);
    return sum + (s && e ? msToHours(e - s) : 0);
  }, 0);
}

/* Dark-window duration in hours (UTC timestamps; a duration, so timezone is
   irrelevant). null when the night has no window (cloudy / missing). */
function availableHours(night) {
  const s = parseUtc(night && night.dark_start_utc);
  const e = parseUtc(night && night.dark_end_utc);
  if (!s || !e) return null;
  const h = msToHours(e - s);
  return h >= 0 ? h : null;
}

function fmtHours(h) {
  const v = Number(h);
  return Number.isNaN(v) ? '—' : `${v.toFixed(1)} h`;
}

/* ---------------- state ---------------- */

const state = {
  rows: [],               // editable table rows (values kept as strings)
  rowErrors: {},          // row index -> [messages]
  serverErrors: {},       // row index -> [messages] from /api/targets/parse
  sourceLabel: '',
  jobId: null,
  result: null,
  pollTimer: null,
  running: false,
  pollFails: 0,
  activeTab: 'overview',
  nightIdx: 0,
  tracksCache: {},        // night index -> /night/{i}/tracks response
  shuttingDown: false,    // exit confirmed: stop polling, suppress error surfacing
};

/* ---------------- API helpers (all fetches live here) ---------------- */

async function request(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (err) {
    throw new Error('Network error contacting the server: ' + err.message);
  }
  if (!res.ok) {
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON error body */ }
    throw new Error(extractError(data, res.status));
  }
  return res.json();
}

function extractError(data, status) {
  const d = data && data.detail !== undefined ? data.detail : data;
  if (typeof d === 'string') return d;
  if (Array.isArray(d) && d.length) {
    return d.map((e) =>
      (e.loc ? e.loc.join('.') + ': ' : '') + (e.msg || e.message || JSON.stringify(e))
    ).join('; ');
  }
  if (d && typeof d === 'object') {
    if (typeof d.message === 'string') return d.message;
    if (typeof d.error === 'string') return d.error;
  }
  return 'Server error (HTTP ' + status + ').';
}

function postJSON(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
}

/* Contract: POST /api/targets/parse takes JSON {"csv": "<raw csv text>"}. */
async function apiParseTargets(csvText) {
  return postJSON('/api/targets/parse', { csv: csvText });
}

async function apiStartSchedule(body) { return postJSON('/api/schedule', body); }
async function apiGetJob(id) { return request('/api/schedule/' + encodeURIComponent(id)); }
async function apiGetTracks(jobId, nightIdx) {
  return request('/api/schedule/' + encodeURIComponent(jobId) + '/night/' + nightIdx + '/tracks');
}
async function apiShutdown() { return postJSON('/api/shutdown', {}); }
async function apiCancelJob(id) {
  return request('/api/schedule/' + encodeURIComponent(id) + '/cancel', { method: 'POST' });
}

/* ---------------- Plotly theme ---------------- */

const CLEAR_COL = '#46c289';
const AMBER_COL = '#e3b341';

/* Stable marker shapes for the sky map's group legend (each has an open
   variant for un-completed targets). */
const GROUP_SYMBOLS = [
  'circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross',
  'x', 'star', 'pentagon', 'hexagon', 'octagon', 'square-cross'
];

/* The exact 11-stop plasma colorscale Plotly uses — replicated here so marker
   EDGE colors (open markers) can be precomputed as literal rgb() strings.
   This avoids driving scattergeo's marker.line.colorscale with numeric
   arrays, which is fragile across Plotly builds. */
const PLASMA_STOPS = [
  [0.0, '#0d0887'], [0.111111, '#46039f'], [0.222222, '#7201a8'],
  [0.333333, '#9c179e'], [0.444444, '#bd3786'], [0.555556, '#d8576b'],
  [0.666667, '#ed7953'], [0.777778, '#fb9f3a'], [0.888889, '#fdca26'], [1.0, '#f0f921']
];

function plasmaColor(t) {
  const v = Math.max(0, Math.min(1, t));
  for (let i = 0; i < PLASMA_STOPS.length - 1; i++) {
    const [t0, c0] = PLASMA_STOPS[i], [t1, c1] = PLASMA_STOPS[i + 1];
    if (v >= t0 && v <= t1) {
      const f = (v - t0) / (t1 - t0);
      const r0 = parseInt(c0.slice(1, 3), 16), r1 = parseInt(c1.slice(1, 3), 16);
      const g0 = parseInt(c0.slice(3, 5), 16), g1 = parseInt(c1.slice(3, 5), 16);
      const b0 = parseInt(c0.slice(5, 7), 16), b1 = parseInt(c1.slice(5, 7), 16);
      const r = Math.round(r0 + (r1 - r0) * f);
      const g = Math.round(g0 + (g1 - g0) * f);
      const b = Math.round(b0 + (b1 - b0) * f);
      return 'rgb(' + r + ',' + g + ',' + b + ')';
    }
  }
  return PLASMA_STOPS[PLASMA_STOPS.length - 1][1];
}

/* Wrap a longitude into [-180, 180]. The backend sends RA in 0..360 so
   lon = -ra can reach -360; normalizing client-side guarantees every marker
   lands inside the Aitoff domain regardless of Plotly build behavior. */
function wrapLon(l) {
  return ((((l + 180) % 360) + 360) % 360) - 180;
}

const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: { format: 'png', filename: 'amase-chart', scale: 2 }
};

function baseLayout(extra) {
  const l = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: '-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            color: '#aeb8c4', size: 12 },
    margin: { l: 64, r: 16, t: 24, b: 44 },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.08)',
      linecolor: 'rgba(255,255,255,0.15)', tickfont: { size: 11 }
    },
    yaxis: {
      gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.08)',
      linecolor: 'rgba(255,255,255,0.15)', tickfont: { size: 11 }
    },
    hoverlabel: {
      bgcolor: '#1a222d', bordercolor: '#2c3745', font: { color: '#e6ebf2', size: 12 }
    }
  };
  return Object.assign(l, extra || {});
}

/* ============================================================
   Targets table
   ============================================================ */

const FIELDS = [
  { key: 'name',     label: 'Name',     num: false, ph: 'e.g. NGC4258', title: '' },
  { key: 'ra',       label: 'RA',       num: true,  ph: '0-360',         title: 'Right ascension, degrees (0-360)' },
  { key: 'dec',      label: 'Dec',      num: true,  ph: '-90 to 90',     title: 'Declination, degrees (-90 to 90)' },
  { key: 'priority', label: 'Priority', num: true,  ph: '>= 1',          title: 'Integer >= 1' },
  { key: 'exp_time', label: 'Exp time', num: true,  ph: 's, <= 3600',    title: 'Exposure time in seconds (<= 3600)' },
  { key: 'n_dither', label: 'N dither', num: true,  ph: '1/3/9/27',      title: 'Must be 1, 3, 9 or 27' },
  { key: 'n_set',    label: 'N set',    num: true,  ph: '>= 1',          title: 'Integer >= 1' },
  { key: 'group',    label: 'Group',    num: false, ph: 'optional',      title: '' }
];

function validateRow(r) {
  const m = [];
  if (!String(r.name || '').trim()) m.push('Name required');
  const ra = parseNum(r.ra);
  if (ra === null || ra < 0 || ra > 360) m.push('RA must be 0-360');
  const dec = parseNum(r.dec);
  if (dec === null || dec < -90 || dec > 90) m.push('Dec must be -90 to 90');
  const pr = parseNum(r.priority);
  if (pr === null || !Number.isInteger(pr) || pr < 1) m.push('Priority: integer >= 1');
  const ex = parseNum(r.exp_time);
  if (ex === null || ex <= 0 || ex > 3600) m.push('Exp time: 1-3600 s');
  const nd = parseNum(r.n_dither);
  if (nd === null || ![1, 3, 9, 27].includes(nd)) m.push('N dither must be 1, 3, 9 or 27');
  const ns = parseNum(r.n_set);
  if (ns === null || !Number.isInteger(ns) || ns < 1) m.push('N set: integer >= 1');
  return m;
}

/* Prefer server parse messages while a row is untouched; fall back to the
   client-side validator after the user edits it. */
function computeRowErrors() {
  const out = {};
  state.rows.forEach((r, i) => {
    const srv = state.serverErrors[i];
    if (srv && srv.length) out[i] = srv;
    else {
      const client = validateRow(r);
      if (client.length) out[i] = client;
    }
  });
  state.rowErrors = out;
}

function buildRow(r, i) {
  const tr = document.createElement('tr');
  tr.className = 'target-row';
  tr.dataset.idx = i;
  FIELDS.forEach((f) => {
    const td = document.createElement('td');
    const inp = document.createElement('input');
    inp.dataset.field = f.key;
    if (f.num) inp.dataset.num = '1';
    inp.placeholder = f.ph || '';
    inp.title = f.title || f.label;
    inp.value = r[f.key] == null ? '' : String(r[f.key]);
    inp.setAttribute('autocomplete', 'off');
    inp.setAttribute('spellcheck', 'false');
    td.appendChild(inp);
    tr.appendChild(td);
  });
  const tdA = document.createElement('td');
  tdA.className = 'col-action';
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'icon-btn';
  btn.textContent = '✕';
  btn.dataset.idx = i;
  btn.title = 'Delete row';
  tdA.appendChild(btn);
  tr.appendChild(tdA);
  return tr;
}

function refreshRowFeedback(idx) {
  const tbody = $('#targetsBody');
  const tr = tbody.querySelector('tr.target-row[data-idx="' + idx + '"]');
  if (!tr) return;
  const msgs = state.rowErrors[idx] || [];
  tr.classList.toggle('has-error', msgs.length > 0);
  let er = tbody.querySelector('tr.row-error[data-idx="' + idx + '"]');
  if (msgs.length) {
    if (!er) {
      er = document.createElement('tr');
      er.className = 'row-error';
      er.dataset.idx = idx;
      tr.insertAdjacentElement('afterend', er);
    }
    er.innerHTML = '<td colspan="' + (FIELDS.length + 1) + '">' +
      '<span class="err-msg">' + esc(msgs.join(' · ')) + '</span></td>';
  } else if (er) {
    er.remove();
  }
}

function renderTargetsTable() {
  const tbody = $('#targetsBody');
  tbody.innerHTML = '';
  state.rows.forEach((r, i) => tbody.appendChild(buildRow(r, i)));
  computeRowErrors();
  state.rows.forEach((_, i) => refreshRowFeedback(i));
  updateTargetsStatus();
  updateRunState();
}

function updateTargetsStatus() {
  const n = state.rows.length;
  const errCount = Object.keys(state.rowErrors).length;
  let text;
  if (!n) text = 'No targets yet — upload a CSV or load the example.';
  else text = n + ' target' + (n === 1 ? '' : 's') + ' · ' +
    (errCount ? errCount + ' row' + (errCount === 1 ? '' : 's') + ' with errors' : 'all rows valid');
  if (state.sourceLabel) text += ' · ' + state.sourceLabel;
  const el = $('#targetsStatus');
  el.textContent = text;
  el.classList.toggle('error', errCount > 0);
}

function addRow() {
  state.rows.push({ name: '', ra: '', dec: '', priority: '1', exp_time: '300', n_dither: '1', n_set: '1', group: '' });
  renderTargetsTable();
  const rows = $$('#targetsBody tr.target-row');
  const last = rows[rows.length - 1];
  if (last) last.querySelector('input').focus();
}

async function handleParse(csvText, sourceLabel) {
  const res = await apiParseTargets(csvText);
  const rows = (res.rows || []).map((r) => ({
    name: r.name != null ? String(r.name) : '',
    ra: r.ra != null ? String(r.ra) : '',
    dec: r.dec != null ? String(r.dec) : '',
    priority: r.priority != null ? String(r.priority) : '',
    exp_time: r.exp_time != null ? String(r.exp_time) : '',
    n_dither: r.n_dither != null ? String(r.n_dither) : '',
    n_set: r.n_set != null ? String(r.n_set) : '',
    group: r.group != null ? String(r.group) : ''
  }));
  state.serverErrors = {};
  const unmapped = [];
  (res.errors || []).forEach((er) => {
    const li = (er.line != null ? er.line : 0) - 1; // 1-based CSV line -> row index
    if (li >= 0 && li < rows.length) {
      state.serverErrors[li] = (state.serverErrors[li] || []).concat([er.message || 'Invalid row']);
    } else {
      unmapped.push('line ' + (er.line || '?') + ': ' + (er.message || 'Invalid row'));
    }
  });
  if (unmapped.length) {
    showError(unmapped.length + ' line' + (unmapped.length === 1 ? '' : 's') +
      ' could not be mapped to a table row: ' + unmapped.join('; '));
  } else {
    hideError();
  }
  state.rows = rows;
  state.sourceLabel = sourceLabel;
  renderTargetsTable();
}

/* ============================================================
   Dates & parameters
   ============================================================ */

function dateModeValue() {
  const el = document.querySelector('input[name="dateMode"]:checked');
  return el ? el.value : 'single';
}

function getDateRange() {
  const mode = dateModeValue();
  if (mode === 'single') {
    const d = $('#singleDate').value;
    return { start: d, end: d };
  }
  return { start: $('#rangeStart').value, end: $('#rangeEnd').value };
}

function validateDates() {
  const { start, end } = getDateRange();
  if (!start || !end) return 'Select start and end dates.';
  if (end < start) return 'End date is before start date.';
  const nights = diffDays(start, end) + 1;
  if (nights > 31) return 'Date range exceeds 31 nights (' + nights + ').';
  return null;
}

function validateParams() {
  const m = [];
  const cp = parseNum($('#clearProb').value);
  if (cp === null || cp < 0 || cp > 1) m.push('Clear prob must be between 0 and 1.');
  const sd = parseNum($('#seed').value);
  if (sd === null || !Number.isInteger(sd)) m.push('Seed must be an integer.');
  const tl = parseNum($('#timeLimit').value);
  if (tl === null || tl <= 0) m.push('Time limit must be > 0.');
  const ep = parseNum($('#eps').value);
  if (ep === null || ep <= 0) m.push('Eps must be > 0.');
  const ga = parseNum($('#gamma').value);
  if (ga === null || ga < 0) m.push('Gamma must be >= 0.');
  const al = parseNum($('#alpha').value);
  if (al === null || al < 0 || al > 1) m.push('Alpha must be between 0 and 1.');
  return m;
}

function updateRunState() {
  const reasons = [];
  if (state.rows.length === 0) reasons.push('Add at least one target.');
  const errCount = Object.keys(state.rowErrors).length;
  if (errCount) reasons.push('Fix errors in ' + errCount + ' row' + (errCount === 1 ? '' : 's') + '.');
  const dateErr = validateDates();
  if (dateErr) reasons.push(dateErr);
  const dateMsg = $('#dateMsg');
  if (dateErr) {
    dateMsg.textContent = dateErr;
    dateMsg.classList.remove('ok');
  } else {
    dateMsg.textContent = '';
  }
  const paramErrs = validateParams();
  reasons.push.apply(reasons, paramErrs);

  const btn = $('#runBtn');
  btn.disabled = state.running || reasons.length > 0;
  const msgEl = $('#runMsg');
  if (state.running) {
    msgEl.textContent = 'Scheduling in progress…';
    msgEl.classList.add('ok');
  } else if (reasons.length) {
    msgEl.textContent = reasons.join(' ');
    msgEl.classList.remove('ok');
  } else {
    msgEl.textContent = 'Ready to schedule.';
    msgEl.classList.add('ok');
  }
}

/* ============================================================
   Run flow: start -> poll -> done/error/cancelled
   ============================================================ */

function showError(msg) {
  if (state.shuttingDown) return;   /* the server is going down — no error banners */
  const b = $('#errorBanner');
  b.textContent = msg;
  b.hidden = false;
}

function hideError() { $('#errorBanner').hidden = true; }

function buildTargetPayload() {
  return state.rows.map((r) => ({
    name: (r.name || '').trim(),
    ra: (r.ra || '').trim(),   // string: sexagesimal like 12h30m00s is valid
    dec: (r.dec || '').trim(),
    priority: parseNum(r.priority),
    exp_time: parseNum(r.exp_time),
    n_dither: parseNum(r.n_dither),
    n_set: parseNum(r.n_set),
    group: (r.group || '').trim()
  }));
}

async function startSchedule() {
  hideError();
  updateRunState();
  if ($('#runBtn').disabled) return;

  const { start, end } = getDateRange();
  const body = {
    targets: buildTargetPayload(),
    start: start,
    end: end,
    clear_prob: parseNum($('#clearProb').value),
    seed: parseNum($('#seed').value),
    time_limit: parseNum($('#timeLimit').value),
    eps: parseNum($('#eps').value),
    gamma: parseNum($('#gamma').value),
    alpha: parseNum($('#alpha').value)
  };

  try {
    const res = await apiStartSchedule(body);
    state.jobId = res.job_id;
    state.result = null;
    state.running = true;
    state.pollFails = 0;
    $('#results').hidden = true;
    $('#runPanel').hidden = false;
    setRunPanelState('running', 'Scheduling night…', 'Night 0/0', 0);
    $('#cancelBtn').disabled = false;
    $('#cancelBtn').textContent = 'Cancel';
    updateRunState();
    pollJob();
  } catch (err) {
    showError(err.message);
  }
}

function setRunPanelState(klass, status, label, pct) {
  const panel = $('#runPanel');
  panel.classList.remove('state-error', 'state-done');
  if (klass) panel.classList.add('state-' + klass);
  $('#runStatus').textContent = status;
  $('#progressLabel').textContent = label;
  $('#progressLabel').style.color = '';
  $('#progressFill').style.width = Math.max(0, Math.min(100, pct)) + '%';
}

function updateProgress(p) {
  if (!p) { setRunPanelState('running', 'Scheduling night…', 'Night 0/0', 0); return; }
  const n = p.n_nights || 0;
  const i = p.night_idx || 0;
  const date = p.date || '';
  const pct = n ? (i / n) * 100 : 0;
  const label = 'Night ' + i + '/' + n + (date ? ' (' + date + ')' : '');
  setRunPanelState('running', 'Scheduling…', label, pct);
}

async function pollJob() {
  clearTimeout(state.pollTimer);
  if (state.shuttingDown || !state.running) return;

  let res;
  try {
    res = await apiGetJob(state.jobId);
    state.pollFails = 0;
  } catch (err) {
    if (state.shuttingDown) return;
    state.pollFails += 1;
    if (state.pollFails >= 5) {
      setRunPanelState('error', 'Lost contact with the server: ' + err.message, '', 0);
      state.running = false;
      $('#cancelBtn').disabled = true;
      return;
    }
    state.pollTimer = setTimeout(pollJob, 1500);
    return;
  }

  const status = res.status;
  if (status === 'running') {
    updateProgress(res.progress);
    state.pollTimer = setTimeout(pollJob, 1000);
  } else if (status === 'done') {
    state.running = false;
    state.result = res.result || {};
    $('#runPanel').hidden = true;
    renderResults();
    updateRunState();
  } else if (status === 'cancelled') {
    state.running = false;
    $('#cancelBtn').disabled = true;
    setRunPanelState('done', 'Scheduling cancelled.', '', 0);
    $('#cancelBtn').textContent = 'Cancel';
    updateRunState();
  } else if (status === 'error') {
    state.running = false;
    $('#cancelBtn').disabled = true;
    setRunPanelState('error', 'Scheduling failed.', '', 0);
    $('#progressLabel').textContent = res.error || 'Unknown error.';
    $('#progressLabel').style.color = 'var(--red-soft)';
    updateRunState();
  } else {
    state.pollTimer = setTimeout(pollJob, 1000);
  }
}

async function cancelJob() {
  if (!state.jobId || !state.running) return;
  $('#cancelBtn').disabled = true;
  $('#cancelBtn').textContent = 'Cancelling…';
  try {
    await apiCancelJob(state.jobId);
  } catch (err) {
    $('#cancelBtn').disabled = false;
    $('#cancelBtn').textContent = 'Cancel';
    showError('Cancel request failed: ' + err.message);
  }
}

/* ============================================================
   Results
   ============================================================ */

function renderResults() {
  const result = state.result;
  if (!result) return;

  hideError();

  /* Summary strip */
  const s = result.summary || {};
  const nights = result.nights || [];
  const progress = result.progress || [];
  const nClear = s.n_clear != null ? s.n_clear
    : nights.filter((n) => n.clear).length;
  const nNights = s.n_nights != null ? s.n_nights : nights.length;
  const nCompleted = s.n_completed != null ? s.n_completed
    : progress.filter((p) => (p.fraction || 0) >= 0.999).length;
  const totalMin = s.total_obs_min != null ? s.total_obs_min
    : nights.reduce((a, n) => a + sumNightHours(n) * 60, 0);

  $('#summaryStrip').innerHTML =
    '<div class="stat"><div class="value">' + nClear + '/' + nNights + '</div>' +
      '<div class="label">Clear nights</div></div>' +
    '<div class="stat"><div class="value">' + nCompleted + '/' + progress.length + '</div>' +
      '<div class="label">Targets completed</div></div>' +
    '<div class="stat"><div class="value">' + fmtHours(totalMin / 60) + '</div>' +
      '<div class="label">Total observing time</div></div>';

  /* Capacity warning (string or null per contract) */
  const cw = result.capacity_warning;
  const cwEl = $('#capacityWarning');
  if (cw && typeof cw === 'string') {
    cwEl.textContent = cw;
    cwEl.hidden = false;
  } else {
    cwEl.hidden = true;
  }

  /* No clear nights note */
  $('#noClearNote').hidden = nClear > 0 || nights.length === 0;

  /* Meta line */
  const meta = [];
  if (result.start_date && result.end_date) meta.push(result.start_date + ' → ' + result.end_date);
  if (result.seed != null) meta.push('seed ' + result.seed);
  if (result.clear_prob != null) meta.push('clear_prob ' + result.clear_prob);
  if (state.jobId) meta.push('job ' + state.jobId);
  $('#resultsMeta').textContent = meta.join(' · ');

  /* Reset per-night selection and tracks cache for the new result */
  state.nightIdx = 0;
  state.tracksCache = {};

  $('#results').hidden = false;
  activateTab('overview');
}

function activateTab(name) {
  state.activeTab = name;
  $$('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab-panel').forEach((p) => { p.hidden = (p.id !== 'panel-' + name); });
  if (!state.result) return;
  if (name === 'overview') { renderOverviewTables(); renderOverviewMap(); }
  else if (name === 'pernight') renderPerNightPanel();
  else if (name === 'download') renderDownloadPanel();
}

function plotReady() {
  if (typeof window.Plotly === 'undefined') {
    $$('.chart').forEach((c) => {
      c.innerHTML = '<div class="empty">Plotly could not be loaded from the CDN — charts are unavailable.</div>';
    });
    return false;
  }
  return true;
}

/* ----- Overview tables ----- */

function progressRowHtml(p) {
  const frac = Math.max(0, Math.min(1, p.fraction || 0));
  const pct = Math.round(frac * 100);
  const barColor = frac >= 0.999 ? CLEAR_COL : frac > 0 ? AMBER_COL : '#3a434f';
  return '<td>' + esc(p.target || '') + '</td>' +
    '<td>' + (p.required != null ? esc(String(p.required)) : '—') + '</td>' +
    '<td>' + (p.done != null ? esc(String(p.done)) : '—') + '</td>' +
    '<td class="frac-cell"><span class="frac-bar"><span class="frac-fill" style="width:' +
      pct + '%;background:' + barColor + '"></span></span><span class="frac-text">' +
      pct + '%</span></td>' +
    '<td>' + (p.nights_observed != null ? esc(String(p.nights_observed)) : '—') + '</td>' +
    '<td>' + fmtHours((p.obs_time_min || 0) / 60) + '</td>';
}

function renderOverviewTables() {
  const result = state.result;
  if (!result) return;

  /* Target completion */
  const cBody = $('#overviewCompletionBody');
  cBody.innerHTML = '';
  const prog = result.progress || [];
  if (!prog.length) {
    cBody.innerHTML = '<tr><td colspan="6" class="empty">No target progress available.</td></tr>';
  } else {
    const sorted = prog.slice().sort((a, b) =>
      (b.fraction || 0) - (a.fraction || 0) ||
      String(a.target || '').localeCompare(String(b.target || '')));
    sorted.forEach((p) => {
      const tr = document.createElement('tr');
      tr.innerHTML = progressRowHtml(p);
      cBody.appendChild(tr);
    });
  }

  /* Observed hours per night */
  const hBody = $('#overviewHoursBody');
  hBody.innerHTML = '';
  const nights = result.nights || [];
  if (!nights.length) {
    hBody.innerHTML = '<tr><td colspan="4" class="empty">No nights available.</td></tr>';
  } else {
    nights.forEach((n) => {
      const h = sumNightHours(n);
      const avail = availableHours(n);
      const availTxt = avail === null ? '—' : avail.toFixed(1);
      /* restrained cue: a clear night with nothing observed */
      const obsMuted = n.clear && h === 0 ? ' class="cell-muted"' : '';
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + esc(n.date || '') + '</td>' +
        '<td>' + (n.clear ? 'Clear' : 'Cloudy') + '</td>' +
        '<td>' + availTxt + '</td>' +
        '<td' + obsMuted + '>' + h.toFixed(1) + '</td>';
      hBody.appendChild(tr);
    });
  }
}

/* ----- Overview: Aitoff sky-distribution map ----- */

function skyHover(p) {
  const ra = Number(p.ra_deg) || 0;
  const dec = Number(p.dec_deg) || 0;
  const hrs = Number(p.required_hours) || 0;
  const req = p.required != null ? p.required : 0;
  const done = p.done != null ? p.done : 0;
  const frac = p.fraction != null ? p.fraction : 0;
  const grp = (p.group || '') === '' ? 'ungrouped' : p.group;
  return '<b>' + esc(p.target || '') + '</b><br>' +
    'group: ' + esc(grp) + '<br>' +
    'RA ' + (ra / 15).toFixed(2) + ' h · Dec ' + dec.toFixed(2) + '°<br>' +
    'required ' + hrs.toFixed(1) + ' h · ' + done + '/' + req + ' (' +
    Math.round(frac * 100) + '%)<extra></extra>';
}

function renderOverviewMap() {
  const chart = $('#chartSkyMap');
  const empty = $('#skyMapEmpty');
  if (!plotReady()) return;
  const result = state.result;
  if (!result) return;
  const prog = result.progress || [];

  if (!prog.length) {
    empty.textContent = 'No targets to map.';
    empty.hidden = false;
    chart.hidden = true;
    return;
  }
  empty.hidden = true;
  chart.hidden = false;

  /* Stable group -> symbol map from first-appearance order (deterministic, so
     re-renders keep the same shapes). "" group -> "ungrouped". */
  const groups = [];
  const groupSymbols = {};
  prog.forEach((p) => {
    const g = (p.group || '') === '' ? 'ungrouped' : p.group;
    if (groups.indexOf(g) === -1) {
      groups.push(g);
      groupSymbols[g] = GROUP_SYMBOLS[(groups.length - 1) % GROUP_SYMBOLS.length];
    }
  });

  /* Log-hours color domain, shared by all marker traces */
  const logHours = prog.map((p) => Math.log10(Math.max(Number(p.required_hours) || 1e-3, 1e-3)));
  let cmin = Math.min.apply(null, logHours);
  let cmax = Math.max.apply(null, logHours);
  if (!isFinite(cmin) || cmin === cmax) { cmin = 0; cmax = Math.max(cmax, cmin + 1); }

  /* Colorbar hugged against the Aitoff ellipse: x just right of the map's
     widest point (0.5 + semi-width, in plot-area paper coords), full plot
     height (len 1), vertically centered. The ellipse is fitted height-first,
     so its paper semi-width = gs.h / gs.w. */
  const gsW = (chart.clientWidth || 1200) - 48;   // margins l=24 r=24
  const gsH = 520 - 44 - 110;                     // CSS height 520 - t=44 b=110
  const cbX = 0.5 + Math.min(gsH / gsW, 0.5) + 0.018;
  const colorbar = {
    title: { text: 'required hours (log)', side: 'right', font: { size: 10.5 } },
    x: cbX, xanchor: 'left', xpad: 4, thickness: 12,
    len: 1.0, y: 0.5, yanchor: 'middle',
    tickfont: { size: 10 }, outlinewidth: 0
  };  const tickLogs = [];
  for (let l = Math.ceil(cmin); l <= Math.floor(cmax); l++) tickLogs.push(l);
  if (tickLogs.length) {
    colorbar.tickvals = tickLogs;
    colorbar.ticktext = tickLogs.map((l) => String(Math.pow(10, l)));
  }
  const edgeColor = (lv) => plasmaColor((lv - cmin) / (cmax - cmin));

  const traces = [];
  const partialByGroup = {};

  groups.forEach((grp) => {
    const base = groupSymbols[grp];
    const open = base + '-open';
    const members = prog.filter((p) => ((p.group || '') === '' ? 'ungrouped' : p.group) === grp);
    const lons = [], lats = [], syms = [], logVals = [], lineW = [], hovers = [];
    members.forEach((p) => {
      const f = p.fraction || 0;
      const done = p.done || 0;
      const h = Math.max(Number(p.required_hours) || 1e-3, 1e-3);
      const lv = Math.log10(h);
      lons.push(wrapLon(-(Number(p.ra_deg) || 0)));
      lats.push(Number(p.dec_deg) || 0);
      logVals.push(lv);
      hovers.push(skyHover(p));
      if (f >= 0.999) { syms.push(base); lineW.push(0); }
      else {
        syms.push(open); lineW.push(1.25);
        if (done > 0) {
          if (!partialByGroup[grp]) partialByGroup[grp] = { lon: [], lat: [], log: [] };
          partialByGroup[grp].lon.push(lons[lons.length - 1]);
          partialByGroup[grp].lat.push(lats[lats.length - 1]);
          partialByGroup[grp].log.push(lv);
        }
      }
    });
    traces.push({
      type: 'scattergeo', mode: 'markers', geo: 'geo',
      lon: lons, lat: lats,
      marker: {
        size: 12,
        symbol: syms,
        color: logVals,
        /* explicit colorscale array — scattergeo ignores the named 'plasma'
           string in plotly 2.35.2 and falls back to autocolorscale */
        colorscale: PLASMA_STOPS,
        autocolorscale: false,
        cmin: cmin, cmax: cmax,
        /* literal rgb() edge colors (plasma-mapped) instead of a numeric
           marker.line.color + marker.line.colorscale on scattergeo */
        line: { color: logVals.map(edgeColor), width: lineW }
      },
      hovertemplate: hovers, hoverlabel: { namelength: -1 },
      name: grp, legendgroup: grp, showlegend: true
    });
  });

  /* Dedicated colorbar carrier: an invisible single marker that always keeps
     the colorbar alive. A colorbar attached to a group trace would vanish when
     that group is toggled off via the legend, so it lives here instead —
     legendless, opacity 0, size 0, and never hidden. */
  traces.push({
    type: 'scattergeo', mode: 'markers', geo: 'geo',
    lon: [0], lat: [-89],
    marker: {
      size: 0, symbol: 'circle',
      color: [cmax],
      colorscale: PLASMA_STOPS, autocolorscale: false,
      cmin: cmin, cmax: cmax,
      opacity: 0,
      showscale: true, colorbar: colorbar
    },
    hoverinfo: 'skip', showlegend: false
  });

  /* Partially completed targets: small filled inner dots. One overlay trace
     PER group, legendgroup-bound to the group's name, so they show/hide with
     their own legend entry — Plotly's single-click toggle and double-click
     isolate/restore both act on every trace sharing a legendgroup. They have
     no legend entry of their own; groups without partial targets get none. */
  Object.keys(partialByGroup).forEach((grp) => {
    const p = partialByGroup[grp];
    traces.push({
      type: 'scattergeo', mode: 'markers', geo: 'geo',
      lon: p.lon, lat: p.lat,
      marker: {
        size: 3.5, symbol: 'circle',
        color: p.log, colorscale: PLASMA_STOPS, autocolorscale: false, cmin: cmin, cmax: cmax,
        line: { width: 0 }
      },
      legendgroup: grp, name: grp,
      hoverinfo: 'skip', showlegend: false
    });
  });

  /* red dashed dec-limit line (never above 30° at Nanshan, lat 43.472°N) */
  const decLimit = 30 - (90 - 43.472);
  const lineLon = [];
  for (let lon = -179.5; lon <= 179.5; lon += 10) lineLon.push(lon);
  traces.push({
    type: 'scattergeo', mode: 'lines', geo: 'geo',
    lon: lineLon, lat: lineLon.map(() => decLimit),
    line: { color: '#e05a52', width: 1.5, dash: 'dash' },
    name: 'never above 30° at Nanshan', hoverinfo: 'skip'
  });

  /* legend-only glyphs explaining the completion fill encoding */
  [
    { name: 'completed', symbol: 'circle' },
    { name: 'partially completed', symbol: 'circle-open-dot' },
    { name: 'not observed', symbol: 'circle-open' }
  ].forEach((d) => {
    traces.push({
      type: 'scattergeo', mode: 'markers', geo: 'geo',
      lon: [0], lat: [0],
      marker: { size: 12, symbol: d.symbol, color: '#dbe2ea', line: { width: 0 } },
      name: d.name, legendgroup: d.name, showlegend: true,
      visible: 'legendonly', hoverinfo: 'skip'
    });
  });

  /* title + subtitle */
  const n = prog.length;
  const m = prog.filter((p) => (p.fraction || 0) >= 0.999).length;
  const x = prog.reduce((a, p) => a + (p.done || 0), 0);
  const y = prog.reduce((a, p) => a + (p.required || 0), 0);
  const z = y ? Math.round((x / y) * 100) : 0;
  const titleText = 'AMASE-P target sky distribution (N=' + n + ') — ' +
    (result.start_date || '') + ' → ' + (result.end_date || '') +
    '\ncompleted ' + m + '/' + n + ' targets · ' + x + '/' + y + ' exposures (' + z + '%)';

  const layout = baseLayout({
    margin: { l: 24, r: 24, t: 44, b: 110 },
    title: { text: titleText, font: { size: 13 } },
    /* legend BELOW the map (horizontal, centered) so it never overlaps the
       plot area or the right-side colorbar */
    legend: {
      x: 0.5, y: -0.12, xanchor: 'center', yanchor: 'top', orientation: 'h',
      bgcolor: 'rgba(11,15,20,0.6)', bordercolor: 'rgba(255,255,255,0.15)',
      borderwidth: 1, font: { size: 10 }
    },
    geo: {
      projection: { type: 'aitoff' },
      bgcolor: 'rgba(0,0,0,0)',
      showland: false, showocean: false, showlakes: false, showrivers: false,
      showcoastlines: false, showcountries: false,
      showframe: true, framecolor: 'rgba(255,255,255,0.28)', framewidth: 1,
      lonaxis: {
        showgrid: true, gridcolor: 'rgba(255,255,255,0.10)', griddash: 'dot', gridwidth: 1,
        dtick: 30
      },
      lataxis: {
        showgrid: true, gridcolor: 'rgba(255,255,255,0.10)', griddash: 'dot', gridwidth: 1,
        dtick: 30
      }
    }
  });

  /* The three fill-encoding legend entries are display-only glyphs. They must
     NEVER become visible, including through Plotly's double-click isolate/
     restore cycle (restore sets every trace back to visible:true, which would
     resurrect the dummy markers). Handlers are attached AFTER the react
     promise resolves — the div only becomes an EventEmitter (.on) once Plotly
     has initialized it. */
  const plotPromise = Plotly.react(chart, traces, layout, PLOTLY_CONFIG);
  const done = (plotPromise && typeof plotPromise.then === 'function')
    ? plotPromise : Promise.resolve();
  done.then(() => {
    /* Correct the colorbar position from the ACTUAL rendered geometry: the
       provisional x was estimated from the container width, but Plotly's real
       plot-area size (gs) is authoritative. Keep the colorbar hugging the
       Aitoff ellipse's right edge with a small gap. */
    if (chart._fullLayout && chart._fullLayout._size && typeof Plotly.restyle === 'function') {
      const gs = chart._fullLayout._size;
      const gsW = Math.max(gs.w, 1);
      const gsH = Math.max(gs.h, 1);
      const correctX = 0.5 + Math.min(gsH / gsW, 0.5) + 0.018;
      const carrierIdx = (chart.data || []).findIndex((t) => t.marker && t.marker.showscale === true);
      if (carrierIdx !== -1 && Math.abs(correctX - colorbar.x) > 0.01) {
        Plotly.restyle(chart, { 'marker.colorbar.x': correctX }, [carrierIdx]);
      }
    }
    if (typeof chart.on === 'function' && !chart._skyLegendClickHooked) {
      chart._skyLegendClickHooked = true;

      const FILL_NAMES = ['completed', 'partially completed', 'not observed'];
      const isFillName = (name) => FILL_NAMES.indexOf(name) !== -1;
      /* evt.data is the FULL gd.data array; the clicked trace is identified by
         evt.curveNumber (fullData carries the trace name). Resolved by name at
         event time so indices stay correct even if trace order changes. */
      const clickedName = (evt) => {
        const idx = evt && evt.curveNumber;
        const tr = evt && (evt.fullData || evt.data);
        return tr && tr[idx] && tr[idx].name;
      };
      /* Force any fill dummy that is visible back to 'legendonly'. Called after
         a group double-click (restore) and after EVERY restyle on this chart so
         no interaction sequence can leave a dummy visible. */
      const rehideDummies = () => {
        if (typeof Plotly.restyle !== 'function') return;
        const indices = [];
        (chart.data || []).forEach((d, i) => {
          if (isFillName(d && d.name) && d.visible !== 'legendonly') indices.push(i);
        });
        if (indices.length) {
          Plotly.restyle(chart, 'visible', 'legendonly', indices);
        }
      };

      chart.on('plotly_legendclick', (evt) => {
        if (isFillName(clickedName(evt))) return false;
      });

      chart.on('plotly_legenddoubleclick', (evt) => {
        if (isFillName(clickedName(evt))) return false;   /* dummy: nothing */
        /* group double-click: let Plotly's isolate/restore proceed, then force
           the dummies back to legendonly (restore sets every trace visible) */
        setTimeout(rehideDummies, 0);
      });

      /* invariant guard across all restyle/redraw paths */
      chart.on('plotly_restyle', rehideDummies);
    }
  });
}

/* ----- Per-night: classic night figure (tracks + block Gantt) ----- */

function renderPerNightPanel() {
  const nights = (state.result.nights) || [];
  const sel = $('#nightSelect');
  sel.innerHTML = nights.map((n, i) =>
    '<option value="' + i + '">' + esc(n.date) +
    (n.clear ? '' : ' · cloudy') + '</option>').join('');

  if (!nights.length) {
    $('#ganttEmpty').textContent = 'No nights available.';
    $('#ganttEmpty').hidden = false;
    $('#chartTracks').hidden = true;
    $('#chartBlocks').hidden = true;
    $('#tracksLoading').hidden = true;
    renderBlocksTable(null);
    return;
  }
  const idx = Math.min(state.nightIdx, nights.length - 1);
  sel.value = idx;
  renderBlocksTable(nights[idx]);
  loadNightTracks(idx);
}

/* Fetch per-night tracks lazily, cached by night index. The Plotly divs
   (#chartTracks / #chartBlocks) are never touched via innerHTML — messages go
   to the sibling #ganttEmpty element, so Plotly can re-render safely. */
async function loadNightTracks(idx) {
  const nights = (state.result.nights) || [];
  const night = nights[idx];
  const trackDiv = $('#chartTracks');
  const blockDiv = $('#chartBlocks');
  const empty = $('#ganttEmpty');
  const loading = $('#tracksLoading');

  /* Cloudy night or missing night window: nothing to draw. */
  if (!night || !night.clear || !night.night_start_utc) {
    empty.textContent = 'No night data (weather loss or nothing to schedule).';
    empty.hidden = false;
    trackDiv.hidden = true;
    blockDiv.hidden = true;
    loading.hidden = true;
    return;
  }

  const cached = state.tracksCache[idx];
  if (cached) {
    loading.hidden = true;
    empty.hidden = true;
    trackDiv.hidden = false;
    blockDiv.hidden = false;
    renderNightFigure(cached, night);
    return;
  }

  loading.hidden = false;
  empty.hidden = true;
  trackDiv.hidden = true;
  blockDiv.hidden = true;

  try {
    const data = await apiGetTracks(state.jobId, idx);
    state.tracksCache[idx] = data;
    if (state.nightIdx !== idx) return;   // user switched away while loading
    loading.hidden = true;
    empty.hidden = true;
    trackDiv.hidden = false;
    blockDiv.hidden = false;
    renderNightFigure(data, night);
  } catch (err) {
    if (state.nightIdx !== idx) return;
    loading.hidden = true;
    trackDiv.hidden = true;
    blockDiv.hidden = true;
    empty.textContent = 'Could not load night tracks: ' + err.message;
    empty.hidden = false;
  }
}

function renderNightFigure(data, night) {
  if (!plotReady()) return;
  renderTracksPanel(data, night);
  renderBlocksPanel(data, night);
}

/* Hourly LOCAL time (UTC+8) tick values shared by both panels so their
   x axes align. The numeric positions stay hours-since-night-start; only the
   labels shift by the fixed +8h offset. */
function buildLocalTicks(nStart, totalH) {
  const tickEvery = totalH > 16 ? 2 : 1;
  const tickvals = [], ticktext = [];
  for (let h = 0; h <= totalH + 0.001; h += tickEvery) {
    tickvals.push(h);
    ticktext.push(fmtLocalHM(new Date(nStart.getTime() + h * 3.6e6)));
  }
  return { tickvals: tickvals, ticktext: ticktext };
}

/* Linearly interpolate a track's altitude at an arbitrary timestamp. */
function interpAlt(track, gridMs, ms) {
  const alts = track.alt || [];
  if (!alts.length) return null;
  if (ms <= gridMs[0]) return alts[0];
  if (ms >= gridMs[gridMs.length - 1]) return alts[alts.length - 1];
  for (let i = 0; i < gridMs.length - 1; i++) {
    if (ms >= gridMs[i] && ms <= gridMs[i + 1]) {
      const f = (ms - gridMs[i]) / (gridMs[i + 1] - gridMs[i]);
      return alts[i] + (alts[i + 1] - alts[i]) * f;
    }
  }
  return alts[alts.length - 1];
}

/* ---- Top panel: altitude tracks + scheduled-block highlights, limit line,
        twilight shading, LST axis. x is hours since night start (UTC). ---- */

function renderTracksPanel(data, night) {
  const wrap = $('#chartTracks');
  const nStart = parseUtc(data.night_start_utc) || parseUtc(night.night_start_utc);
  if (!nStart) return;
  const nEnd = parseUtc(data.night_end_utc) || parseUtc(night.night_end_utc);
  const totalH = Math.max(msToHours((nEnd || new Date(nStart.getTime() + 12 * 3.6e6)) - nStart), 0.5);

  const gridMs = (data.grid_utc || [])
    .map((iso) => parseUtc(iso))
    .filter((d) => d !== null)
    .map((d) => d.getTime());
  const hOf = (ms) => msToHours(ms - nStart.getTime());

  /* Only scheduled targets are drawn — unscheduled context tracks are
     filtered out client-side (the backend still sends them). */
  const tracks = (data.tracks || []).filter((t) => t.scheduled);
  const colors = data.colors || {};
  const altLimit = data.alt_limit_deg != null ? data.alt_limit_deg : 30;
  const localTicks = buildLocalTicks(nStart, totalH);
  const traces = [];

  tracks.forEach((t) => {
    const name = t.name || '';
    const ys = t.alt || [];
    const ht = gridMs.map((ms, i) => {
      const d = new Date(ms);
      return '<b>' + esc(name) + '</b><br>' + fmtLocalHM(d) + ' local (' + fmtHM(d) + ' UTC)<br>alt ' +
        (ys[i] != null ? Number(ys[i]).toFixed(1) : '—') + '°<extra></extra>';
    });
    traces.push({
      type: 'scatter', mode: 'lines',
      x: gridMs.map(hOf), y: ys,
      line: { color: colors[name] || AMBER_COL, width: 1 },
      hovertemplate: ht, hoverlabel: { namelength: -1 },
      name: name, showlegend: true
    });
  });

  /* Thick colored segments over each scheduled block, following its track.
     Each segment is slightly inset at both ends so adjacent same-target blocks
     keep a visible boundary (a thin notch of the underlying track). */
  const gridIndex = new Map();
  gridMs.forEach((ms, i) => gridIndex.set(ms, i));
  const segPadH = Math.max(2 / 60, totalH * 0.004);
  (night.blocks || []).forEach((b) => {
    const t = tracks.find((tr) => tr.name === b.target);
    const s = parseUtc(b.start_utc), e = parseUtc(b.end_utc);
    if (!t || !s || !e) return;
    const sMs = s.getTime(), eMs = e.getTime();
    const padMs = Math.min(segPadH * 3.6e6, (eMs - sMs) * 0.25);
    const xs = [hOf(sMs + padMs)], ys = [interpAlt(t, gridMs, sMs + padMs)];
    gridMs.forEach((ms) => {
      if (ms > sMs + padMs && ms < eMs - padMs) {
        xs.push(hOf(ms));
        ys.push(t.alt[gridIndex.get(ms)]);
      }
    });
    xs.push(hOf(eMs - padMs));
    ys.push(interpAlt(t, gridMs, eMs - padMs));
    traces.push({
      type: 'scatter', mode: 'lines',
      x: xs, y: ys,
      line: { color: colors[b.target] || AMBER_COL, width: 4 },
      hoverinfo: 'skip', showlegend: false
    });
  });

  /* Red dashed altitude limit line. */
  traces.push({
    type: 'scatter', mode: 'lines',
    x: [-0.1, totalH + 0.6], y: [altLimit, altLimit],
    line: { color: '#e05a52', width: 1.5, dash: 'dash' },
    name: altLimit + '° limit', hoverinfo: 'skip'
  });

  /* Twilight shading: gray vertical shapes over grid spans flagged true. */
  const tw = data.twilight || [];
  const shapes = [];
  if (tw.length && tw.length === gridMs.length) {
    let runStart = null;
    for (let i = 0; i <= tw.length; i++) {
      const on = i < tw.length && tw[i];
      if (on && runStart === null) runStart = i;
      else if (!on && runStart !== null) {
        const x0 = gridMs[runStart], x1 = gridMs[Math.min(i, gridMs.length - 1)];
        if (x1 > x0) {
          shapes.push({
            type: 'rect', xref: 'x', yref: 'y',
            x0: hOf(x0), x1: hOf(x1), y0: 0, y1: 90,
            fillcolor: 'rgba(148,155,166,0.16)', line: { width: 0 }, layer: 'below'
          });
        }
        runStart = null;
      }
    }
  }

  /* LST axis on top, overlaid on the same x domain. */
  const lstTickvals = [], lstTicktext = [];
  (data.lst_ticks || []).forEach((lt) => {
    const d = parseUtc(lt.utc);
    if (d) { lstTickvals.push(hOf(d.getTime())); lstTicktext.push(lt.label); }
  });

  const layout = baseLayout({
    margin: { l: 110, r: 16, t: 132, b: 4 },
    title: { text: 'AMASE-P schedule — ' + (data.date || night.date || ''), font: { size: 13 } },
    legend: {
      orientation: 'h', x: 0, y: 1.15, xanchor: 'left', yanchor: 'bottom',
      font: { size: 10.5 }
    },
    xaxis: {
      range: [-0.1, totalH + 0.6], tickvals: localTicks.tickvals, ticktext: localTicks.ticktext,
      showticklabels: false, showgrid: true,
      gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.15)', zeroline: false
    },
    yaxis: {
      range: [0, 90], title: { text: 'Altitude (deg)', font: { size: 11 } },
      gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.15)', zeroline: false
    },
    shapes: shapes
  });
  if (lstTickvals.length) {
    layout.xaxis2 = {
      overlaying: 'x', side: 'top',
      range: [-0.1, totalH + 0.6],
      tickvals: lstTickvals, ticktext: lstTicktext,
      tickfont: { size: 10.5 }, showgrid: false, showline: false,
      title: { text: 'LST', font: { size: 11 } }
    };
  }

  const height = Math.max(300, Math.min(460, tracks.length * 4 + 300));
  wrap.style.height = height + 'px';
  Plotly.react(wrap, traces, layout, PLOTLY_CONFIG);
}

/* ---- Bottom panel: one bar per exposure block + translucent overhead tail ---- */

function renderBlocksPanel(data, night) {
  const wrap = $('#chartBlocks');
  const nStart = parseUtc(data.night_start_utc) || parseUtc(night.night_start_utc);
  if (!nStart) return;
  const nEnd = parseUtc(data.night_end_utc) || parseUtc(night.night_end_utc);
  const totalH = Math.max(msToHours((nEnd || new Date(nStart.getTime() + 12 * 3.6e6)) - nStart), 0.5);

  const blocks = night.blocks || [];
  const colors = data.colors || {};
  const overheadH = msToHours((data.overhead_min != null ? data.overhead_min : 10) * 60000);
  const localTicks = buildLocalTicks(nStart, totalH);

  /* Scheduled targets in order of first appearance. */
  const targetOrder = [];
  blocks.forEach((b) => { if (targetOrder.indexOf(b.target) === -1) targetOrder.push(b.target); });

  const traces = [];

  /* One bordered bar per exposure block (dark edge on ALL sides, like the
     matplotlib barh) so adjacent same-target blocks read as distinct cells.
     Each block starts at `base` with length `x` along the shared time axis;
     bars in the same row with disjoint ranges do not overlap. */
  const blockYs = [], blockXs = [], blockBases = [], blockColors = [], blockHover = [];
  const tailYs = [], tailXs = [], tailBases = [], tailColors = [];
  blocks.forEach((b) => {
    const s = parseUtc(b.start_utc), e = parseUtc(b.end_utc);
    if (!s || !e) return;
    const sH = msToHours(s - nStart.getTime());
    const eH = msToHours(e - nStart.getTime());
    const color = colors[b.target] || AMBER_COL;
    blockYs.push(b.target);
    blockXs.push(eH - sH);
    blockBases.push(sH);
    blockColors.push(color);
    blockHover.push(
      '<b>' + esc(b.target) + '</b><br>' +
      'exposure ' + (b.exposure != null ? b.exposure : '—') + '<br>' +
      fmtLocalHM(s) + ' – ' + fmtLocalHM(e) + ' local (' + fmtHM(s) + ' – ' + fmtHM(e) + ' UTC)<br>' +
      'alt ' + (b.altitude_deg != null ? Number(b.altitude_deg).toFixed(1) : '—') + '° · az ' +
      (b.azimuth_deg != null ? Number(b.azimuth_deg).toFixed(1) : '—') + '°<br>' +
      'moon sep ' + (b.moon_sep_deg != null ? Number(b.moon_sep_deg).toFixed(1) : '—') + '°<extra></extra>');
    /* translucent overhead tail after the block */
    tailYs.push(b.target);
    tailXs.push(overheadH);
    tailBases.push(eH);
    tailColors.push(color);
  });
  /* tails first so the solid block cells draw on top of them */
  if (tailYs.length) {
    traces.push({
      type: 'bar', orientation: 'h',
      y: tailYs, x: tailXs, base: tailBases,
      marker: { color: tailColors, opacity: 0.3 },
      hoverinfo: 'skip', showlegend: false
    });
  }
  if (blockYs.length) {
    traces.push({
      type: 'bar', orientation: 'h',
      y: blockYs, x: blockXs, base: blockBases,
      marker: { color: blockColors, line: { color: '#0a0e14', width: 1 } },
      hovertemplate: blockHover, hoverlabel: { namelength: -1 }, showlegend: false
    });
  }

  const yLabels = targetOrder.map((n) => truncate(n, 18));
  const layout = baseLayout({
    margin: { l: 110, r: 16, t: 8, b: 46 },
    /* The tails and block bars are two bar traces on the same categorical y
       axis. 'overlay' (not the default 'group') keeps both traces on each
       category's center — the translucent tails sit beneath the solid block
       cells instead of being dodged into a second row above them. */
    barmode: 'overlay',
    xaxis: {
      range: [-0.1, totalH + 0.6], tickvals: localTicks.tickvals, ticktext: localTicks.ticktext,
      showgrid: true, gridcolor: 'rgba(255,255,255,0.12)', griddash: 'dot',
      linecolor: 'rgba(255,255,255,0.15)', zeroline: false,
      title: { text: 'Local time (UTC+8)', font: { size: 11 } }
    },
    yaxis: {
      tickmode: 'array', tickvals: targetOrder, ticktext: yLabels,
      autorange: 'reversed', ticklen: 0,
      gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.15)', zeroline: false
    }
  });

  const height = Math.max(140, Math.min(460, targetOrder.length * 40 + 30));
  wrap.style.height = height + 'px';
  Plotly.react(wrap, traces, layout, PLOTLY_CONFIG);
}

function renderBlocksTable(night) {
  const tbody = $('#blocksBody');
  tbody.innerHTML = '';
  const blocks = (night && night.blocks) || [];
  if (!blocks.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="8" class="empty">' +
      (night && !night.clear ? 'Cloudy night — no blocks.' : 'No blocks.') + '</td>';
    tbody.appendChild(tr);
    return;
  }
  blocks.forEach((b) => {
    const s = parseUtc(b.start_utc), e = parseUtc(b.end_utc);
    const tr = document.createElement('tr');
    const cells = [
      esc(b.target || ''),
      b.exposure != null ? esc(String(b.exposure)) : '—',
      fmtLocal(s, true),
      fmtLocal(e, true),
      b.altitude_deg != null ? Number(b.altitude_deg).toFixed(1) : '—',
      b.azimuth_deg != null ? Number(b.azimuth_deg).toFixed(1) : '—',
      b.moon_sep_deg != null ? Number(b.moon_sep_deg).toFixed(1) : '—',
      s && e ? ((e - s) / 60000).toFixed(1) : '—'
    ];
    cells.forEach((c) => {
      const td = document.createElement('td');
      td.innerHTML = c;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

/* ----- Download tab ----- */

function renderDownloadPanel() {
  if (!state.jobId) return;
  const base = '/api/schedule/' + encodeURIComponent(state.jobId) + '/download/';
  $('#dlBlocks').href = base + 'blocks';
  $('#dlTargets').href = base + 'targets';
  $('#dlNights').href = base + 'nights';
  $('#downloadNote').textContent = 'Job ' + state.jobId +
    ' · files are generated from the in-memory result and are valid until the server restarts.';
}

/* ============================================================
   Exit / server shutdown
   ============================================================ */

function jobRunning() {
  return state.running || !$('#runPanel').hidden;
}

function openExitPopover() {
  const msg = $('#exitPopoverMsg');
  msg.textContent = jobRunning()
    ? 'A scheduling job is running and will be lost. Stop the server and exit?'
    : 'Stop the local server and exit?';
  $('#exitPopover').hidden = false;
}

function closeExitPopover() {
  $('#exitPopover').hidden = true;
}

async function confirmExit() {
  if (state.shuttingDown) return;
  state.shuttingDown = true;           /* suppress polling, errors, banners */
  closeExitPopover();
  clearTimeout(state.pollTimer);       /* stop any in-flight polling loop */
  hideError();
  try {
    await apiShutdown();
  } catch (err) {
    /* server already unreachable — still show the calm stopped state */
  }
  $('#stoppedOverlay').hidden = false;
}

/* ============================================================
   Bootstrap
   ============================================================ */

function initDefaults() {
  const today = new Date();
  const start = toISODate(today);
  const end = toISODate(new Date(today.getTime() + 3 * 86400000));
  $('#singleDate').value = start;
  $('#rangeStart').value = start;
  $('#rangeEnd').value = end;
}

function wireEvents() {
  /* Targets */
  $('#uploadBtn').addEventListener('click', () => $('#csvFile').click());
  $('#csvFile').addEventListener('change', async (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!f) return;
    try {
      const text = await f.text();
      await handleParse(text, 'uploaded ' + f.name);
    } catch (err) {
      showError('Could not parse the file: ' + err.message);
    }
  });
  $('#exampleBtn').addEventListener('click', async () => {
    try {
      const text = await (await fetch('/api/targets/example')).text();
      await handleParse(text, 'example catalog');
    } catch (err) {
      showError('Could not load the example catalog: ' + err.message);
    }
  });
  $('#addRowBtn').addEventListener('click', addRow);

  $('#targetsBody').addEventListener('input', (e) => {
    const inp = e.target.closest('input');
    const tr = e.target.closest('tr.target-row');
    if (!inp || !tr) return;
    const idx = +tr.dataset.idx;
    state.rows[idx][inp.dataset.field] = inp.value;
    delete state.serverErrors[idx];
    computeRowErrors();
    refreshRowFeedback(idx);
    updateTargetsStatus();
    updateRunState();
  });

  $('#targetsBody').addEventListener('click', (e) => {
    const btn = e.target.closest('.icon-btn');
    if (!btn) return;
    const idx = +btn.dataset.idx;
    state.rows.splice(idx, 1);
    const shifted = {};
    Object.keys(state.serverErrors).forEach((k) => {
      const v = +k;
      if (v < idx) shifted[v] = state.serverErrors[k];
      else if (v > idx) shifted[v - 1] = state.serverErrors[k];
    });
    state.serverErrors = shifted;
    renderTargetsTable();
  });

  /* Dates */
  $$('input[name="dateMode"]').forEach((r) =>
    r.addEventListener('change', () => {
      const single = dateModeValue() === 'single';
      $('#singleDateRow').hidden = !single;
      $('#rangeDateRow').hidden = single;
      if (single) {
        $('#clearProb').value = '1.0';
        $('#seed').value = '2027';
      } else {
        $('#clearProb').value = '0.5';
        $('#seed').value = '42';
      }
      updateRunState();
    }));
  ['singleDate', 'rangeStart', 'rangeEnd'].forEach((id) =>
    $('#' + id).addEventListener('change', updateRunState));

  /* Parameters & run */
  ['clearProb', 'seed', 'timeLimit', 'eps', 'gamma', 'alpha'].forEach((id) =>
    $('#' + id).addEventListener('input', updateRunState));
  $('#runBtn').addEventListener('click', startSchedule);
  $('#cancelBtn').addEventListener('click', cancelJob);

  /* Tabs */
  $$('.tab-btn').forEach((b) =>
    b.addEventListener('click', () => activateTab(b.dataset.tab)));

  /* Per-night selection */
  $('#nightSelect').addEventListener('change', (e) => {
    state.nightIdx = +e.target.value;
    const nights = (state.result.nights) || [];
    renderBlocksTable(nights[state.nightIdx]);
    loadNightTracks(state.nightIdx);
  });

  /* Exit / shutdown */
  $('#exitBtn').addEventListener('click', openExitPopover);
  $('#exitConfirmBtn').addEventListener('click', confirmExit);
  $('#exitCancelBtn').addEventListener('click', closeExitPopover);
  if (typeof document.addEventListener === 'function') {
    document.addEventListener('click', (e) => {
      const t = e.target;
      if (t && t.closest && !t.closest('#exitPopover') && !t.closest('#exitBtn')) {
        closeExitPopover();
      }
    });
  }
}

(function init() {
  initDefaults();
  wireEvents();
  renderTargetsTable();
  updateRunState();
})();
