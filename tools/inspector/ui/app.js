const state = { view: 'overview', artifacts: [], report: null, quality: null, metrics: null, map: null, history: [] };
const byId = id => document.getElementById(id);
const safeText = value => value === null || value === undefined || value === '' ? 'N/A' : String(value);
function el(tag, text = '', className = '') { const item = document.createElement(tag); if (text !== '') item.textContent = text; if (className) item.className = className; return item; }
async function api(path, options = {}) { const response = await fetch(path, { headers: { 'Accept': 'application/json' }, ...options }); const body = await response.json(); if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`); return body; }
function banner(message, kind = 'success') { const box = byId('banner'); box.textContent = message; box.className = `banner ${kind}`; window.setTimeout(() => box.classList.add('hidden'), 5000); }
function setView(view) {
  state.view = view;
  document.querySelectorAll('.view').forEach(section => section.classList.toggle('active', section.id === `view-${view}`));
  document.querySelectorAll('.nav-item').forEach(button => {
    const active = button.dataset.view === view;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  const title = { overview: 'Overview', data: 'Data Explorer', models: 'Model Lab', spatial: 'Spatial Viewer', reports: 'QA & Reports' };
  byId('page-title').textContent = title[view] || 'Overview';
}
function renderOverview() {
  const report = state.report || {}; const quality = state.quality || {}; const integration = report.integration || {};
  const hasScan = Boolean(report.scan_id);
  const studyCount = Object.values((state.metrics || {}).studies || {}).filter(study => (study.artifacts || []).length > 0).length;
  const stats = byId('overview-stats'); stats.replaceChildren();
  for (const [label, value, note] of [
    ['Artifacts', hasScan ? report.artifact_count : '—', hasScan ? 'in configured roots' : 'awaiting first scan'],
    ['Changed', hasScan ? report.changed_count : '—', hasScan ? 'during the last scan' : 'awaiting first scan'],
    ['Attention', hasScan ? (quality.attention_count ?? 'N/A') : '—', hasScan ? 'items to review' : 'awaiting first scan'],
    ['Evidence streams', hasScan ? studyCount : '—', hasScan ? 'kept independent' : 'awaiting first scan']
  ]) {
    const item = el('div', '', 'overview-stat');
    item.append(el('dt', label), el('dd', safeText(value)), el('small', note));
    stats.append(item);
  }
  byId('scan-state').textContent = report.scan_id ? 'Scan complete' : 'Awaiting scan';
  byId('scan-state').className = `pill ${report.scan_id ? 'good' : 'neutral'}`;
  const summary = byId('scan-summary'); summary.replaceChildren();
  if (!report.scan_id) summary.append(el('div', 'Start a refresh to index configured artifacts.', 'empty-state'));
  else { const lines = [['Scan ID', report.scan_id], ['Completed', new Date(report.created_at).toLocaleString()], ['Artifacts', report.artifact_count], ['Read errors', report.error_count]]; for (const [key, value] of lines) { summary.append(el('strong', key)); summary.append(el('span', safeText(value))); } summary.className = 'scan-lines'; }
  const status = integration.status || 'NOT_EVALUATED'; byId('integration-state').textContent = status.replaceAll('_', ' '); byId('integration-state').className = `pill ${status === 'BLOCKED' ? 'bad' : status === 'WARNING' ? 'warn' : status === 'PASS' ? 'good' : 'neutral'}`;
  byId('integration-summary').textContent = status === 'NOT_EVALUATED' ? 'No complete Platform series payload was evaluated; absence is not a readiness PASS.' : `${integration.candidate_payload_count || 0} candidate artifact(s) · read-only audit; no exports are changed.`;
  const checks = byId('integration-checks'); checks.replaceChildren();
  for (const check of integration.checks || []) { const row = el('div', '', 'integration-check'); row.append(el('strong', check.check.replaceAll('_', ' ')), statusPill(check.status), el('span', check.reason || 'No explanation provided')); checks.append(row); }
  const warnings = byId('integration-warnings'); warnings.replaceChildren();
  if (!(integration.checks || []).length) warnings.append(el('div', 'No check results are available.', 'muted'));
  renderStudyCards(byId('study-cards'), (state.metrics || {}).studies || {});
}
function statusPill(status) { const s = String(status || 'UNKNOWN'); const cls = ['OK', 'Completed', 'PASS'].includes(s) ? 'good' : ['WARNING', 'UNSUPPORTED', 'RESTRICTED', 'RESTRICTED/UNKNOWN_SPLIT', 'Deferred', 'Not Applicable', 'Pending', 'Running'].includes(s) ? 'warn' : ['ERROR', 'INCOMPLETE', 'FAIL', 'Failed', 'BLOCKED'].includes(s) ? 'bad' : 'neutral'; return el('span', s.replaceAll('_', ' '), `pill ${cls}`); }
function renderArtifacts() {
  const body = byId('artifact-rows'); body.replaceChildren();
  const search = byId('artifact-search').value.trim().toLowerCase(); const role = byId('role-filter').value;
  const visible = state.artifacts.filter(item => (!role || item.role === role) && (!search || `${item.display_name} ${item.relative_path} ${item.format} ${item.role}`.toLowerCase().includes(search)));
  byId('artifact-count').textContent = `${visible.length} / ${state.artifacts.length}`;
  for (const item of visible.slice(0, 2000)) {
    const row = document.createElement('tr'); const nameCell = document.createElement('td'); const button = el('button', item.display_name, 'artifact-button'); button.type = 'button'; button.addEventListener('click', () => showArtifact(item.id)); nameCell.append(button);
    const path = el('div', item.relative_path, 'muted'); nameCell.append(path); row.append(nameCell, el('td', item.format), el('td', item.role), document.createElement('td'));
    row.lastChild.append(statusPill(item.status)); body.append(row);
  }
  const roleSelect = byId('role-filter'); const prior = roleSelect.value; const roles = [...new Set(state.artifacts.map(item => item.role))].sort(); roleSelect.replaceChildren(new Option('All roles', ''));
  for (const value of roles) roleSelect.append(new Option(value.replaceAll('_', ' '), value)); roleSelect.value = roles.includes(prior) ? prior : '';
}
async function showArtifact(id) {
  const panel = byId('artifact-detail'); panel.replaceChildren(el('div', 'Loading artifact…', 'empty-state'));
  try { const item = await api(`/api/artifact/${encodeURIComponent(id)}`); panel.replaceChildren(); panel.append(el('h3', item.display_name)); panel.append(el('div', `${item.format} · ${item.role}`, 'detail-path')); panel.append(el('p', item.relative_path, 'detail-path')); panel.append(statusPill(item.status)); if (item.message) panel.append(el('p', item.message, 'muted')); panel.append(el('h4', 'Manifest & inspection')); panel.append(el('pre', JSON.stringify(item, null, 2), 'json-view')); }
  catch (error) { panel.replaceChildren(el('div', error.message, 'empty-state')); }
}
function renderStudyCards(target, studies) {
  target.replaceChildren();
  const labels = { dev: ['Development evaluations', 'Recorded reports/dev outputs and experiment log.'], architecture_benchmark: ['Architecture benchmark', 'A1–A10 and suppression evidence; incomplete cells stay explicit.'], regional_representation: ['Regional representation capstone', 'Pretraining, transfer, retrieval, generalization, and claim gates.'] };
  for (const key of ['dev', 'architecture_benchmark', 'regional_representation']) {
    const study = studies[key] || { artifacts: [], status_counts: {} }; const [title, desc] = labels[key];
    const row = el('div', '', 'evidence-row');
    const copy = el('div', '', 'evidence-copy'); copy.append(el('h4', title), el('p', desc));
    const meta = el('div', '', 'evidence-meta');
    const count = (study.artifacts || []).length;
    meta.append(el('strong', `${count} ${count === 1 ? 'artifact' : 'artifacts'}`));
    meta.append(el('span', Object.entries(study.status_counts || {}).map(([name, value]) => `${name}: ${value}`).join(' · ') || 'No outputs yet'));
    row.append(copy, meta); target.append(row);
  }
}
function renderModels() {
  const target = byId('models-content'); target.replaceChildren(); const studies = (state.metrics || {}).studies || {};
  const titles = { dev: 'Development evaluations', architecture_benchmark: 'Architecture benchmark', regional_representation: 'Regional representation capstone' };
  for (const key of ['dev', 'architecture_benchmark', 'regional_representation']) {
    const study = studies[key] || { artifacts: [] }; const section = el('article', '', 'panel study-section'); section.append(el('h3', titles[key]));
    section.append(el('p', key === 'dev' ? 'Existing evaluator output only; no metrics are recomputed.' : key === 'architecture_benchmark' ? 'Architecture matrix and suppression experiments. Not merged with capstone results.' : 'Capstone task ladders and gates. `near_matches` remains exploratory.', 'study-desc'));
    if (!study.artifacts?.length) section.append(el('div', 'No report artifacts discovered in this lineage.', 'empty-state'));
    else { const grid = el('div', '', 'study-artifacts'); for (const item of study.artifacts.slice(0, 90)) { const chip = el('div', '', 'artifact-chip'); chip.append(el('strong', item.path)); chip.append(el('small', `${item.format} · ${item.registry_status || item.status}${item.protocol ? ` · ${item.protocol}` : ''}${item.claim_rung ? ` · ${item.claim_rung}` : ''}`)); if (item.metrics_display && Object.keys(item.metrics_display).length) chip.append(el('small', `Recorded values · ${JSON.stringify(item.metrics_display)}`)); else chip.append(el('small', `Comparison · ${item.comparison_state || 'N/A'}`)); if (item.restricted_from_comparison && item.restriction_reason) chip.append(el('small', `Restricted · ${item.restriction_reason}`)); if (item.evidence_class) chip.append(el('small', `Evidence · ${item.evidence_class}`)); if (item.message) chip.append(el('small', item.message)); grid.append(chip); } section.append(grid); }
    target.append(section);
  }
}
function renderQuality() {
  const quality = state.quality || {}; const counts = quality.status_counts || {}; const target = byId('quality-status'); target.replaceChildren();
  for (const [status, count] of Object.entries(counts).sort((a,b) => b[1]-a[1])) { const row = el('div', '', 'status-row'); row.append(el('span', status.replaceAll('_', ' ')), el('strong', count)); target.append(row); }
  if (!Object.keys(counts).length) target.append(el('div', 'No scan report yet.', 'empty-state'));
  const attention = byId('quality-attention'); attention.replaceChildren(); const flagged = state.artifacts.filter(item => !['OK', 'DISCOVERED'].includes(item.status)).slice(0, 40);
  if (!flagged.length) attention.append(el('div', 'No attention items recorded.', 'muted'));
  for (const item of flagged) attention.append(el('div', `${item.status} · ${item.relative_path}${item.message ? ` — ${item.message}` : ''}`, 'compact-item'));
  const history = byId('history-rows'); history.replaceChildren();
  for (const item of state.history.slice(0, 100)) { const row = document.createElement('tr'); for (const value of [item.scan_id, new Date(item.created_at).toLocaleString(), item.artifact_count, item.changed_count, item.error_count]) row.append(el('td', safeText(value))); history.append(row); }
}
function renderMap() {
  const svg = byId('farm-map'); svg.replaceChildren(); const data = state.map || {};
  byId('map-count').textContent = data.available ? `${data.farm_count} farms` : 'Unavailable'; byId('map-count').className = `pill ${data.available ? 'good' : 'warn'}`;
  if (!data.available || !data.farms?.length) { const text = document.createElementNS('http://www.w3.org/2000/svg', 'text'); text.setAttribute('x','450'); text.setAttribute('y','280'); text.setAttribute('text-anchor','middle'); text.setAttribute('fill','#77827a'); text.textContent = data.reason || 'Farm geometry unavailable'; svg.append(text); return; }
  const coords = data.farms.flatMap(farm => farm.polygon.length ? farm.polygon : [farm.centroid]); const lats = coords.map(p => p[0]), lons = coords.map(p => p[1]); const minLat = Math.min(...lats), maxLat = Math.max(...lats), minLon = Math.min(...lons), maxLon = Math.max(...lons); const pad = 30;
  const point = ([lat,lon]) => [pad + (lon-minLon)/(maxLon-minLon || 1)*(900-pad*2), 530 - ((lat-minLat)/(maxLat-minLat || 1))*(560-pad*2)];
  for (const farm of data.farms) {
    if (farm.polygon.length >= 3) {
      const polygon = document.createElementNS('http://www.w3.org/2000/svg','polygon');
      polygon.setAttribute('points', farm.polygon.map(p => point(p).join(',')).join(' '));
      polygon.setAttribute('class','farm-shape'); polygon.setAttribute('tabindex','0'); polygon.setAttribute('role','button');
      polygon.setAttribute('aria-pressed','false'); polygon.setAttribute('aria-label', `${farm.id} ${farm.code}`);
      polygon.addEventListener('click', () => selectFarm(farm, polygon));
      polygon.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectFarm(farm, polygon); } });
      svg.append(polygon);
    } else {
      const [x,y] = point(farm.centroid); const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
      circle.setAttribute('cx',x); circle.setAttribute('cy',y); circle.setAttribute('r','5'); circle.setAttribute('class','farm-point');
      circle.setAttribute('tabindex','0'); circle.setAttribute('role','button'); circle.setAttribute('aria-pressed','false');
      circle.setAttribute('aria-label', `${farm.id} ${farm.code}`);
      circle.addEventListener('click', () => selectFarm(farm,circle));
      circle.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectFarm(farm,circle); } });
      svg.append(circle);
    }
  }
}
function selectFarm(farm, shape) {
  document.querySelectorAll('.farm-shape.selected, .farm-point.selected').forEach(node => { node.classList.remove('selected'); node.setAttribute('aria-pressed', 'false'); });
  if (shape) { shape.classList.add('selected'); shape.setAttribute('aria-pressed', 'true'); }
  state.selectedFarmId = farm.id;
  const panel = byId('farm-detail'); panel.replaceChildren(el('h3', farm.id || 'Farm'), el('div', farm.code || 'Registry code not recorded', 'farm-code'));
  const grid = el('div', '', 'farm-details-grid'); for (const [label, value] of [['Block', farm.block], ['Latitude', farm.centroid?.[0]], ['Longitude', farm.centroid?.[1]], ['Polygon points', farm.polygon?.length]]) { const cell = el('div'); cell.append(el('small',label), el('strong',safeText(value))); grid.append(cell); } panel.append(grid,el('p','Geometry shown from the configured farm registry. Timeline values are recorded observations only.','muted'));
  const section = el('section', '', 'farm-timeline'); section.append(el('h4', 'Index timeline'), el('div', 'Loading this farm’s recorded series…', 'empty-state')); panel.append(section);
  api(`/api/series?farm_id=${encodeURIComponent(farm.id)}`).then(data => {
    if (state.selectedFarmId !== farm.id) return;
    section.replaceChildren(el('h4', 'Index timeline'));
    if (!data.available) { section.append(el('div', `${data.status || 'UNAVAILABLE'} · ${data.reason || 'Series unavailable'}`, 'empty-state')); return; }
    renderTimeline(section, data);
  }).catch(error => { if (state.selectedFarmId === farm.id) { section.replaceChildren(el('h4', 'Index timeline'), el('div', error.message, 'empty-state')); } });
}
function renderTimeline(target, data) {
  const observations = data.observations || [];
  const markerMessage = data.stress_marker_status === 'available' ? 'Suspected-stress markers use recorded v5 model flags only; they are not verified stress labels.'
    : data.stress_marker_status === 'unavailable_v4' ? 'Stress markers unavailable: this is a v4 series; no v5 flag provenance is present.'
    : data.stress_marker_status === 'no_suspected_flags' ? 'No suspected-stress flags are recorded for this farm.'
    : 'Stress markers unavailable: recorded v5 stress flags are missing.';
  target.append(el('p', `${observations.length} non-test observations · ${data.excluded_test_observation_count || 0} test observation(s) excluded.`, 'muted'), el('p', markerMessage, 'muted'));
  const points = observations.map(item => ({...item, time: Date.parse(item.date)})).filter(item => Number.isFinite(item.time)).sort((a, b) => a.time - b.time);
  const channelNames = (data.index_channels || []).filter(name => points.some(item => Number.isFinite(item.channels?.[name])));
  if (!points.length || !channelNames.length) { target.append(el('div', 'No dated index values are available for this farm.', 'empty-state')); return; }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('viewBox', '0 0 800 300'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Recorded index values over actual acquisition dates; test observations excluded'); svg.classList.add('timeline-chart');
  const ns = 'http://www.w3.org/2000/svg'; const addSvg = (tag, attrs, text = '') => { const node = document.createElementNS(ns, tag); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value)); if (text) node.textContent = text; svg.append(node); return node; };
  const left = 54, right = 782, top = 18, bottom = 246; const minTime = Math.min(...points.map(p => p.time)), maxTime = Math.max(...points.map(p => p.time));
  const values = points.flatMap(item => channelNames.map(name => item.channels?.[name])).filter(Number.isFinite); let minValue = Math.min(...values), maxValue = Math.max(...values); if (minValue === maxValue) { minValue -= .05; maxValue += .05; }
  const x = time => left + (time - minTime) / (maxTime - minTime || 1) * (right - left); const y = value => bottom - (value - minValue) / (maxValue - minValue) * (bottom - top);
  for (let tick = 0; tick <= 4; tick++) { const val = minValue + (maxValue - minValue) * tick / 4; const yy = y(val); addSvg('line', {x1:left,y1:yy,x2:right,y2:yy,class:'timeline-grid'}); addSvg('text', {x:left-8,y:yy+4,'text-anchor':'end',class:'timeline-axis'}, val.toFixed(2)); }
  const colors = ['#176b57','#a96b1e','#315f91','#7757a7','#3e7f78','#6d6258'];
  channelNames.forEach((name, seriesIndex) => { const coords = points.filter(item => Number.isFinite(item.channels?.[name])).map(item => `${x(item.time)},${y(item.channels[name])}`); if (coords.length) addSvg('polyline', {points:coords.join(' '),class:'timeline-line',stroke:colors[seriesIndex % colors.length]}); });
  for (const item of points) for (let idx = 0; idx < channelNames.length; idx++) { const value = item.channels?.[channelNames[idx]]; if (Number.isFinite(value)) addSvg('circle',{cx:x(item.time),cy:y(value),r:2.4,fill:colors[idx % colors.length]}); }
  const firstDate = new Date(minTime).toISOString().slice(0,10), lastDate = new Date(maxTime).toISOString().slice(0,10); addSvg('text',{x:left,y:270,class:'timeline-axis'},firstDate); addSvg('text',{x:right,y:270,'text-anchor':'end',class:'timeline-axis'},lastDate);
  const markers = points.filter(item => item.split !== 'test' && item.stress_flag === 'suspected_stress');
  for (const item of markers) { const xx = x(item.time); const provenance = Object.entries(item.channel_provenance || {}).map(([name, state]) => `${name}: ${Array.isArray(state) ? state[0] : state}`).join(', ') || 'channel provenance unavailable'; const description = `${item.date} · suspected stress · model ${safeText(item.stress_flag_model)} · split ${item.split} · ${provenance}`; const line = addSvg('line',{x1:xx,y1:top,x2:xx,y2:bottom,class:'stress-marker',tabindex:0,role:'img','aria-label':description}); const title = document.createElementNS(ns,'title'); title.textContent = description; line.append(title); const dot = addSvg('circle',{cx:xx,cy:top+4,r:4,class:'stress-marker-dot',tabindex:0,'aria-label':description}); const dotTitle = document.createElementNS(ns,'title'); dotTitle.textContent = description; dot.append(dotTitle); }
  target.append(svg);
  const legend = el('div', '', 'timeline-legend'); channelNames.forEach((name, idx) => { const item = el('span', '', 'timeline-key'); const swatch = el('i'); swatch.style.backgroundColor = colors[idx % colors.length]; item.append(swatch, document.createTextNode(name)); legend.append(item); });
  if (markers.length) { const markerKey = el('span', '', 'timeline-key'); const swatch = el('i'); swatch.className = 'stress-swatch'; markerKey.append(swatch, document.createTextNode('Recorded suspected stress')); legend.append(markerKey); }
  target.append(legend);
  if (markers.length) {
    const list = el('div', '', 'stress-events'); list.append(el('h5', 'Recorded suspected-stress dates'));
    for (const item of markers) { const provenance = Object.entries(item.channel_provenance || {}).map(([name, state]) => `${name}: ${Array.isArray(state) ? state[0] : state}`).join(' · ') || 'Channel provenance unavailable';
      list.append(el('div', `${item.date} · split ${item.split} · flag model ${safeText(item.stress_flag_model)} · ladder ${safeText(item.ladder_rung)} · ${provenance}`, 'stress-event')); }
    target.append(list);
  }
  target.append(el('p', 'Channel provenance is recorded per observation; values and dates from the locked test split are not displayed.', 'muted'));
}
async function refreshData() {
  const button = byId('refresh'); button.disabled = true; button.querySelector('span').textContent = 'Scanning…';
  try { state.report = await api('/api/refresh', { method: 'POST' }); banner(`Scan complete · ${state.report.artifact_count} artifacts · ${state.report.changed_count} inspected`, 'success'); await loadData(false); }
  catch (error) { banner(error.message, 'error'); }
  finally { button.disabled = false; button.querySelector('span').textContent = 'Refresh scan'; }
}
async function loadData(showErrors = true) {
  try {
    const [status, report, quality, metrics, artifacts, history, map] = await Promise.all([
      api('/api/status'), api('/api/report/latest').catch(() => null), api('/api/quality'), api('/api/metrics'), api('/api/artifacts'), api('/api/history'), api('/api/map')
    ]);
    state.report = report || state.report; state.quality = quality; state.metrics = metrics; state.artifacts = artifacts.artifacts || []; state.history = history.history || []; state.map = map;
    byId('last-scan').textContent = status.last_scan ? `Last scan · ${new Date(status.last_scan.created_at).toLocaleString()}` : 'Last scan · waiting';
    byId('root-count').textContent = `${status.roots.filter(root=>root.exists).length} configured roots available`;
    renderOverview(); renderArtifacts(); renderModels(); renderQuality(); renderMap();
  } catch (error) { if (showErrors) banner(error.message, 'error'); }
}
for (const button of document.querySelectorAll('.nav-item')) button.addEventListener('click', () => setView(button.dataset.view));
byId('refresh').addEventListener('click', refreshData);
byId('artifact-search').addEventListener('input', renderArtifacts);
byId('role-filter').addEventListener('change', renderArtifacts);
byId('download-json').addEventListener('click', async () => { try { const data = await api('/api/report/latest'); const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'}); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'thesis-inspector-latest.json'; link.click(); URL.revokeObjectURL(link.href); } catch (error) { banner(error.message,'error'); } });
loadData();
