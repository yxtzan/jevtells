'use strict';
/* JevTells workbench: one screen, the same API and parameter mapping as M9. */
const $ = id => document.getElementById(id);
const state = {
  lang: 'zh', defaultLang: 'zh', dict: {}, token: '', checks: [], stages: [],
  input: '', info: null, samples: [], focus: 1, visited: {}, showPicker: false, showDone: false,
  time: 0, playing: false, target: [], blur: [], others: [],
  timeline: null, output: null, results: null, resultLayout: 'both',
  job: null, watching: null, jobStarted: 0, renderOnly: false, queuedRun: false, poll: null,
  peopleReady: false, peopleCache: new Map(), people: null, peopleNote: '', inspecting: false,
  preparedSig: null, preparingSig: null, command: null, outputDir: 'work', reframe: 'auto',
};
const t = k => state.dict[k] ?? k;
const tf = (k, vars) => String(t(k)).replace(/\{(\w+)\}/g, (_, n) => vars[n] ?? '');
const text = (node, value) => { node.textContent = value; };
const el = (tag, cls, value) => { const n = document.createElement(tag); if (cls) n.className = cls; if (value != null) n.textContent = value; return n; };
const media = path => '/media/' + path.split('/').map(encodeURIComponent).join('/');
const frameUrl = (path, at) => '/api/frame?' + new URLSearchParams({input: path, at: Math.max(0, at).toFixed(3)});
const busy = (job = state.job) => !!job && ['running', 'cancelling'].includes(job.status);
const auto = () => $('autoStart').checked;
const pad = n => String(Math.floor(n)).padStart(2, '0');
const clock = s => `${Math.floor(s / 60)}:${pad(s % 60)}.${Math.floor((s * 10) % 10)}`;
const mmss = s => `${pad(s / 60)}:${pad(s % 60)}`;
const circled = n => n >= 1 && n <= 20 ? String.fromCharCode(0x245F + n) : '#' + n;

let noticeTimer;
function notice(message, kind = '') {
  clearTimeout(noticeTimer);
  text($('notice'), t(message));
  $('notice').className = kind;
  $('notice').hidden = false;
  if (kind !== 'sticky') noticeTimer = setTimeout(() => { $('notice').hidden = true; }, kind === 'error' ? 6000 : 3500);
}
function hideNotice() { $('notice').hidden = true; }
function error(e) { console.error(e); notice(e.message || 'operation_failed', 'error'); }
async function api(path, data) {
  const options = {headers: {}};
  if (data !== undefined) {
    options.method = 'POST';
    options.headers = {'Content-Type': 'application/json', 'X-JevTells-Token': state.token};
    options.body = JSON.stringify(data);
  }
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'operation_failed');
  return result;
}
function handle(fn) { return async event => { try { await fn(event); } catch (e) { error(e); } }; }
function button(label, fn, cls = '') { const b = el('button', cls, label); b.type = 'button'; b.onclick = handle(fn); return b; }
function chip(parent, label, onRemove, onClick, cls = '') {
  const b = el('button', 'chip ' + cls); b.type = 'button';
  b.append(el('span', '', label));
  if (onRemove) {
    const x = el('span', 'xs', '✕'); x.setAttribute('role', 'button'); x.title = t('remove');
    x.onclick = handle(e => { e.stopPropagation(); return onRemove(); });
    b.append(x);
  }
  b.onclick = handle(onClick || onRemove);
  parent.append(b);
  return b;
}

/* ---------- language and chrome ---------- */
async function language(lang) {
  state.lang = lang;
  state.dict = await (await fetch('/static/locales/' + lang + '.json')).json();
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach(node => text(node, t(node.dataset.i18n)));
  document.querySelectorAll('[data-i18n-title]').forEach(node => { node.title = t(node.dataset.i18nTitle); node.setAttribute('aria-label', node.title); });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  text($('language'), lang === 'zh' ? 'EN' : '中');
  renderAll();
  if (!$('drawerHistory').hidden) await history();
  await samples();
}
function stepDone(n) {
  if (n === 1) return !!state.info;
  if (n === 2) return !!($('speaker').value.trim() && (state.target.length || $('targetDesc').value.trim()));
  if (n === 3) return !!(state.visited[3] || state.blur.length);
  if (n === 4) return !!(state.timeline && state.visited[4]);
  return !!state.results;
}
function renderSteps() {
  const nav = $('steps');
  nav.replaceChildren();
  t('steps').forEach((label, i) => {
    const n = i + 1;
    const b = button('', () => focusStep(n), 'step');
    b.append(el('em', '', String(n)), document.createTextNode(label));
    const current = state.input ? n === state.focus : n === 1;
    b.classList.toggle('current', current);
    b.classList.toggle('done', !current && stepDone(n));
    b.disabled = !state.input && n > 1;
    if (current) b.setAttribute('aria-current', 'step');
    nav.append(b);
  });
}
function missing() {
  const rows = [];
  if (!state.input) rows.push(['missSource', 1]);
  else {
    if (!$('speaker').value.trim()) rows.push(['missSpeaker', 2]);
    if (!state.target.length && !$('targetDesc').value.trim()) rows.push(['missTarget', 2]);
    if (!auto() && +$('rangeEnd').value <= +$('rangeStart').value) rows.push(['range_error', 1]);
  }
  return rows;
}
function renderGenerate() {
  const running = busy() && state.job.mode === 'run';
  const miss = missing();
  text($('generateLabel'), running ? t('generating') : state.queuedRun ? t('waitingPrepare') : t('generateCta'));
  $('generate').querySelector('b').hidden = running || state.queuedRun;
  $('generate').disabled = running || state.queuedRun;
  $('generate').setAttribute('aria-disabled', String(!!miss.length && !running));
  if (miss.length && !running) $('generateWrap').dataset.tip = t('missingPrefix') + '\n' + miss.map(([k]) => '· ' + t(k)).join('\n');
  else delete $('generateWrap').dataset.tip;
  $('generateBig').disabled = !!miss.length || running || state.queuedRun;
}
function renderBlockers() {
  const failed = state.checks.filter(c => !c.ok);
  $('blockers').hidden = !failed.length;
  text($('blockers'), tf('blockers', {n: failed.length}));
}
function renderAll() {
  renderSteps();
  renderGenerate();
  renderBlockers();
  renderStage();
  renderAnchors();
  renderBlurs();
  renderIntervals();
  renderRange();
  renderEstimate();
  renderChecks();
  renderJob();
  renderTimeline();
  const empty = !state.input || !state.info;
  $('inspector').classList.toggle('is-disabled', empty);
  $('timelineCard').classList.toggle('is-disabled', empty);
  text($('legendTarget'), $('speaker').value.trim() || t('targetTalk'));
  text($('subtitleStatus'), state.info ? t(state.info.subtitles.bounds ? 'subtitleFound' : 'subtitleNone') : '');
}

/* ---------- stage card ---------- */
function currentView() {
  if (!state.input || !state.info || (state.focus === 1 && state.showPicker)) return 'picker';
  if (state.queuedRun || (busy() && state.job.mode === 'run')) return 'job';
  if (state.focus === 5) return state.results && state.showDone ? 'done' : 'summary';
  return 'edit';
}
function renderStage() {
  const view = currentView();
  for (const [name, id] of [['picker', 'viewPicker'], ['edit', 'viewEdit'], ['summary', 'viewSummary'], ['job', 'viewJob'], ['done', 'viewDone']]) $(id).hidden = view !== name;
  let title, hint, next = null;
  if (view === 'picker') {
    [title, hint] = state.input && state.info ? ['pickTitle', 'pickHint'] : ['emptyTitle', 'emptyHint'];
    if (state.input && state.info) next = ['useThis', 2];
  } else if (view === 'job') [title, hint] = ['jobTitle', 'jobHint'];
  else if (view === 'done') [title, hint] = ['doneTitle', 'cacheHint'];
  else if (view === 'summary') [title, hint] = ['summaryTitle', 'summaryHint'];
  else if (state.focus === 3) { [title, hint] = ['blurTitle', 'blurHint']; next = ['nextTalk', 4]; }
  else if (state.focus === 4) { [title, hint] = ['talkTitle', state.timeline ? 'talkHint' : 'talkWait']; next = ['nextGenerate', 5]; }
  else { [title, hint] = ['personTitle', state.peopleReady ? 'personHint' : 'personDetecting']; next = ['nextBlur', 3]; }
  text($('stageTitle'), t(title));
  text($('stageHint'), t(state.inspecting ? 'inspecting' : hint));
  $('nextStep').hidden = !next;
  if (next) { text($('nextStep'), t(next[0])); $('nextStep').onclick = handle(() => focusStep(next[1])); }
  if (view === 'picker') renderSamples();
  if (view === 'edit') renderCanvas();
}
async function focusStep(n) {
  if (n > 1 && !state.input) throw Error('select_source');
  const previous = state.focus;
  state.focus = n;
  state.visited[n] = true;
  state.showPicker = n === 1;
  state.showDone = n === 5 && !!state.results;
  if (state.playing) $('stageVideo').pause();
  renderAll();
  flash(n);
  if (n === 5) await summary();
  if (n === 2 || n === 3) { setFrame(true); loadPeople().catch(error); }
  if (previous === 2 && n !== 2) maybePrepare();
}
function flash(n) {
  const id = {1: 'secRange', 2: 'secSubject', 3: 'secBlur', 4: 'timelineCard', 5: 'secEstimate'}[n];
  const node = $(id);
  if (!node || !state.info) return;
  node.classList.remove('flash'); void node.offsetWidth; node.classList.add('flash');
  if (id !== 'timelineCard') node.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}

/* ---------- sources ---------- */
async function samples() {
  state.samples = await api('/api/samples');
  if (currentView() === 'picker') renderSamples();
}
const durations = new Map();
function renderSamples() {
  const list = $('sampleList');
  list.replaceChildren();
  if (!state.samples.length) { list.append(el('p', 'empty-note', t('noSamples'))); return; }
  for (const row of state.samples) {
    const card = button('', () => selectSource(row.path), 'sample');
    card.classList.toggle('current', row.path === state.input);
    card.classList.toggle('loading', row.path === state.input && state.inspecting);
    const img = el('img'); img.alt = ''; img.loading = 'lazy'; img.src = frameUrl(row.path, 1);
    img.onerror = () => { img.onerror = null; img.src = frameUrl(row.path, 0); };
    const meta = el('span'); const name = el('b', '', row.name); const length = el('i', '', durations.get(row.path) || '');
    meta.append(name, length); card.append(img, meta); card.title = row.path;
    if (!durations.has(row.path)) {
      durations.set(row.path, '');
      const probe = document.createElement('video'); probe.preload = 'metadata'; probe.muted = true;
      probe.onloadedmetadata = () => { if (isFinite(probe.duration)) { durations.set(row.path, mmss(probe.duration)); renderSamplesLater(); } probe.removeAttribute('src'); probe.load(); };
      probe.src = media(row.path);
    }
    list.append(card);
  }
}
let samplesTimer;
function renderSamplesLater() { clearTimeout(samplesTimer); samplesTimer = setTimeout(() => { if (currentView() === 'picker') renderSamples(); }, 80); }
async function waitIdle() {
  for (let i = 0; i < 40 && busy(); i++) { await new Promise(r => setTimeout(r, 200)); state.job = await api('/api/jobs'); }
}
async function selectSource(input, restore = false) {
  if (busy()) {
    if (state.job.mode === 'run') throw Error('busy');
    state.watching = null;
    await api('/api/cancel', {});
    await waitIdle();
  }
  if (state.playing) $('stageVideo').pause();
  state.input = input; state.inspecting = true;
  if (!restore) state.info = null;
  notice('inspecting', 'sticky');
  renderAll();
  try { state.info = await api('/api/inspect?' + new URLSearchParams({input})); }
  finally { state.inspecting = false; }
  if (!restore) {
    state.target = []; state.blur = []; state.others = []; state.results = null; state.showDone = false; state.visited = {}; state.reframe = 'auto';
    $('rangeStart').value = 0; $('autoStart').checked = false; $('targetDesc').value = ''; $('speaker').value = ''; $('title').value = '';
    delete $('title').dataset.edited;
    $('burned').checked = !!state.info.subtitles.bounds; $('subtitleSource').value = 'auto';
    $('narrationLang').value = state.defaultLang; $('layout').value = 'both';
  }
  state.timeline = null; state.output = null; state.preparedSig = null; state.queuedRun = false;
  state.peopleReady = false; state.peopleCache = new Map(); state.people = null; state.peopleNote = '';
  for (const id of ['rangeStart', 'rangeEnd']) $(id).max = state.info.duration - .001;
  $('rangeEnd').max = state.info.duration; $('rangeEnd').value = state.info.duration;
  $('frame').style.setProperty('--ar', (state.info.width / state.info.height).toFixed(4));
  $('stageVideo').src = media(input);
  rangeChange();
  state.time = clipOffset();
  hideNotice();
  state.focus = 2; state.showPicker = false; state.visited[1] = true; state.visited[2] = true;
  renderAll();
  setFrame(true);
  if (!restore) await loadPeople();
}
async function upload(file) {
  if (!file) return;
  if (busy()) {
    if (state.job.mode === 'run') throw Error('busy');
    state.watching = null;
    await api('/api/cancel', {});
    await waitIdle();
  }
  notice('uploading', 'sticky');
  const response = await fetch('/api/upload', {method: 'POST', headers: {'X-JevTells-Token': state.token, 'X-Filename': encodeURIComponent(file.name)}, body: file});
  const result = await response.json();
  if (!response.ok) throw Error(result.error);
  await samples();
  await selectSource(result.path);
  notice('uploaded');
}

/* ---------- parameters (unchanged mapping) ---------- */
function sync() {
  if (!state.info) return;
  state.speaker = $('speaker').value.trim();
  state.title = $('title').value.trim();
  state.start = auto() ? 'auto' : Number($('rangeStart').value);
  const end = Number($('rangeEnd').value);
  state.duration = state.start === 'auto' ? null : Math.abs(end - state.info.duration) < .04 ? null : end - state.start;
}
function params(fallbackSpeaker = '') {
  sync();
  if (!state.input) throw Error('select_source');
  const speaker = state.speaker || fallbackSpeaker;
  if (!speaker) throw Error('input_and_speaker_required');
  if (!state.target.length && !$('targetDesc').value.trim()) throw Error('select_target');
  if (state.start !== 'auto' && Number($('rangeEnd').value) <= state.start) throw Error('range_error');
  return {input: state.input, speaker, start: state.start, duration: state.duration, target: state.target,
    target_desc: $('targetDesc').value.trim() || null, face_identity: $('faceIdentity').checked, lang: $('narrationLang').value,
    title: state.title || null, blur: state.blur, others_speaking: state.others, subtitles: $('burned').checked ? 'off' : 'on',
    subtitle_source: $('subtitleSource').value, layout: $('layout').value, reframe: state.reframe || 'auto', output_dir: state.outputDir || 'work'};
}
function clipOffset() { return auto() ? 0 : +$('rangeStart').value; }
function rangeChange(event) {
  if (event) { state.target = []; state.others = []; }
  $('rangeStart').disabled = auto(); $('rangeEnd').disabled = auto();
  sync();
  state.timeline = null; state.preparedSig = null;
  if (state.info) {
    const a = axis();
    state.time = Math.min(Math.max(state.time, a.start), a.start + a.dur - .001);
  }
  if (event) { renderAll(); setFrame(); updateCommand(); }
}
function renderRange() {
  const start = +$('rangeStart').value, end = +$('rangeEnd').value;
  text($('startValue'), start.toFixed(2) + 's');
  text($('endValue'), state.info ? end.toFixed(2) + 's' : '—');
  const full = state.info && start < .04 && Math.abs(end - state.info.duration) < .04;
  text($('rangeLabel'), !state.info ? '' : auto() ? t('autoRange') : full ? tf('fullRange', {d: state.info.duration.toFixed(1)}) : tf('partRange', {d: (end - start).toFixed(1)}));
}
function prepareSig() {
  sync();
  return JSON.stringify([state.input, state.start, state.duration, state.target, $('faceIdentity').checked, $('subtitleSource').value, $('targetDesc').value.trim()]);
}

/* ---------- canvas ---------- */
function axis() {
  if (!state.info) return {start: 0, dur: 1};
  if (state.timeline && state.timeline.duration) return {start: auto() ? Number(state.timeline.actual_start || 0) : +$('rangeStart').value, dur: state.timeline.duration};
  const start = auto() ? 0 : +$('rangeStart').value;
  const end = auto() ? state.info.duration : +$('rangeEnd').value;
  return {start, dur: Math.max(.001, end - start)};
}
let frameTimer;
function setFrame(now = false) {
  clearTimeout(frameTimer);
  const go = () => { if (state.info) $('stageImage').src = frameUrl(state.input, state.time); };
  if (now) go(); else frameTimer = setTimeout(go, 110);
}
function renderTime() {
  const a = axis();
  $('scrub').min = a.start; $('scrub').max = a.start + a.dur; $('scrub').value = state.time;
  text($('clock'), clock(state.time)); text($('clockEnd'), clock(a.start + a.dur));
  $('playhead').style.left = Math.max(0, Math.min(100, (state.time - a.start) / a.dur * 100)) + '%';
}
function renderCanvas() {
  const frame = $('frame');
  frame.classList.toggle('mode-blur', state.focus === 3);
  frame.classList.toggle('playing', state.playing);
  const s = state.info?.subtitles;
  const band = $('subtitleBand');
  band.hidden = !(state.focus === 3 && s?.bounds);
  if (!band.hidden) {
    band.style.left = s.bounds[0] / state.info.width * 100 + '%';
    band.style.width = (s.bounds[1] - s.bounds[0]) / state.info.width * 100 + '%';
    band.style.top = s.strip_y / state.info.height * 100 + '%';
    band.style.height = (1 - s.strip_y / state.info.height) * 100 + '%';
  }
  $('playIcon').setAttribute('d', state.playing ? 'M7 5h3.5v14H7zM13.5 5H17v14h-3.5z' : 'M8 5.5v13l10-6.5z');
  renderPeople();
  renderTime();
}
let peopleSeq = 0;
async function loadPeople() {
  if (!state.info || state.playing || ![2, 3].includes(state.focus)) return;
  const at = +state.time.toFixed(3), input = state.input, key = input + '@' + at.toFixed(2);
  if (state.peopleCache.has(key)) { state.people = {at, rows: state.peopleCache.get(key)}; state.peopleNote = ''; renderPeople(); return; }
  const seq = ++peopleSeq;
  try {
    const rows = await api('/api/people?' + new URLSearchParams({input, at}));
    state.peopleCache.set(key, rows);
    if (seq !== peopleSeq || input !== state.input) return;
    state.peopleReady = true; state.people = {at, rows}; state.peopleNote = rows.length ? '' : 'nobody';
  } catch (e) {
    if (seq !== peopleSeq) return;
    state.people = null;
    if (e.message === 'detect_first') { state.peopleReady = false; state.peopleNote = 'personDetecting'; if (!busy()) await startJob('people'); }
    else if (e.message === 'busy') state.peopleNote = state.peopleReady ? 'peopleBusy' : 'personDetecting';
    else throw e;
  }
  renderStage();
}
function renderPeople() {
  const box = $('personBoxes');
  box.replaceChildren();
  const note = $('frameNote');
  note.hidden = !['peopleBusy', 'nobody'].includes(state.peopleNote) || ![2, 3].includes(state.focus);
  text(note, t(state.peopleNote));
  if (!state.info || !state.people || ![2, 3].includes(state.focus) || Math.abs(state.people.at - state.time) > .02) return;
  const relative = state.people.at - clipOffset();
  const name = $('speaker').value.trim();
  for (const row of state.people.rows) {
    const [x, y, x1, y1] = row.box;
    const selected = state.target.some(([n, s]) => n === row.number && Math.abs(s - relative) < .04);
    const node = el('div', 'pbox' + (selected ? ' selected' : '') + (y / state.info.height < .06 ? ' top' : ''));
    Object.assign(node.style, {left: x / state.info.width * 100 + '%', top: y / state.info.height * 100 + '%', width: (x1 - x) / state.info.width * 100 + '%', height: (y1 - y) / state.info.height * 100 + '%'});
    node.tabIndex = 0; node.setAttribute('role', 'button');
    node.setAttribute('aria-label', tf('selectPerson', {n: row.number}));
    node.append(el('span', '', selected ? `${circled(row.number)} ${name || t('subject')} · ${t('subjectTag')}` : circled(row.number)));
    const choose = handle(() => choosePerson(row.number));
    node.onpointerdown = e => e.stopPropagation();
    node.onclick = choose;
    node.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } };
    box.append(node);
  }
}
function choosePerson(number) {
  const at = state.people ? state.people.at : state.time;
  const relative = at - clipOffset();
  if (relative < 0 || (!auto() && at >= +$('rangeEnd').value)) throw Error('range_error');
  state.target = state.target.filter(([, s]) => Math.abs(s - relative) > .04);
  state.target.push([number, +relative.toFixed(6)]);
  state.target.sort((a, b) => a[1] - b[1]);
  renderAll();
  updateCommand();
  if (state.focus !== 2) maybePrepare();
}
function renderAnchors() {
  const list = $('anchors');
  list.replaceChildren();
  if (!state.target.length) { list.append(el('span', 'none', t('noAnchors'))); return; }
  state.target.forEach(([n, at], index) => chip(list, `${circled(n)} · ${at.toFixed(2)}s`, () => {
    state.target.splice(index, 1); renderAll(); updateCommand(); maybePrepare();
  }, async () => { state.time = clipOffset() + at; await focusStep(2); }, 'lime'));
}

/* ---------- blur rectangles ---------- */
const positionName = ([x, y, w, h]) => {
  const col = Math.min(2, Math.floor((x + w / 2) / state.info.width * 3));
  const row = Math.min(2, Math.floor((y + h / 2) / state.info.height * 3));
  return t('positions')[row][col];
};
function renderBlurs() {
  const layer = $('blurBoxes'), list = $('blurList');
  layer.replaceChildren(); list.replaceChildren();
  if (!state.info) return;
  state.blur.forEach((rect, i) => {
    const [x, y, w, h] = rect;
    const box = el('div', 'blur-box');
    box.dataset.index = i;
    Object.assign(box.style, {left: x / state.info.width * 100 + '%', top: y / state.info.height * 100 + '%', width: w / state.info.width * 100 + '%', height: h / state.info.height * 100 + '%'});
    box.append(el('span', 'tag', tf('blurTag', {n: i + 1})));
    const x0 = el('button', 'x', '✕'); x0.type = 'button'; x0.title = t('remove');
    x0.onpointerdown = e => e.stopPropagation();
    x0.onclick = () => { state.blur.splice(i, 1); renderBlurs(); updateCommand(); };
    box.append(x0);
    for (const corner of ['nw', 'ne', 'sw', 'se']) { const h0 = el('i', 'h ' + corner); h0.dataset.corner = corner; box.append(h0); }
    layer.append(box);
    const c = chip(list, `${tf('blurChip', {n: i + 1})} · ${positionName(rect)}`, () => { state.blur.splice(i, 1); renderBlurs(); updateCommand(); }, () => focusStep(3));
    c.title = `${x}, ${y} · ${w} × ${h}`;
  });
  chip(list, t('blurAdd'), null, () => focusStep(3), 'add');
}
let blurDrag = null;
function displayRect(i, r) { const [x, y, w, h] = state.blur[i]; return [x / state.info.width * r.width, y / state.info.height * r.height, w / state.info.width * r.width, h / state.info.height * r.height]; }
$('frame').onpointerdown = e => {
  if (!state.info || state.focus !== 3 || state.playing || e.button !== 0) return;
  const r = $('frame').getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  const box = e.target.closest('.blur-box');
  if (box && box.dataset.index != null) {
    const index = +box.dataset.index;
    blurDrag = {kind: e.target.dataset.corner ? 'resize' : 'move', corner: e.target.dataset.corner, index, r, px, py, start: displayRect(index, r)};
  } else blurDrag = {kind: 'new', r, px, py};
  $('frame').setPointerCapture(e.pointerId);
  e.preventDefault();
};
function dragRect(e) {
  const {kind, corner, r, px, py, start} = blurDrag;
  const nx = Math.max(0, Math.min(r.width, e.clientX - r.left)), ny = Math.max(0, Math.min(r.height, e.clientY - r.top));
  if (kind === 'new') return [Math.min(px, nx), Math.min(py, ny), Math.abs(nx - px), Math.abs(ny - py)];
  let [x, y, w, h] = start;
  if (kind === 'move') {
    x = Math.max(0, Math.min(r.width - w, x + nx - px)); y = Math.max(0, Math.min(r.height - h, y + ny - py));
    return [x, y, w, h];
  }
  let x1 = x + w, y1 = y + h;
  if (corner.includes('w')) x = nx; else x1 = nx;
  if (corner.includes('n')) y = ny; else y1 = ny;
  return [Math.min(x, x1), Math.min(y, y1), Math.abs(x1 - x), Math.abs(y1 - y)];
}
$('frame').onpointermove = e => {
  if (!blurDrag) return;
  const [x, y, w, h] = dragRect(e), r = blurDrag.r;
  const draft = $('dragBox');
  draft.hidden = false;
  Object.assign(draft.style, {left: x / r.width * 100 + '%', top: y / r.height * 100 + '%', width: w / r.width * 100 + '%', height: h / r.height * 100 + '%'});
  if (blurDrag.kind !== 'new') { const original = $('blurBoxes').children[blurDrag.index]; if (original) original.style.visibility = 'hidden'; }
};
$('frame').onpointerup = handle(async e => {
  if (!blurDrag) return;
  const drag = blurDrag, rect = dragRect(e);
  blurDrag = null;
  $('dragBox').hidden = true;
  const moved = Math.abs(e.clientX - drag.r.left - drag.px) > 3 || Math.abs(e.clientY - drag.r.top - drag.py) > 3;
  if (moved && rect[2] > 3 && rect[3] > 3) {
    const result = await api('/api/coordinates', {rect, display: [drag.r.width, drag.r.height], source: [state.info.width, state.info.height]});
    if (drag.kind === 'new') state.blur.push(result.rect); else state.blur[drag.index] = result.rect;
    state.visited[3] = true;
    updateCommand();
  }
  renderBlurs(); renderSteps();
});
$('frame').onpointercancel = () => { blurDrag = null; $('dragBox').hidden = true; renderBlurs(); };

/* ---------- playback ---------- */
$('play').onclick = handle(async () => {
  const video = $('stageVideo');
  if (!video.paused) { video.pause(); return; }
  const a = axis();
  if (state.time >= a.start + a.dur - .05) state.time = a.start;
  video.currentTime = state.time;
  try { await video.play(); } catch { notice('cannotPlay', 'error'); return; }
  state.playing = true; renderCanvas();
});
$('stageVideo').onpause = () => {
  if (!state.playing) return;
  state.playing = false; state.time = $('stageVideo').currentTime;
  setFrame(true); renderCanvas(); loadPeople().catch(error);
};
$('stageVideo').ontimeupdate = () => {
  if (!state.playing) return;
  state.time = $('stageVideo').currentTime;
  const a = axis();
  if (state.time >= a.start + a.dur) $('stageVideo').pause();
  renderTime();
};
let seekTimer;
function seek(at) {
  const a = axis();
  state.time = Math.max(a.start, Math.min(a.start + a.dur - .001, at));
  if (state.playing) $('stageVideo').currentTime = state.time;
  renderTime();
  if (!state.playing) {
    setFrame();
    clearTimeout(seekTimer);
    seekTimer = setTimeout(() => loadPeople().catch(error), 220);
    renderPeople();
  }
}
$('scrub').oninput = () => seek(+$('scrub').value);

/* ---------- jobs ---------- */
function modeStages(job) {
  if (!job) return [];
  if (job.mode === 'people') return ['prepare', 'pose'];
  if (job.mode === 'prepare') return state.stages.slice(0, state.stages.indexOf('segment') + 1);
  return job.render_only || state.renderOnly ? ['render'] : state.stages;
}
function jobProgress(job) {
  if (job.status === 'complete') return 100;
  const list = modeStages(job), i = list.indexOf(job.stage);
  return i < 0 ? 0 : Math.round((i + .5) / list.length * 100);
}
async function startJob(mode) {
  let p;
  if (mode === 'people') p = {input: state.input, speaker: $('speaker').value.trim() || 'Subject', target: [[1, 0]], face_identity: false};
  else if (mode === 'prepare') p = params('Subject');
  else p = params();
  const sig = mode === 'prepare' ? prepareSig() : null;
  const result = await api('/api/jobs', {params: p, mode});
  state.job = result; state.watching = result.id; state.jobStarted = Date.now(); state.renderOnly = !!result.render_only;
  if (mode === 'prepare') state.preparingSig = sig;
  if (mode === 'run') { state.queuedRun = false; state.results = null; state.showDone = false; state.resultLayout = p.layout; state.output = result.output; }
  clearTimeout(state.poll);
  renderAll();
  await poll();
}
async function poll() {
  clearTimeout(state.poll);
  const job = await api('/api/jobs');
  state.job = job;
  if (busy(job) && state.watching !== job.id) { state.watching = job.id; state.jobStarted = Date.now(); if (job.mode === 'run') { state.output = job.output; state.focus = 5; } }
  renderJob(); renderGenerate();
  if (job.mode === 'run') renderStage();
  if (busy(job)) { state.poll = setTimeout(() => poll().catch(error), 750); return; }
  if (state.watching !== job.id) return;
  state.watching = null;
  await finished(job);
}
async function finished(job) {
  const sig = state.preparingSig;
  state.preparingSig = null;
  if (job.status === 'complete') {
    if (job.mode === 'people') { state.peopleReady = true; state.peopleCache.clear(); state.peopleNote = ''; await loadPeople(); }
    else if (job.mode === 'prepare') { if (sig === prepareSig()) { state.preparedSig = sig; await timeline(job.output); } }
    else if (job.mode === 'run') {
      state.preparedSig = prepareSig();
      await timeline(job.output);
      showResults(job.output, state.resultLayout);
      state.focus = 5; state.showDone = true; state.visited[5] = true;
    }
  } else if (job.status === 'failed') { notice(job.mode === 'run' ? 'failed' : 'prepareFailed', 'error'); state.queuedRun = false; }
  else if (job.status === 'cancelled' && job.mode === 'run') notice('cancelled');
  if (state.queuedRun) { await startJob('run'); return; }
  renderAll();
  maybePrepare();
}
let prepareTimer;
function maybePrepare() { clearTimeout(prepareTimer); prepareTimer = setTimeout(() => autoPrepare().catch(() => {}), 900); }
async function autoPrepare() {
  // Background preparation only uses local models; description selection stays manual (API).
  if (!state.info || !state.target.length || state.focus === 2 || $('targetDesc').value.trim() || state.queuedRun) return;
  const sig = prepareSig();
  if (sig === state.preparedSig) return;
  if (busy()) {
    if (state.job.mode === 'prepare' && state.preparingSig !== sig && state.job.status === 'running') await api('/api/cancel', {});
    return; // finished() calls maybePrepare again
  }
  await startJob('prepare');
}
function renderJob() {
  const job = state.job;
  const running = busy(job);
  const background = running && (job.mode !== 'run' || currentView() !== 'job');
  $('bgBar').style.width = background ? jobProgress(job) + '%' : '0%';
  $('bgStatus').classList.toggle('busy', background);
  const action = $('bgAction');
  action.hidden = true;
  if (background) {
    text($('bgStatus'), `${t('doing')[job.stage] || t('stages')[job.stage] || t('loading')} ${jobProgress(job)}%`);
    action.hidden = job.status !== 'running'; text(action, t('cancel'));
    action.onclick = handle(async () => { state.queuedRun = false; await api('/api/cancel', {}); });
  } else if (state.timeline) {
    const rows = state.timeline.windows || [];
    const other = rows.filter(isOther).length;
    text($('bgStatus'), tf('cueCount', {n: rows.length, a: rows.length - other, b: other}));
  } else if (state.info && $('targetDesc').value.trim()) {
    text($('bgStatus'), '');
    action.hidden = false; text(action, t('prepare'));
    action.onclick = handle(() => startJob('prepare'));
  } else text($('bgStatus'), '');
  if (!job || job.mode !== 'run') return;
  const list = modeStages(job), current = list.indexOf(job.stage);
  $('progress').value = jobProgress(job);
  text($('jobStage'), state.queuedRun && !running ? t('waitingPrepare') : running ? `${t('stages')[job.stage] || job.stage} · ${Math.max(1, current + 1)}/${list.length}` : t(job.status));
  const seconds = Math.max(0, (Date.now() - state.jobStarted) / 1000);
  text($('jobElapsed'), running ? mmss(seconds) : $('jobElapsed').textContent);
  text($('jobCost'), costText(job));
  $('stageList').replaceChildren(...list.map((s, i) => el('li', job.status === 'complete' || i < current ? 'done' : i === current && running ? 'current' : '', t('stages')[s] || s)));
  text($('logs'), job.logs.join('\n'));
  $('cancel').hidden = !running && !state.queuedRun;
}
function costText(job) {
  if (busy(job)) return t('costPending');
  if (job.cost_unknown) return t('unknownCost');
  return `$${Number(job.cost || 0).toFixed(4)}` + (job.cached_cost != null ? ` · ${t('historicalCost')} $${Number(job.cached_cost).toFixed(4)}` : '');
}
setInterval(() => { if (busy() && state.job.mode === 'run') text($('jobElapsed'), mmss((Date.now() - state.jobStarted) / 1000)); }, 1000);

/* ---------- timeline ---------- */
async function timeline(output) {
  state.timeline = await api('/api/timeline?' + new URLSearchParams({output}));
  state.output = output;
  const a = axis();
  state.time = Math.min(Math.max(state.time, a.start), a.start + a.dur - .001);
  renderAll();
}
const timeOffset = () => auto() && state.timeline ? Number(state.timeline.actual_start || 0) : 0;
function isOther(row) {
  const offset = timeOffset(), a = row.t0 + offset, b = row.t1 + offset;
  return state.others.reduce((v, [x, y]) => v + Math.max(0, Math.min(b, y) - Math.max(a, x)), 0) >= (b - a) / 2;
}
function toggleInterval(a, b) {
  const overlap = state.others.reduce((sum, [x, y]) => sum + Math.max(0, Math.min(b, y) - Math.max(a, x)), 0);
  if (overlap >= (b - a) / 2) state.others = state.others.flatMap(([x, y]) => y <= a || x >= b ? [[x, y]] : [[x, Math.min(a, y)], [Math.max(b, x), y]].filter(([l, r]) => r - l > .00001));
  else state.others.push([a, b]);
  mergeIntervals();
  state.visited[4] = true;
  renderTimeline(); renderIntervals(); renderSteps(); renderJob(); updateCommand();
}
function mergeIntervals() {
  state.others.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const pair of state.others) {
    if (merged.length && pair[0] <= merged.at(-1)[1]) merged.at(-1)[1] = Math.max(merged.at(-1)[1], pair[1]);
    else merged.push([...pair]);
  }
  state.others = merged;
}
function renderIntervals() {
  const list = $('intervalList');
  list.replaceChildren();
  state.others.forEach(([a, b], i) => chip(list, `${t('otherShort')} ${a.toFixed(1)}–${b.toFixed(1)}s`, () => {
    state.others.splice(i, 1); renderTimeline(); renderIntervals(); renderJob(); updateCommand();
  }, () => seek(auto() ? a : clipOffset() + a)));
}
let thumbKey = '';
function niceStep(duration) { for (const s of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]) if (duration / s <= 9) return s; return 1200; }
function renderTimeline() {
  if (!state.info) { thumbKey = ''; for (const id of ['ticks', 'thumbs', 'absence', 'shotLines', 'anchorMarks', 'others']) $(id).replaceChildren(); $('cues').querySelectorAll('button').forEach(b => b.remove()); text($('cuePlaceholder'), ''); return; }
  const a = axis(), data = state.timeline, pct = v => Math.max(0, Math.min(100, v / a.dur * 100)) + '%';
  const ticks = $('ticks'); ticks.replaceChildren();
  const step = niceStep(a.dur);
  for (let s = Math.ceil(a.start / step) * step; s <= a.start + a.dur + .001; s += step) {
    const span = el('span', '', mmss(s)); span.style.left = pct(s - a.start);
    if (s - a.start < a.dur * .03) span.className = 'first'; else if (a.start + a.dur - s < a.dur * .03) span.className = 'last';
    ticks.append(span);
  }
  const width = $('thumbs').clientWidth || 1200, height = $('thumbs').clientHeight || 46;
  const count = Math.max(6, Math.min(30, Math.round(width / (height * state.info.width / state.info.height))));
  const key = [data ? data.media : state.input, a.start, a.dur, count].join('|');
  if (key !== thumbKey) {
    thumbKey = key;
    $('thumbs').replaceChildren(...Array.from({length: count}, (_, i) => {
      const img = el('img'); img.alt = ''; img.loading = 'lazy';
      const at = (i + .5) / count * a.dur;
      img.src = data ? frameUrl(data.media, at) : frameUrl(state.input, a.start + at);
      return img;
    }));
  }
  $('absence').replaceChildren(); $('shotLines').replaceChildren();
  for (const shot of data?.shots || []) {
    if (shot.t0 > .05) { const line = el('span'); line.style.left = pct(shot.t0); $('shotLines').append(line); }
    if (shot.label !== 'target') { const band = el('span'); band.style.left = pct(shot.t0); band.style.width = pct(shot.t1 - shot.t0); $('absence').append(band); }
  }
  $('anchorMarks').replaceChildren(...state.target.map(([n, at]) => { const mark = el('span', '', circled(n)); mark.style.left = pct(clipOffset() + at - a.start); return mark; }));
  $('cues').querySelectorAll('button').forEach(b => b.remove());
  const offset = timeOffset();
  for (const row of data?.windows || []) {
    const other = isOther(row);
    const cue = button('', () => { toggleInterval(row.t0 + offset, row.t1 + offset); seek(a.start + row.t0); if (state.focus !== 4) focusStep(4); }, other ? 'other' : '');
    cue.append(el('span', '', row.subtitle || row.id));
    cue.style.left = pct(row.t0); cue.style.width = pct(row.t1 - row.t0);
    cue.title = `${row.t0.toFixed(2)}–${row.t1.toFixed(2)}s · ${row.subtitle || ''} · ${other ? t('otherTalk') : ($('speaker').value.trim() || t('targetTalk'))}`;
    cue.onpointerdown = e => e.stopPropagation();
    $('cues').append(cue);
  }
  $('others').replaceChildren();
  for (const [x, y] of state.others) {
    const left = Math.max(0, x - offset), right = Math.min(a.dur, y - offset);
    if (right <= left) continue;
    const bar = el('span'); bar.style.left = pct(left); bar.style.width = pct(right - left); $('others').append(bar);
  }
  const placeholder = data ? '' : busy() && state.job.mode === 'prepare' ? t('cuesLoading') : state.target.length ? t('cuesAfterPrepare') : t('cuesNeedTarget');
  text($('cuePlaceholder'), placeholder);
  renderTime();
}
let timeDrag = null;
$('timeline').onpointerdown = e => {
  if (!state.info || e.target.closest('button') || e.button !== 0) return;
  const r = $('timeline').getBoundingClientRect();
  timeDrag = {x: e.clientX - r.left, r};
  $('timeline').setPointerCapture(e.pointerId);
};
$('timeline').onpointermove = e => {
  if (!timeDrag) return;
  const x = e.clientX - timeDrag.r.left;
  if (!state.timeline) { seek(axis().start + Math.max(0, Math.min(1, x / timeDrag.r.width)) * axis().dur); return; }
  if (Math.abs(x - timeDrag.x) < 5) return;
  const brush = $('brush'); brush.hidden = false;
  const left = Math.max(0, Math.min(x, timeDrag.x)), right = Math.min(timeDrag.r.width, Math.max(x, timeDrag.x));
  brush.style.left = left + 'px'; brush.style.width = right - left + 'px';
};
$('timeline').onpointerup = handle(async e => {
  if (!timeDrag) return;
  const {x, r} = timeDrag;
  timeDrag = null; $('brush').hidden = true;
  const end = e.clientX - r.left, a = axis();
  if (Math.abs(end - x) < 5 || !state.timeline) { seek(a.start + Math.max(0, Math.min(1, end / r.width)) * a.dur); return; }
  const result = await api('/api/coordinates', {kind: 'timeline', a: x, b: end, width: r.width, duration: state.timeline.duration});
  const offset = timeOffset();
  state.others.push(...result.intervals.map(([p, q]) => [p + offset, q + offset]));
  mergeIntervals();
  state.visited[4] = true;
  if (state.focus !== 4) await focusStep(4); else { renderTimeline(); renderIntervals(); renderSteps(); renderJob(); }
  updateCommand();
});
$('timeline').onpointercancel = () => { timeDrag = null; $('brush').hidden = true; };

/* ---------- estimate, summary, results ---------- */
let commandTimer;
function updateCommand() {
  clearTimeout(commandTimer);
  commandTimer = setTimeout(async () => {
    try { const p = params(); state.command = await api('/api/command', {params: p}); }
    catch { state.command = null; }
    renderEstimate(); renderGenerate();
    if (currentView() === 'summary') renderSummary();
  }, 250);
}
function renderEstimate() {
  if (!state.info) { text($('estimate'), t('estimateNone')); text($('estimateBasis'), ''); text($('ctaEstimate'), ''); return; }
  const duration = auto() ? state.info.duration : Math.max(0, +$('rangeEnd').value - +$('rangeStart').value);
  const renderOnly = state.command?.args?.slice(-2).join(' ') === '--from render';
  // Measured on the 52 s Tim sample: render ≈ 2.5× clip length, API ≈ $0.012.
  const minutes = Math.max(1, Math.round(duration * (renderOnly ? 2.5 : 3.2) / 60));
  const value = renderOnly ? tf('estimateRender', {m: minutes}) : tf('estimateFull', {m: minutes, c: Math.max(.01, duration * .00023).toFixed(2)});
  text($('estimate'), value); text($('ctaEstimate'), value);
  text($('estimateBasis'), t(renderOnly ? 'estimateCached' : 'estimateBasis'));
}
async function summary() {
  try { state.command = await api('/api/command', {params: params()}); } catch (e) { state.command = null; renderSummary(e); return; }
  renderEstimate();
  renderSummary();
}
function renderSummary(problem) {
  const list = $('summary');
  list.replaceChildren();
  let p;
  try { p = params(); } catch (e) { problem = problem || e; }
  if (problem || !p) {
    list.append(el('dt', '', t('missingPrefix')), el('dd', '', missing().map(([k]) => t(k)).join('、') || t(problem?.message || 'operation_failed')));
    text($('command'), '');
    return;
  }
  for (const [label, value] of [['source', p.input], ['speaker', p.speaker], ['title', p.title || tf('defaultTitle', {name: p.speaker})],
    ['range', p.start === 'auto' ? t('autoStart') : `${p.start.toFixed(2)}s → ${p.duration == null ? t('end') : (p.start + p.duration).toFixed(2) + 's'}`],
    ['anchors', p.target_desc || p.target.map(([n, at]) => `${circled(n)} @ ${at.toFixed(2)}s`).join('，')],
    ['faceIdentity', p.face_identity ? '✓' : '—'], ['blur', p.blur.length ? tf('blurCount', {n: p.blur.length}) : '—'],
    ['burned', p.subtitles === 'off' ? '✓' : '—'], ['others', p.others_speaking.map(([a, b]) => `${a.toFixed(2)}–${b.toFixed(2)}s`).join('，') || '—'],
    ['languageLabel', p.lang === 'zh' ? '中文' : 'English'], ['layoutLabel', t({both: 'layoutBoth', h: 'layoutH', v: 'layoutV'}[p.layout])]]) {
    list.append(el('dt', '', t(label)), el('dd', '', value));
  }
  text($('command'), state.command?.command || '');
}
function showResults(output, layout = 'both') {
  state.results = output; state.output = output;
  for (const [type, id] of [['h', 'H'], ['v', 'V']]) {
    const present = layout === 'both' || layout === type;
    $('figure' + id).hidden = !present; $('download' + id).hidden = !present;
    if (!present) continue;
    const path = output + `/output_${type}.mp4`;
    $('output' + id).src = media(path);
    $('download' + id).href = media(path) + '?download=1';
    $('download' + id).download = `output_${type}.mp4`;
  }
  text($('doneCost'), state.job && state.job.mode === 'run' && state.job.status === 'complete' ? `${t('cost')} ${costText(state.job)}` : '');
}

/* ---------- settings and history drawers ---------- */
function openDrawer(id) {
  for (const d of ['drawerHistory', 'drawerSettings']) $(d).hidden = d !== id;
  $('scrim').hidden = false;
  $(id).querySelector('.close').focus();
}
function closeDrawers() { $('drawerHistory').hidden = true; $('drawerSettings').hidden = true; $('scrim').hidden = true; }
function repair(row) {
  if (row.name === 'OPENROUTER_API_KEY') return t('repairKey');
  if (row.name === 'ffmpeg') return t('repairFfmpeg');
  if (row.name === 'Python') return t('repairPython');
  if (row.name === 'RapidOCR CPU') return t('repairOcr');
  return t('repairModel');
}
function renderChecks() {
  $('checks').replaceChildren();
  for (const check of state.checks) {
    const row = el('div', 'check-row' + (check.ok ? '' : ' fail'));
    const left = el('div');
    left.append(el('span', '', t('checksNames')[check.name] || check.name));
    if (!check.ok) left.append(el('small', '', repair(check)));
    row.append(left, el('span', 'status', t(check.ok ? 'checkOk' : 'checkFail')));
    $('checks').append(row);
  }
  const has = state.checks.find(c => c.name === 'OPENROUTER_API_KEY')?.ok;
  text($('keyStatus'), t(has ? 'set' : 'unset'));
  $('keyStatus').classList.toggle('ok', !!has);
}
async function history() {
  const rows = await api('/api/history');
  const list = $('historyList');
  list.replaceChildren();
  if (!rows.length) list.append(el('p', 'hint', t('emptyHistory')));
  for (const row of rows) {
    const card = el('div', 'history-card');
    const thumb = el('div', 'thumb');
    const path = row.h || row.v;
    if (path) { const video = document.createElement('video'); video.preload = 'metadata'; video.muted = true; video.src = media(path) + '#t=1'; thumb.append(video); }
    const meta = el('div', 'meta');
    meta.append(el('h3', '', row.params?.title || row.output.split('/').pop()), el('span', 'path', row.output));
    const actions = el('div', 'row');
    if (row.params) actions.append(button(t('reopen'), () => restore(row.params, row.output, row), 'primary'));
    actions.append(button(t('reveal'), () => api('/api/reveal', {path: path})));
    meta.append(actions);
    card.append(thumb, meta);
    list.append(card);
  }
}
async function restore(p, output, row) {
  closeDrawers();
  await selectSource(p.input, true);
  $('speaker').value = p.speaker; $('title').value = p.title || ''; $('title').dataset.edited = '1';
  $('autoStart').checked = p.start === 'auto';
  $('rangeStart').value = p.start === 'auto' ? 0 : p.start;
  $('rangeEnd').value = p.duration == null ? state.info.duration : (Number(p.start) || 0) + p.duration;
  $('faceIdentity').checked = p.face_identity; $('targetDesc').value = p.target_desc || '';
  $('burned').checked = p.subtitles === 'off'; $('subtitleSource').value = p.subtitle_source || 'auto';
  $('narrationLang').value = p.lang || state.defaultLang; $('layout').value = p.layout || 'both';
  state.outputDir = p.output_dir; state.reframe = p.reframe || 'auto';
  rangeChange();
  state.target = p.target; state.blur = p.blur; state.others = p.others_speaking;
  state.time = clipOffset();
  state.visited = {1: true, 2: true, 3: true, 4: true, 5: true};
  if (output) {
    await timeline(output);
    state.preparedSig = prepareSig();
    showResults(output, row && row.h && row.v ? 'both' : row && row.v ? 'v' : row && row.h ? 'h' : p.layout);
    state.focus = 5; state.showDone = true;
  }
  renderAll();
  updateCommand();
}

/* ---------- events ---------- */
$('upload').onchange = handle(e => upload(e.target.files[0]));
for (const target of [$('drop'), $('viewPicker')]) {
  target.ondragover = e => { e.preventDefault(); $('drop').classList.add('over'); };
  target.ondragleave = () => $('drop').classList.remove('over');
  target.ondrop = handle(async e => { e.preventDefault(); e.stopPropagation(); $('drop').classList.remove('over'); await upload(e.dataTransfer.files[0]); });
}
for (const id of ['rangeStart', 'rangeEnd', 'autoStart']) $(id).oninput = rangeChange;
for (const id of ['rangeStart', 'rangeEnd', 'autoStart', 'faceIdentity', 'subtitleSource']) $(id).addEventListener('change', () => { updateCommand(); maybePrepare(); });
for (const id of ['burned', 'narrationLang', 'layout', 'title']) $(id).addEventListener('change', updateCommand);
$('burned').addEventListener('change', () => renderSteps());
$('targetDesc').oninput = () => { renderSteps(); renderGenerate(); renderJob(); updateCommand(); };
$('speaker').oninput = () => {
  const name = $('speaker').value;
  if (!$('title').dataset.edited) $('title').value = name.trim() ? tf('defaultTitle', {name}) : '';
  renderSteps(); renderGenerate(); renderPeople(); text($('legendTarget'), name.trim() || t('targetTalk'));
  updateCommand();
};
$('title').oninput = () => { $('title').dataset.edited = '1'; };
$('narrationLang').addEventListener('change', () => { if (!$('title').dataset.edited && $('speaker').value.trim()) $('title').value = tf('defaultTitle', {name: $('speaker').value}); });
async function generate() {
  const miss = missing();
  if (miss.length) { notice(t('missingPrefix') + ' ' + miss.map(([k]) => t(k)).join('、'), 'error'); await focusStep(miss[0][1]); return; }
  params();
  if (busy()) {
    if (state.job.mode === 'run') return;
    if (state.job.mode === 'prepare' && state.preparingSig !== prepareSig()) await api('/api/cancel', {});
    state.queuedRun = true; state.focus = 5; renderAll(); return;
  }
  state.focus = 5;
  await startJob('run');
}
$('generate').onclick = handle(generate);
$('generateBig').onclick = handle(generate);
$('cancel').onclick = handle(async () => { if (state.queuedRun && !(busy() && state.job.mode === 'run')) { state.queuedRun = false; renderAll(); return; } await api('/api/cancel', {}); });
$('historyNav').onclick = handle(async () => { openDrawer('drawerHistory'); await history(); });
$('settingsNav').onclick = () => openDrawer('drawerSettings');
$('blockers').onclick = () => openDrawer('drawerSettings');
$('scrim').onclick = closeDrawers;
document.querySelectorAll('[data-close]').forEach(b => { b.onclick = closeDrawers; });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawers(); });
$('language').onclick = handle(() => language(state.lang === 'zh' ? 'en' : 'zh'));
$('copy').onclick = handle(async () => { await navigator.clipboard.writeText($('command').textContent); notice('copied'); });
$('reveal').onclick = handle(() => api('/api/reveal', {path: state.results + '/output_' + ($('figureH').hidden ? 'v' : 'h') + '.mp4'}));
$('backToEdit').onclick = handle(() => focusStep(2));
$('saveSettings').onclick = handle(async () => {
  const data = {lang: $('defaultLanguage').value, output_dir: $('outputDir').value};
  if ($('key').value) data.key = $('key').value;
  try {
    const result = await api('/api/settings', data);
    state.checks = result.checks; state.outputDir = result.settings.output_dir; state.defaultLang = result.settings.lang;
    await language(result.settings.lang);
    notice('saved');
  } finally { $('key').value = ''; }
});
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderTimeline, 150); });

(async () => {
  const boot = await api('/api/bootstrap');
  state.token = boot.token; state.checks = boot.checks; state.stages = boot.stages;
  state.outputDir = boot.settings.output_dir; state.defaultLang = boot.settings.lang;
  $('outputDir').value = state.outputDir; $('defaultLanguage').value = boot.settings.lang; $('narrationLang').value = boot.settings.lang;
  await language(boot.settings.lang);
  await poll();
})().catch(error);
