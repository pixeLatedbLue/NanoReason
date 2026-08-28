
'use strict';

const $ = (sel, root) => (root || document).querySelector(sel);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

const isNum = (v) => typeof v === 'number' && isFinite(v);

function fmt(v, digits) {
  return isNum(v) ? v.toFixed(digits === undefined ? 3 : digits) : '—';
}

function fmtSigned(v, digits) {
  if (!isNum(v)) return '—';
  const d = digits === undefined ? 3 : digits;
  return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
}

function fmtPValue(p) {
  if (!isNum(p)) return '—';
  return p < 0.0001 ? 'p < 0.0001' : 'p = ' + p.toFixed(4);
}

function fmtDate(unix) {
  if (!isNum(unix)) return '—';
  try { return new Date(unix * 1000).toLocaleString(); } catch (e) { return '—'; }
}

function chip(text, kind) { return el('span', kind ? 'chip chip--' + kind : 'chip', text); }

function setBusy(host, busy) {
  if (host) host.setAttribute('aria-busy', busy ? 'true' : 'false');
}

function showSkeleton(host, tall) {
  clear(host);
  setBusy(host, true);
  const card = el('div', 'card');
  card.appendChild(el('div', 'skel skel--line'));
  card.appendChild(el('div', 'skel skel--line'));
  if (tall) card.appendChild(el('div', 'skel skel--tall'));
  host.appendChild(card);
}

function toast(title, message) {
  const host = $('#toasts');
  const box = el('div', 'toast');
  const body = el('div', 'toast__body');
  body.appendChild(el('div', 'toast__title', title));
  if (message) body.appendChild(el('div', 'toast__msg', message));
  const close = el('button', 'toast__x', '×');
  close.type = 'button';
  close.setAttribute('aria-label', 'Dismiss notification');
  close.addEventListener('click', () => box.remove());
  box.appendChild(body);
  box.appendChild(close);
  host.appendChild(box);
  setTimeout(() => box.remove(), 10000);
}

async function toastHttp(res, what) {
  let detail = '';
  try {
    const body = await res.clone().json();
    if (body && typeof body === 'object') detail = String(body.detail || body.message || '');
  } catch (e) {  }
  if (!detail) {
    try { detail = (await res.text()).slice(0, 300); } catch (e) {  }
  }
  toast('HTTP ' + res.status + (res.statusText ? ' ' + res.statusText : ''),
    (what ? what + ' — ' : '') + (detail || 'The server returned no detail.'));
}

function toastNetwork(err, what) {
  toast('Request failed', (what ? what + ' — ' : '') + String((err && err.message) || err));
}

async function loadHealth() {
  const badge = $('#engine-badge');
  const meta = $('#engine-meta');
  try {
    const res = await fetch('/api/health', { headers: { Accept: 'application/json' } });
    if (!res.ok) { await toastHttp(res, 'engine health'); throw new Error('HTTP ' + res.status); }
    const health = await res.json();
    const engine = (health && health.engine) || {};
    badge.className = 'badge';
    if (engine.is_demo) {
      badge.classList.add('badge--warn');
      badge.textContent = 'DEMO ENGINE — scripted output, not a trained model';
    } else {
      badge.classList.add('badge--ok');
      badge.textContent = 'LIVE ENGINE — ' + (engine.model || 'model unnamed');
    }
    const bits = [];
    if (engine.kind) bits.push('kind: ' + engine.kind);
    if (engine.device) bits.push('device: ' + engine.device);
    bits.push('adapter: ' + (engine.adapter || 'none'));
    meta.textContent = bits.join('  ·  ');
    $('#version').textContent = health && health.version ? 'v' + health.version : '';
  } catch (err) {
    badge.className = 'badge badge--warn';
    badge.textContent = 'ENGINE UNAVAILABLE — ' + String((err && err.message) || err);
    meta.textContent = '';
  }
}

const REWARD_MIN = -1;
const REWARD_MAX = 3.5;
const DIVERSITY_THRESHOLD = 0.35;

function rewardGeometry(value) {
  const span = REWARD_MAX - REWARD_MIN;
  const zero = ((0 - REWARD_MIN) / span) * 100;
  const clamped = Math.max(REWARD_MIN, Math.min(REWARD_MAX, value));
  const point = ((clamped - REWARD_MIN) / span) * 100;
  return { zero: zero, left: Math.min(zero, point), width: Math.abs(point - zero), negative: clamped < 0 };
}

function rewardRow(label, value, isTotal) {
  const row = el('div', isTotal ? 'rw rw--total' : 'rw');
  row.appendChild(el('div', 'rw__label', label));
  const track = el('div', 'rw__track');
  if (isNum(value)) {
    const geo = rewardGeometry(value);
    const zeroTick = el('span', 'rw__zero');
    zeroTick.style.left = geo.zero + '%';
    const fill = el('span', geo.negative ? 'rw__fill rw__fill--neg' : 'rw__fill');
    fill.style.left = geo.left + '%';
    fill.style.width = geo.width + '%';
    track.appendChild(zeroTick);
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el('div', 'rw__val', fmtSigned(value)));
  } else {
    track.appendChild(el('span', 'rw__na', 'n/a — no gold answer'));
    row.appendChild(track);
    row.appendChild(el('div', 'rw__val rw__val--na', 'n/a'));
  }
  return row;
}

function answerSection(analysis) {
  const sub = el('section', 'sub');
  const line = el('div', 'ans');
  line.appendChild(el('span', 'ans__label', 'Final answer'));

  const hasAnswer = analysis.answer !== null && analysis.answer !== undefined && analysis.answer !== '';
  line.appendChild(el('span', hasAnswer ? 'ans__value' : 'ans__value ans__value--none',
    hasAnswer ? analysis.answer : 'no answer parsed'));

  const chips = el('div', 'ans__chips');
  if (analysis.answer_source === 'marker') {
    chips.appendChild(chip('#### marker', 'accent'));
  } else if (analysis.answer_source === 'fallback') {
    chips.appendChild(chip('fallback: last number', 'warn'));
  }
  chips.appendChild(analysis.format_ok
    ? chip('✓ format ok', 'ok')
    : chip('✗ format not followed', 'bad'));

  if (analysis.correct === true) {
    chips.appendChild(chip('✓ correct', 'ok'));
  } else if (analysis.correct === false) {
    chips.appendChild(chip('✗ incorrect', 'bad'));
  }
  line.appendChild(chips);
  sub.appendChild(line);

  const goldText = (analysis.gold === null || analysis.gold === undefined || analysis.gold === '')
    ? 'No gold answer was supplied, so correctness is not scored — only the reasoning is checked.'
    : 'Gold answer: ' + analysis.gold;
  sub.appendChild(el('p', 'hint hint--block', goldText));
  return sub;
}

function rewardSection(analysis) {
  const rewards = analysis.rewards || {};
  const sub = el('section', 'sub');
  sub.appendChild(el('h3', 'h3', 'Reward breakdown'));
  sub.appendChild(rewardRow('outcome', rewards.outcome, false));
  sub.appendChild(rewardRow('format', rewards.format, false));
  sub.appendChild(rewardRow('process', rewards.process, false));
  sub.appendChild(rewardRow('diversity', rewards.diversity, false));
  sub.appendChild(rewardRow('total', rewards.total, true));
  sub.appendChild(el('p', 'hint hint--block',
    'Bars share one scale from ' + fmtSigned(REWARD_MIN, 1) + ' to ' + fmtSigned(REWARD_MAX, 1) +
    '; the tick is zero, and bars to its left are penalties.'));
  return sub;
}

function stepSection(analysis) {
  const steps = Array.isArray(analysis.steps) ? analysis.steps : [];
  const sub = el('section', 'sub');
  sub.appendChild(el('h3', 'h3', 'Arithmetic steps'));

  if (!steps.length) {
    sub.appendChild(el('div', 'empty', 'No arithmetic steps were found, so the verifier had nothing to check. ' +
      'A completion with no checkable equations earns no process reward.'));
    return sub;
  }

  const scroller = el('div', 'scroll-x');
  const table = el('table', 'tbl');
  const head = el('thead');
  const headRow = el('tr');
  ['#', 'Equation', 'Computed', 'Verified', 'Note'].forEach((title) => {
    const th = el('th', null, title);
    th.scope = 'col';
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);

  const body = el('tbody');
  steps.forEach((step) => {
    const tr = el('tr', step.ok ? null : 'row--bad');
    tr.appendChild(el('td', 'num', isNum(step.index) ? step.index : '—'));

    const written = step.raw ? step.raw
      : [step.lhs, step.op, step.rhs, '=', step.stated].filter((p) => p !== null && p !== undefined).join(' ');
    tr.appendChild(el('td', 'num', written));

    tr.appendChild(el('td', 'num', isNum(step.computed) ? String(step.computed) : '—'));

    const verdict = el('td');
    const mark = el('span', step.ok ? 'mark mark--ok' : 'mark mark--bad',
      step.ok ? '✓ correct' : '✗ wrong');
    verdict.appendChild(mark);
    tr.appendChild(verdict);

    const note = el('td');
    if (step.duplicate) note.appendChild(chip('duplicate', 'warn'));
    else note.appendChild(el('span', 'stack__sub', '—'));
    tr.appendChild(note);

    body.appendChild(tr);
  });
  table.appendChild(body);

  const caption = el('caption', null,
    fmt(analysis.steps_correct, 0) + ' of ' + fmt(analysis.steps_total, 0) + ' steps verified  ·  ' +
    fmt(analysis.steps_unique, 0) + ' unique');
  table.insertBefore(caption, table.firstChild);

  scroller.appendChild(table);
  sub.appendChild(scroller);
  return sub;
}

function diversitySection(analysis) {
  const value = analysis.token_diversity;
  const sub = el('section', 'sub');
  sub.appendChild(el('h3', 'h3', 'Token diversity'));

  const row = el('div', 'meter__row');
  const meter = el('div', 'meter');
  meter.setAttribute('role', 'img');
  const pct = isNum(value) ? Math.max(0, Math.min(1, value)) * 100 : 0;
  const low = isNum(value) && value < DIVERSITY_THRESHOLD;
  const fill = el('span', low ? 'meter__fill meter__fill--low' : 'meter__fill');
  fill.style.width = 'calc(' + pct + '% - 4px)';
  meter.appendChild(fill);
  const mark = el('span', 'meter__mark');
  mark.style.left = (DIVERSITY_THRESHOLD * 100) + '%';
  meter.appendChild(mark);
  meter.setAttribute('aria-label', 'Token diversity ' + fmt(value) +
    ' against a collapse threshold of ' + DIVERSITY_THRESHOLD);
  row.appendChild(meter);
  row.appendChild(el('span', 'rw__val', fmt(value)));
  sub.appendChild(row);

  const verdict = isNum(value)
    ? (low ? '✗ below the ' + DIVERSITY_THRESHOLD + ' collapse threshold — the model is repeating itself'
      : '✓ above the ' + DIVERSITY_THRESHOLD + ' collapse threshold')
    : 'no diversity score reported';
  sub.appendChild(el('p', 'meter__cap', verdict + '  (the tick marks ' + DIVERSITY_THRESHOLD + ').'));
  return sub;
}

function renderAnalysis(host, analysis) {
  clear(host);
  setBusy(host, false);
  if (!analysis || typeof analysis !== 'object') {
    host.appendChild(el('div', 'card', 'The server returned no analysis for this request.'));
    return;
  }
  const card = el('div', 'card');
  card.appendChild(answerSection(analysis));
  card.appendChild(rewardSection(analysis));
  card.appendChild(stepSection(analysis));
  card.appendChild(diversitySection(analysis));
  host.appendChild(card);
}

const EXAMPLES = [
  {
    label: 'clips (two steps)',
    text: 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. ' +
      'How many clips did Natalia sell altogether in April and May?'
  },
  {
    label: 'bolts of fibre',
    text: 'A robe takes 2 bolts of blue fibre and half that much white fibre. How many bolts does it take in total?'
  },
  {
    label: 'weekly wage',
    text: 'Weng earns 12 dollars an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?'
  },
  {
    label: 'example with a bad step',
    trap: true,
    text: 'A trapme warehouse receives 12 boxes holding 8 widgets each. 17 widgets arrive damaged and are thrown away. ' +
      'How many usable widgets are left?'
  }
];

function buildExamples() {
  const row = $('#solve-examples');
  EXAMPLES.forEach((example) => {
    const button = el('button', example.trap ? 'ex ex--trap' : 'ex', example.label);
    button.type = 'button';
    button.title = example.text;
    button.addEventListener('click', () => {
      $('#solve-input').value = example.text;
      $('#solve-input').focus();
    });
    row.appendChild(button);
  });
}

function parseFrame(frame) {
  const payload = frame.split(/\r?\n/)
    .filter((line) => line.indexOf('data:') === 0)
    .map((line) => line.slice(5).trim())
    .join('');
  if (!payload) return null;
  try { return JSON.parse(payload); } catch (e) { return null; }
}

async function runSolve() {
  const question = $('#solve-input').value.trim();
  if (!question) { toast('Nothing to solve', 'Enter a word problem first.'); return; }

  const params = new URLSearchParams({ question: question });
  const maxTokens = parseInt($('#solve-tokens').value, 10);
  if (Number.isFinite(maxTokens) && maxTokens > 0) params.set('max_new_tokens', String(maxTokens));
  const gold = $('#solve-gold').value.trim();
  if (gold) params.set('gold', gold);

  const button = $('#solve-btn');
  const status = $('#solve-status');
  const output = $('#solve-output');
  const host = $('#solve-analysis');
  button.disabled = true;
  status.textContent = 'streaming…';
  output.textContent = '';
  clear(host);
  setBusy(host, true);

  let sawTerminal = false;
  try {
    const res = await fetch('/api/solve/stream?' + params.toString(), { headers: { Accept: 'text/event-stream' } });
    if (!res.ok) { await toastHttp(res, 'streaming solve'); status.textContent = 'failed'; return; }
    if (!res.body) {
      toast('Streaming unsupported', 'The response carried no readable body.');
      status.textContent = 'failed';
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      for (;;) {
        const split = /\r?\n\r?\n/.exec(buffer);
        if (!split) break;
        const frame = buffer.slice(0, split.index);
        buffer = buffer.slice(split.index + split[0].length);
        const event = parseFrame(frame);
        if (!event) continue;
        if (event.type === 'token') {
          output.textContent += (event.text || '');
          output.scrollTop = output.scrollHeight;
        } else if (event.type === 'done') {
          sawTerminal = true;
          if (typeof event.completion === 'string') output.textContent = event.completion;
          renderAnalysis(host, event.analysis);
          status.textContent = isNum(event.latency_ms)
            ? 'completed in ' + Math.round(event.latency_ms) + ' ms' : 'completed';
        } else if (event.type === 'error') {
          sawTerminal = true;
          toast('Stream error', String(event.message || 'The server reported an error frame.'));
          status.textContent = 'failed';
        }
      }
    }
    if (!sawTerminal) {
      toast('Stream ended early', 'The connection closed before a final frame arrived.');
      status.textContent = 'interrupted';
    }
  } catch (err) {
    toastNetwork(err, 'streaming solve');
    status.textContent = 'failed';
  } finally {
    button.disabled = false;
    setBusy(host, false);
    if (!host.firstChild) clear(host);
  }
}

async function runVerify() {
  const text = $('#verify-input').value;
  if (!text.trim()) { toast('Nothing to verify', 'Paste some reasoning first.'); return; }
  const gold = $('#verify-gold').value.trim();

  const button = $('#verify-btn');
  const status = $('#verify-status');
  const host = $('#verify-analysis');
  button.disabled = true;
  status.textContent = 'verifying…';
  showSkeleton(host, true);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ text: text, gold: gold || null })
    });
    if (!res.ok) { await toastHttp(res, 'analyze'); clear(host); status.textContent = 'failed'; return; }
    renderAnalysis(host, await res.json());
    status.textContent = 'verified on the CPU — no model involved';
  } catch (err) {
    toastNetwork(err, 'analyze');
    clear(host);
    status.textContent = 'failed';
  } finally {
    button.disabled = false;
    setBusy(host, false);
  }
}

let RUNS = [];

function taskChips(tasks) {
  const box = el('div', 'tasks');
  const names = tasks && typeof tasks === 'object' ? Object.keys(tasks) : [];
  if (!names.length) { box.appendChild(el('span', 'stack__sub', 'no tasks recorded')); return box; }
  names.forEach((name) => {
    const stat = tasks[name] || {};
    box.appendChild(chip(name + '  ' + fmt(stat.accuracy) +
      '  (' + fmt(stat.correct, 0) + '/' + fmt(stat.total, 0) + ')'));
  });
  return box;
}

function renderRunsTable() {
  const host = $('#runs-host');
  clear(host);
  setBusy(host, false);

  if (!RUNS.length) {
    const empty = el('div', 'empty');
    empty.appendChild(el('strong', null, 'No evaluation runs found. '));
    empty.appendChild(document.createTextNode(
      'Nothing is shown here because nothing has been measured — no numbers are invented to fill the gap. ' +
      'Evaluations need a GPU; run '));
    empty.appendChild(el('code', null, 'python -m nanoreason.evaluate'));
    empty.appendChild(document.createTextNode(
      ' on a GPU box and the resulting JSON in results/ will appear here.'));
    host.appendChild(empty);
    return;
  }

  const scroller = el('div', 'scroll-x');
  const table = el('table', 'tbl');
  const head = el('thead');
  const headRow = el('tr');
  ['Run', 'Base model', 'Adapter', 'Created', 'Per-task accuracy'].forEach((title) => {
    const th = el('th', null, title);
    th.scope = 'col';
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);

  const body = el('tbody');
  RUNS.forEach((run) => {
    const tr = el('tr');
    const first = el('td');
    const stack = el('div', 'stack');
    stack.appendChild(el('span', null, run.run_name || run.name || '—'));
    stack.appendChild(el('span', 'stack__sub', run.name || ''));
    if (run.config_path) stack.appendChild(el('span', 'stack__sub', run.config_path));
    first.appendChild(stack);
    tr.appendChild(first);
    tr.appendChild(el('td', 'num', run.base_model || '—'));
    tr.appendChild(el('td', 'num', run.adapter || 'none'));
    tr.appendChild(el('td', null, fmtDate(run.created_at_unix)));
    const tasksCell = el('td');
    tasksCell.appendChild(taskChips(run.tasks));
    tr.appendChild(tasksCell);
    body.appendChild(tr);
  });
  table.appendChild(body);
  scroller.appendChild(table);
  host.appendChild(scroller);
}

function fillSelects() {
  [$('#cmp-baseline'), $('#cmp-candidate')].forEach((select, index) => {
    clear(select);
    if (!RUNS.length) {
      const option = el('option', null, 'no runs available');
      option.value = '';
      select.appendChild(option);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    RUNS.forEach((run) => {
      const option = el('option', null, run.name + (run.adapter ? '  (adapter)' : '  (base)'));
      option.value = run.name;
      select.appendChild(option);
    });
    select.selectedIndex = RUNS.length > 1 ? (index === 0 ? 1 : 0) : 0;
  });
  $('#cmp-btn').disabled = RUNS.length < 2;
}

async function loadRuns() {
  const host = $('#runs-host');
  showSkeleton(host, false);
  try {
    const res = await fetch('/api/runs', { headers: { Accept: 'application/json' } });
    if (!res.ok) { await toastHttp(res, 'runs'); RUNS = []; }
    else {
      const data = await res.json();
      RUNS = data && Array.isArray(data.runs) ? data.runs : [];
    }
  } catch (err) {
    toastNetwork(err, 'runs');
    RUNS = [];
  }
  renderRunsTable();
  fillSelects();
}

function statCell(stat) {
  const td = el('td', 'num');
  if (!stat || typeof stat !== 'object') { td.textContent = '—'; return td; }
  const stack = el('div', 'stack');
  const line = el('span');
  line.appendChild(document.createTextNode(fmt(stat.accuracy) + ' '));
  line.appendChild(el('span', 'ci', '[' + fmt(stat.ci_low) + ', ' + fmt(stat.ci_high) + ']'));
  stack.appendChild(line);
  stack.appendChild(el('span', 'stack__sub', fmt(stat.correct, 0) + ' / ' + fmt(stat.total, 0) + ' correct'));
  td.appendChild(stack);
  return td;
}

function pairedCell(paired) {
  const td = el('td');
  if (!paired || typeof paired !== 'object') {
    td.appendChild(el('span', 'stack__sub', 'no paired items — unpaired'));
    return td;
  }
  const stack = el('div', 'stack');
  stack.appendChild(paired.significant
    ? chip('✓ significant', 'ok')
    : chip('✗ not significant', 'bad'));
  stack.appendChild(el('span', 'stack__sub', fmtPValue(paired.p_value)));
  stack.appendChild(el('span', 'stack__sub',
    fmt(paired.n_both, 0) + ' paired · ' +
    '+' + fmt(paired.only_candidate, 0) + ' candidate-only · ' +
    '+' + fmt(paired.only_baseline, 0) + ' baseline-only'));
  td.appendChild(stack);
  return td;
}

function renderCompare(data) {
  const host = $('#compare-host');
  clear(host);
  setBusy(host, false);
  const rows = data && Array.isArray(data.rows) ? data.rows : [];

  const heading = el('p', 'hint hint--block',
    'candidate ' + (data.candidate || '?') + '  vs  baseline ' + (data.baseline || '?') +
    '  at a threshold of ' + fmt(data.min_improvement));
  host.appendChild(heading);

  if (!rows.length) {
    host.appendChild(el('div', 'empty', 'These two runs share no task in common, so there is nothing to compare.'));
    return;
  }

  const scroller = el('div', 'scroll-x');
  const table = el('table', 'tbl');
  const head = el('thead');
  const headRow = el('tr');
  ['Task', 'Baseline accuracy [95% CI]', 'Candidate accuracy [95% CI]', 'Delta', 'Threshold', 'McNemar'].forEach((title) => {
    const th = el('th', null, title);
    th.scope = 'col';
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);

  const body = el('tbody');
  rows.forEach((row) => {
    const tr = el('tr');
    tr.appendChild(el('td', null, row.task || '—'));
    tr.appendChild(statCell(row.baseline));
    tr.appendChild(statCell(row.candidate));

    const deltaCell = el('td', 'num');
    if (isNum(row.delta)) {
      deltaCell.appendChild(el('span', row.delta >= 0 ? 'delta--up' : 'delta--down',
        (row.delta >= 0 ? '▲ ' : '▼ ') + fmtSigned(row.delta)));
    } else {
      deltaCell.textContent = '—';
      deltaCell.appendChild(el('span', 'stack__sub', ' task missing from one run'));
    }
    tr.appendChild(deltaCell);

    const passCell = el('td');
    if (row.passes === true) passCell.appendChild(chip('✓ passes', 'ok'));
    else if (row.passes === false) passCell.appendChild(chip('✗ below threshold', 'bad'));
    else passCell.appendChild(el('span', 'stack__sub', 'not assessable'));
    tr.appendChild(passCell);

    tr.appendChild(pairedCell(row.paired));
    body.appendChild(tr);
  });
  table.appendChild(body);
  scroller.appendChild(table);
  host.appendChild(scroller);
}

async function runCompare() {
  const baseline = $('#cmp-baseline').value;
  const candidate = $('#cmp-candidate').value;
  if (!baseline || !candidate) { toast('Pick two runs', 'A baseline and a candidate are both required.'); return; }

  const params = new URLSearchParams({ baseline: baseline, candidate: candidate });
  const threshold = parseFloat($('#cmp-threshold').value);
  if (Number.isFinite(threshold)) params.set('min_improvement', String(threshold));

  const button = $('#cmp-btn');
  const status = $('#cmp-status');
  const host = $('#compare-host');
  button.disabled = true;
  status.textContent = 'comparing…';
  showSkeleton(host, false);

  try {
    const res = await fetch('/api/compare?' + params.toString(), { headers: { Accept: 'application/json' } });
    if (!res.ok) { await toastHttp(res, 'compare'); clear(host); status.textContent = 'failed'; return; }
    renderCompare(await res.json());
    status.textContent = '';
  } catch (err) {
    toastNetwork(err, 'compare');
    clear(host);
    status.textContent = 'failed';
  } finally {
    button.disabled = false;
    setBusy(host, false);
  }
}

function initTabs() {
  const tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));

  function select(index) {
    tabs.forEach((tab, i) => {
      const on = i === index;
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      tab.tabIndex = on ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      if (panel) panel.hidden = !on;
    });
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(i));
    tab.addEventListener('keydown', (event) => {
      let next = null;
      if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      if (next !== null) { event.preventDefault(); select(next); tabs[next].focus(); }
    });
  });
  select(0);
}

function init() {
  initTabs();
  buildExamples();
  $('#solve-btn').addEventListener('click', runSolve);
  $('#verify-btn').addEventListener('click', runVerify);
  $('#cmp-btn').addEventListener('click', runCompare);
  $('#runs-refresh').addEventListener('click', loadRuns);
  loadHealth();
  loadRuns();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
