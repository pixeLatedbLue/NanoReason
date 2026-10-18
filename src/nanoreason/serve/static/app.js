"use strict";

const $ = (sel, root) => (root || document).querySelector(sel);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

const isNum = (v) => typeof v === "number" && isFinite(v);

function fmt(v, digits) {
  return isNum(v) ? v.toFixed(digits === undefined ? 3 : digits) : "·";
}

function fmtSigned(v, digits) {
  if (!isNum(v)) return "·";
  const d = digits === undefined ? 3 : digits;
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
}

function fmtPValue(p) {
  if (!isNum(p)) return "·";
  return p < 0.0001 ? "p < 0.0001" : "p = " + p.toFixed(4);
}

function pct(value, digits = 1) {
  return isNum(value) ? (value * 100).toFixed(digits) + "%" : "n/a";
}

function fmtDate(unix) {
  if (!isNum(unix)) return "·";
  try {
    return new Date(unix * 1000).toLocaleString();
  } catch (e) {
    return "·";
  }
}

function chip(text, kind) {
  return el("span", kind ? "chip chip--" + kind : "chip", text);
}

function lockInputs(selectors, disabled) {
  document.querySelectorAll(selectors).forEach((input) => {
    input.disabled = disabled;
  });
}

function invalidateSolve() {
  if ($("#solve-output").textContent.trim())
    $("#solve-status").textContent = "Previous response · Solve to update";
  clear($("#solve-analysis"));
}

function setBusy(host, busy) {
  if (host) host.setAttribute("aria-busy", busy ? "true" : "false");
}

function showSkeleton(host, tall) {
  clear(host);
  setBusy(host, true);
  const card = el("div", "card");
  card.appendChild(el("div", "skel skel--line"));
  card.appendChild(el("div", "skel skel--line"));
  if (tall) card.appendChild(el("div", "skel skel--tall"));
  host.appendChild(card);
}

function toast(title, message) {
  const host = $("#toasts");
  const box = el("div", "toast");
  const body = el("div", "toast__body");
  body.appendChild(el("div", "toast__title", title));
  if (message) body.appendChild(el("div", "toast__msg", message));
  const close = el("button", "toast__x", "×");
  close.type = "button";
  close.setAttribute("aria-label", "Dismiss notification");
  close.addEventListener("click", () => box.remove());
  box.appendChild(body);
  box.appendChild(close);
  host.appendChild(box);
  setTimeout(() => box.remove(), 10000);
}

async function toastHttp(res, what) {
  let detail = "";
  try {
    const body = await res.clone().json();
    if (body && typeof body === "object")
      detail = String(body.detail || body.message || "");
  } catch (e) {}
  if (!detail) {
    try {
      detail = (await res.text()).slice(0, 300);
    } catch (e) {}
  }
  toast(
    "HTTP " + res.status + (res.statusText ? " " + res.statusText : ""),
    (what ? what + " · " : "") + (detail || "The server returned no detail."),
  );
}

function toastNetwork(err, what) {
  toast(
    "Request failed",
    (what ? what + " · " : "") + String((err && err.message) || err),
  );
}

async function loadHealth() {
  const badge = $("#engine-badge");
  try {
    const res = await fetch("/api/health", {
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const health = await res.json();
    const e = health.engine || {};
    const shortModel = e.model ? e.model.split("/").pop() : "Unnamed model";
    $("#model-name").textContent = e.is_demo ? "Scripted demo" : shortModel;
    const adapterName = e.adapter
      ? e.adapter.split(/[\\/]/).pop()
      : "Base weights";
    $("#model-adapter").textContent =
      adapterName +
      (e.adapter
        ? e.adapter_active
          ? " · Adapter active"
          : " · Not loaded"
        : "");
    const ready = !e.is_demo && e.device !== "unloaded";
    badge.className = "badge " + (ready ? "badge--ok" : "badge--warn");
    badge.textContent = e.is_demo
      ? "Demo · scripted output"
      : ready
        ? "Live · " + (e.device || "device unknown")
        : "Loads on first answer";
    $("#mobile-engine").textContent = e.is_demo
      ? "Scripted demo"
      : ready
        ? "Live model · " + e.device
        : "Model not loaded";
    $("#engine-meta").textContent = [
      "Model: " + (e.model || "scripted demo"),
      "Engine: " + (e.kind || "unknown"),
      "Device: " + (e.device || "unknown"),
      "Adapter: " + (e.adapter || "none"),
      e.adapter ? "Adapter active: " + Boolean(e.adapter_active) : "",
    ]
      .filter(Boolean)
      .join(" · ");
    $("#version").textContent = health.version ? "v" + health.version : "";
  } catch (err) {
    badge.className = "badge badge--warn";
    badge.textContent = "Engine unavailable";
    $("#model-name").textContent = "Connection unavailable";
    $("#model-adapter").textContent = "Check the local server";
    $("#mobile-engine").textContent = "Engine unavailable";
    $("#engine-meta").textContent = "Health check failed: " + err.message;
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
  return {
    zero: zero,
    left: Math.min(zero, point),
    width: Math.abs(point - zero),
    negative: clamped < 0,
  };
}

function rewardRow(label, value, isTotal) {
  const row = el("div", isTotal ? "rw rw--total" : "rw");
  row.appendChild(el("div", "rw__label", label));
  const track = el("div", "rw__track");
  if (isNum(value)) {
    const geo = rewardGeometry(value);
    const zeroTick = el("span", "rw__zero");
    zeroTick.style.left = geo.zero + "%";
    const fill = el(
      "span",
      geo.negative ? "rw__fill rw__fill--neg" : "rw__fill",
    );
    fill.style.left = geo.left + "%";
    fill.style.width = geo.width + "%";
    track.appendChild(zeroTick);
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("div", "rw__val", fmtSigned(value)));
  } else {
    track.appendChild(el("span", "rw__na", "n/a · no gold answer"));
    row.appendChild(track);
    row.appendChild(el("div", "rw__val rw__val--na", "n/a"));
  }
  return row;
}

function answerSection(analysis) {
  const sub = el("section", "sub");
  const line = el("div", "ans");
  line.appendChild(el("span", "ans__label", "Final answer"));

  const hasAnswer =
    analysis.answer !== null &&
    analysis.answer !== undefined &&
    analysis.answer !== "";
  line.appendChild(
    el(
      "span",
      hasAnswer ? "ans__value" : "ans__value ans__value--none",
      hasAnswer ? analysis.answer : "no answer parsed",
    ),
  );

  const chips = el("div", "ans__chips");
  if (analysis.answer_source === "marker") {
    chips.appendChild(chip("#### marker", "accent"));
  } else if (analysis.answer_source === "fallback") {
    chips.appendChild(chip("fallback: last number", "warn"));
  }
  chips.appendChild(
    analysis.format_ok
      ? chip("✓ format ok", "ok")
      : chip("✗ format not followed", "bad"),
  );

  if (analysis.correct === true) {
    chips.appendChild(chip("Matches gold answer", "ok"));
  } else if (analysis.correct === false) {
    chips.appendChild(chip("Does not match gold", "bad"));
  }
  line.appendChild(chips);
  sub.appendChild(line);

  const goldText =
    analysis.gold === null ||
    analysis.gold === undefined ||
    analysis.gold === ""
      ? "No gold answer supplied. Answer correctness is unscored; only explicit arithmetic and formatting are checked."
      : "Gold answer: " + analysis.gold;
  sub.appendChild(el("p", "hint hint--block", goldText));
  return sub;
}

function rewardSection(analysis) {
  const rewards = analysis.rewards || {};
  const sub = el("section", "sub");
  sub.appendChild(el("h3", "h3", "Reward breakdown"));
  sub.appendChild(rewardRow("outcome", rewards.outcome, false));
  sub.appendChild(rewardRow("format", rewards.format, false));
  sub.appendChild(rewardRow("process", rewards.process, false));
  sub.appendChild(rewardRow("diversity", rewards.diversity, false));
  sub.appendChild(rewardRow("total", rewards.total, true));
  sub.appendChild(
    el(
      "p",
      "hint hint--block",
      "Bars share one scale from " +
        fmtSigned(REWARD_MIN, 1) +
        " to " +
        fmtSigned(REWARD_MAX, 1) +
        "; the tick is zero, and bars to its left are penalties.",
    ),
  );
  return sub;
}

function stepSection(analysis) {
  const steps = Array.isArray(analysis.steps) ? analysis.steps : [];
  const sub = el("section", "sub");
  sub.appendChild(el("h3", "h3", "Arithmetic steps"));

  if (!steps.length) {
    sub.appendChild(
      el(
        "div",
        "empty",
        "No arithmetic steps were found, so the verifier had nothing to check. " +
          "A completion with no checkable equations earns no process reward.",
      ),
    );
    return sub;
  }

  const scroller = el("div", "scroll-x");
  const table = el("table", "tbl");
  const head = el("thead");
  const headRow = el("tr");
  ["#", "Equation", "Computed", "Verified", "Note"].forEach((title) => {
    const th = el("th", null, title);
    th.scope = "col";
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);

  const body = el("tbody");
  steps.forEach((step) => {
    const tr = el("tr", step.ok ? null : "row--bad");
    tr.appendChild(el("td", "num", isNum(step.index) ? step.index : "·"));

    const written = step.raw
      ? step.raw
      : [step.lhs, step.op, step.rhs, "=", step.stated]
          .filter((p) => p !== null && p !== undefined)
          .join(" ");
    tr.appendChild(el("td", "num", written));

    tr.appendChild(
      el("td", "num", isNum(step.computed) ? String(step.computed) : "·"),
    );

    const verdict = el("td");
    const mark = el(
      "span",
      step.ok ? "mark mark--ok" : "mark mark--bad",
      step.ok ? "✓ correct" : "✗ wrong",
    );
    verdict.appendChild(mark);
    tr.appendChild(verdict);

    const note = el("td");
    if (step.duplicate) note.appendChild(chip("duplicate", "warn"));
    else note.appendChild(el("span", "stack__sub", "·"));
    tr.appendChild(note);

    body.appendChild(tr);
  });
  table.appendChild(body);

  const caption = el(
    "caption",
    null,
    fmt(analysis.steps_correct, 0) +
      " of " +
      fmt(analysis.steps_total, 0) +
      " steps verified  ·  " +
      fmt(analysis.steps_unique, 0) +
      " unique",
  );
  table.insertBefore(caption, table.firstChild);

  scroller.appendChild(table);
  sub.appendChild(scroller);
  return sub;
}

function diversitySection(analysis) {
  const value = analysis.token_diversity;
  const sub = el("section", "sub");
  sub.appendChild(el("h3", "h3", "Word diversity"));

  const row = el("div", "meter__row");
  const meter = el("div", "meter");
  meter.setAttribute("role", "img");
  const pct = isNum(value) ? Math.max(0, Math.min(1, value)) * 100 : 0;
  const low = isNum(value) && value < DIVERSITY_THRESHOLD;
  const fill = el("span", low ? "meter__fill meter__fill--low" : "meter__fill");
  fill.style.width = pct + "%";
  meter.appendChild(fill);
  const mark = el("span", "meter__mark");
  mark.style.left = DIVERSITY_THRESHOLD * 100 + "%";
  meter.appendChild(mark);
  meter.setAttribute(
    "aria-label",
    "Word diversity " +
      fmt(value) +
      " against a low-diversity threshold of " +
      DIVERSITY_THRESHOLD,
  );
  row.appendChild(meter);
  row.appendChild(el("span", "rw__val", fmt(value)));
  sub.appendChild(row);

  const verdict = isNum(value)
    ? low
      ? "Low word diversity (below " + DIVERSITY_THRESHOLD + ")"
      : "Above the low-diversity threshold of " + DIVERSITY_THRESHOLD
    : "no diversity score reported";
  sub.appendChild(
    el(
      "p",
      "meter__cap",
      verdict + "  (the tick marks " + DIVERSITY_THRESHOLD + ").",
    ),
  );
  return sub;
}

function renderAnalysis(host, analysis) {
  clear(host);
  setBusy(host, false);
  if (!analysis || typeof analysis !== "object") {
    host.appendChild(
      el("div", "empty", "The server returned no analysis for this request."),
    );
    return;
  }
  const card = el("div", "card analysis-card");
  const summary = answerSection(analysis);
  summary.classList.add("analysis-summary");
  card.appendChild(summary);
  const grid = el("div", "analysis-grid");
  grid.appendChild(stepSection(analysis));
  const detail = el("div");
  detail.appendChild(rewardSection(analysis));
  const diversity = el("details", "diversity-details");
  diversity.appendChild(el("summary", null, "Inspect word diversity"));
  diversity.appendChild(diversitySection(analysis));
  detail.appendChild(diversity);
  grid.appendChild(detail);
  card.appendChild(grid);
  host.appendChild(card);
  reveal(card);
}

const EXAMPLES = [
  {
    label: "Two-step counting",
    gold: "72",
    text: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  },
  {
    label: "Materials",
    gold: "3",
    text: "A robe takes 2 bolts of blue fibre and half that much white fibre. How many bolts does it take in total?",
  },
  {
    label: "Hourly pay",
    gold: "10",
    text: "Weng earns 12 dollars an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?",
  },
];

function buildExamples() {
  const row = $("#solve-examples");
  EXAMPLES.forEach((example) => {
    const button = el("button", "ex", example.label);
    button.type = "button";
    button.title = example.text;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      row
        .querySelectorAll("button")
        .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      invalidateSolve();
      $("#solve-input").value = example.text;
      $("#solve-gold").value = example.gold;
      $("#solve-input").focus();
    });
    row.appendChild(button);
  });
  $("#solve-input").addEventListener("input", () => {
    invalidateSolve();
    row
      .querySelectorAll("button")
      .forEach((b) => b.setAttribute("aria-pressed", "false"));
  });
}

function parseFrame(frame) {
  const payload = frame
    .split(/\r?\n/)
    .filter((line) => line.indexOf("data:") === 0)
    .map((line) => line.slice(5).trim())
    .join("");
  if (!payload) return null;
  try {
    return JSON.parse(payload);
  } catch (e) {
    return null;
  }
}

async function runSolve() {
  if ($("#solve-btn").disabled) return;
  if (!$("#solve-tokens").reportValidity()) return;
  const question = $("#solve-input").value.trim();
  if (!question) {
    toast("Nothing to solve", "Enter a word problem first.");
    return;
  }

  const params = new URLSearchParams({ question: question });
  const maxTokens = parseInt($("#solve-tokens").value, 10);
  if (Number.isFinite(maxTokens) && maxTokens > 0)
    params.set("max_new_tokens", String(maxTokens));
  const gold = $("#solve-gold").value.trim();
  if (gold) params.set("gold", gold);

  const button = $("#solve-btn");
  const status = $("#solve-status");
  const output = $("#solve-output");
  const host = $("#solve-analysis");
  button.disabled = true;
  status.textContent = "Generating response…";
  $("#response-empty").hidden = true;
  output.hidden = false;
  $("#copy-response").disabled = true;
  $(".response-card").dataset.streaming = "true";
  button.firstChild.textContent = "Generating…";
  lockInputs(
    "#solve-input, #solve-gold, #solve-tokens, #solve-examples button",
    true,
  );
  output.textContent = "";
  clear(host);
  setBusy(host, true);

  let sawTerminal = false;
  try {
    const res = await fetch("/api/solve/stream?" + params.toString(), {
      headers: { Accept: "text/event-stream" },
    });
    if (!res.ok) {
      await toastHttp(res, "streaming solve");
      status.textContent = "failed";
      return;
    }
    if (!res.body) {
      toast("Streaming unsupported", "The response carried no readable body.");
      status.textContent = "failed";
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
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
        if (event.type === "token") {
          output.textContent += event.text || "";
          output.scrollTop = output.scrollHeight;
        } else if (event.type === "done") {
          sawTerminal = true;
          if (typeof event.completion === "string")
            output.textContent = event.completion;
          renderAnalysis(host, event.analysis);
          status.textContent = event.failed
            ? "failed · partial output"
            : isNum(event.latency_ms)
              ? "Completed in " + (event.latency_ms / 1000).toFixed(1) + " s"
              : "completed";
        } else if (event.type === "error") {
          sawTerminal = true;
          toast(
            "Stream error",
            String(event.message || "The server reported an error frame."),
          );
          status.textContent = "failed";
        }
      }
    }
    if (!sawTerminal) {
      toast(
        "Stream ended early",
        "The connection closed before a final frame arrived.",
      );
      status.textContent = "interrupted";
    }
  } catch (err) {
    toastNetwork(err, "streaming solve");
    status.textContent = "failed";
  } finally {
    button.disabled = false;
    setBusy(host, false);
    $(".response-card").dataset.streaming = "false";
    button.firstChild.textContent = "Solve problem";
    lockInputs(
      "#solve-input, #solve-gold, #solve-tokens, #solve-examples button",
      false,
    );
    $("#copy-response").disabled = !output.textContent.trim();
    if (!output.textContent.trim()) {
      output.hidden = true;
      $("#response-empty").hidden = false;
    }
    if (!host.firstChild)
      host.appendChild(
        el(
          "div",
          "waiting-state",
          "No completed analysis. You can retry the problem.",
        ),
      );
    loadHealth();
  }
}

async function runVerify() {
  if ($("#verify-btn").disabled) return;
  const text = $("#verify-input").value;
  if (!text.trim()) {
    toast("Nothing to verify", "Paste some reasoning first.");
    return;
  }
  const gold = $("#verify-gold").value.trim();

  const button = $("#verify-btn");
  const status = $("#verify-status");
  const host = $("#verify-analysis");
  button.disabled = true;
  status.textContent = "verifying…";
  lockInputs("#verify-input, #verify-gold, #verify-sample", true);
  showSkeleton(host, true);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ text: text, gold: gold || null }),
    });
    if (!res.ok) {
      await toastHttp(res, "analyze");
      clear(host);
      status.textContent = "failed";
      return;
    }
    renderAnalysis(host, await res.json());
    status.textContent = "Checked on CPU · No model used";
  } catch (err) {
    toastNetwork(err, "analyze");
    clear(host);
    status.textContent = "failed";
  } finally {
    button.disabled = false;
    lockInputs("#verify-input, #verify-gold, #verify-sample", false);
    if (!host.firstChild)
      host.appendChild(
        el(
          "div",
          "waiting-state",
          "Verification unavailable. Retry when the local server is ready.",
        ),
      );
    setBusy(host, false);
  }
}

let RUNS = [];

function taskChips(tasks) {
  const box = el("div", "tasks");
  const names = tasks && typeof tasks === "object" ? Object.keys(tasks) : [];
  if (!names.length) {
    box.appendChild(el("span", "stack__sub", "no tasks recorded"));
    return box;
  }
  names.forEach((name) => {
    const stat = tasks[name] || {};
    box.appendChild(
      chip(
        name.toUpperCase() +
          "  " +
          pct(stat.accuracy) +
          "  (" +
          fmt(stat.correct, 0) +
          "/" +
          fmt(stat.total, 0) +
          ")",
      ),
    );
  });
  return box;
}

function runLabel(run) {
  const name = typeof run === "string" ? run : run.name;
  return (
    { base: "Base model", sft: "SFT adapter", grpo: "GRPO adapter" }[name] ||
    name ||
    "Unnamed run"
  );
}

function renderRunsTable() {
  const host = $("#runs-host");
  clear(host);
  setBusy(host, false);
  $("#run-count").textContent = String(RUNS.length);
  if (!RUNS.length) {
    host.appendChild(
      el(
        "div",
        "empty",
        "No evaluation runs found. Saved evaluation artifacts will appear here when they are available.",
      ),
    );
    return;
  }
  const ordered = [...RUNS].sort((a, b) => {
    const order = { base: 0, sft: 1, grpo: 2 };
    return (order[a.name] ?? 3) - (order[b.name] ?? 3);
  });
  const tasks = [
    ...new Set(ordered.flatMap((r) => Object.keys(r.tasks || {}))),
  ];
  tasks.forEach((task) => {
    const figure = el("figure", "run-chart");
    const caption = el(
      "figcaption",
      "h3",
      task.toUpperCase() + " · Answer accuracy",
    );
    figure.appendChild(caption);
    const axis = el("div", "chart-axis");
    axis.setAttribute("aria-hidden", "true");
    ["0%", "25%", "50%", "75%", "100%"].forEach((t) =>
      axis.appendChild(el("span", null, t)),
    );
    figure.appendChild(axis);
    ordered.forEach((run) => {
      const stat = (run.tasks || {})[task];
      if (!stat) return;
      const known = ["base", "sft", "grpo"].includes(run.name)
        ? run.name
        : "other";
      const row = el("div", "chart-row chart-row--" + known);
      const name = el("div", "chart-name", runLabel(run));
      name.appendChild(
        el(
          "small",
          null,
          fmt(stat.correct, 0) + " / " + fmt(stat.total, 0) + " correct",
        ),
      );
      row.appendChild(name);
      const track = el("div", "chart-line");
      track.setAttribute("aria-hidden", "true");
      if (isNum(stat.accuracy)) {
        const bar = el("i");
        bar.style.width = Math.max(0, Math.min(100, stat.accuracy * 100)) + "%";
        track.appendChild(bar);
      }
      row.appendChild(track);
      row.appendChild(el("div", "chart-value", pct(stat.accuracy)));
      figure.appendChild(row);
    });
    const legend = el("div", "chart-legend");
    legend.appendChild(
      el("span", null, "Within-run accuracy · Higher is better"),
    );
    legend.appendChild(
      el("span", null, "Use the paired comparison to assess differences"),
    );
    figure.appendChild(legend);
    host.appendChild(figure);
  });
  const records = el("details", "run-records");
  records.appendChild(el("summary", null, "Run details & source artifacts"));
  ordered.forEach((run) => {
    const record = el("article", "run-record");
    const head = el("div", "run-record__head");
    head.appendChild(el("strong", null, run.run_name || runLabel(run)));
    const link = el("a", null, "Download JSON ↗");
    link.href = "/api/runs/" + encodeURIComponent(run.name);
    link.download = run.name + ".json";
    head.appendChild(link);
    record.appendChild(head);
    const dl = el("dl");
    [
      ["Base model", run.base_model || "Not recorded"],
      ["Adapter", run.adapter || "Base weights"],
      ["Config", run.config_path || "Not recorded"],
      ["Recorded", fmtDate(run.created_at_unix)],
    ].forEach(([key, value]) => {
      dl.appendChild(el("dt", null, key));
      dl.appendChild(el("dd", "num", value));
    });
    record.appendChild(dl);
    records.appendChild(record);
  });
  host.appendChild(records);
}

function fillSelects() {
  [$("#cmp-baseline"), $("#cmp-candidate")].forEach((select, index) => {
    const previous = select.value;
    clear(select);
    if (!RUNS.length) {
      const option = el("option", null, "No runs available");
      option.value = "";
      select.appendChild(option);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    RUNS.forEach((run) => {
      const option = el("option", null, runLabel(run));
      option.value = run.name;
      select.appendChild(option);
    });
    const preferred = index === 0 ? "base" : "grpo";
    if (RUNS.some((r) => r.name === previous)) select.value = previous;
    else if (RUNS.some((r) => r.name === preferred)) select.value = preferred;
    else select.selectedIndex = Math.min(index, RUNS.length - 1);
  });
  $("#cmp-btn").disabled = RUNS.length < 2;
}

async function loadRuns() {
  const host = $("#runs-host");
  if ($("#runs-refresh").disabled) return;
  $("#runs-refresh").disabled = true;
  $("#cmp-btn").disabled = true;
  showSkeleton(host, false);
  clear($("#compare-host"));
  $("#cmp-status").textContent = "";
  try {
    const res = await fetch("/api/runs", {
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    RUNS = data && Array.isArray(data.runs) ? data.runs : [];
    renderRunsTable();
    fillSelects();
  } catch (err) {
    RUNS = [];
    clear(host);
    host.appendChild(
      el(
        "div",
        "empty",
        "Could not load evaluation runs. " +
          err.message +
          ". Use Refresh to try again.",
      ),
    );
    $("#run-count").textContent = "";
    fillSelects();
  } finally {
    $("#runs-refresh").disabled = false;
    setBusy(host, false);
  }
}

function statCell(stat) {
  const td = el("td", "num");
  if (!stat || typeof stat !== "object") {
    td.textContent = "·";
    return td;
  }
  const stack = el("div", "stack");
  const line = el("span");
  line.appendChild(document.createTextNode(pct(stat.accuracy) + " "));
  line.appendChild(
    el("span", "ci", pct(stat.ci_low) + " to " + pct(stat.ci_high)),
  );
  stack.appendChild(line);
  stack.appendChild(
    el(
      "span",
      "stack__sub",
      fmt(stat.correct, 0) + " / " + fmt(stat.total, 0) + " correct",
    ),
  );
  td.appendChild(stack);
  return td;
}

function pairedCell(paired) {
  const td = el("td");
  if (!paired || typeof paired !== "object") {
    td.appendChild(el("span", "stack__sub", "no paired items · unpaired"));
    return td;
  }
  const stack = el("div", "stack");
  stack.appendChild(
    paired.significant
      ? chip("Significant difference", "accent")
      : chip("Not significant"),
  );
  stack.appendChild(el("span", "stack__sub", fmtPValue(paired.p_value)));
  stack.appendChild(
    el(
      "span",
      "stack__sub",
      fmt(paired.n_both, 0) +
        " paired · " +
        "+" +
        fmt(paired.only_candidate, 0) +
        " candidate-only · " +
        "+" +
        fmt(paired.only_baseline, 0) +
        " baseline-only",
    ),
  );
  td.appendChild(stack);
  return td;
}

function renderCompare(data) {
  const host = $("#compare-host");
  clear(host);
  setBusy(host, false);
  const rows = data && Array.isArray(data.rows) ? data.rows : [];

  const heading = el(
    "p",
    "hint hint--block",
    "candidate " +
      (data.candidate || "?") +
      "  vs  baseline " +
      (data.baseline || "?") +
      "  · Minimum gain: " +
      (isNum(data.min_improvement)
        ? (data.min_improvement * 100).toFixed(1) + " pp"
        : "n/a"),
  );
  host.appendChild(heading);

  if (!rows.length) {
    host.appendChild(
      el(
        "div",
        "empty",
        "These two runs share no task in common, so there is nothing to compare.",
      ),
    );
    return;
  }

  const scroller = el("div", "scroll-x");
  const table = el("table", "tbl");
  const head = el("thead");
  const headRow = el("tr");
  [
    "Task",
    "Baseline accuracy [95% CI]",
    "Candidate accuracy [95% CI]",
    "Delta",
    "Threshold",
    "McNemar",
  ].forEach((title) => {
    const th = el("th", null, title);
    th.scope = "col";
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);

  const body = el("tbody");
  rows.forEach((row) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, row.task || "·"));
    tr.appendChild(statCell(row.baseline));
    tr.appendChild(statCell(row.candidate));

    const deltaCell = el("td", "num");
    if (isNum(row.delta)) {
      deltaCell.appendChild(
        el(
          "span",
          row.delta >= 0 ? "delta--up" : "delta--down",
          fmtSigned(row.delta * 100, 1) + " pp",
        ),
      );
    } else {
      deltaCell.textContent = "·";
      deltaCell.appendChild(
        el("span", "stack__sub", " task missing from one run"),
      );
    }
    tr.appendChild(deltaCell);

    const passCell = el("td");
    if (row.passes === true) passCell.appendChild(chip("✓ passes", "ok"));
    else if (row.passes === false)
      passCell.appendChild(chip("Below threshold"));
    else passCell.appendChild(el("span", "stack__sub", "not assessable"));
    tr.appendChild(passCell);

    tr.appendChild(pairedCell(row.paired));
    body.appendChild(tr);
  });
  table.appendChild(body);
  scroller.appendChild(table);
  scroller.tabIndex = 0;
  scroller.setAttribute(
    "aria-label",
    "Comparison table; scroll horizontally for all columns",
  );
  host.appendChild(
    el("p", "scroll-hint", "Swipe the table to see the full comparison."),
  );
  host.appendChild(scroller);
  rows.forEach((row) => {
    if (!row.paired) return;
    const insight = el("div", "comparison-insight");
    const significant = row.paired.significant;
    const text = significant
      ? row.delta > 0
        ? "A paired improvement was detected on these evaluated items."
        : "A paired decline was detected on these evaluated items."
      : "An accuracy difference has not been established.";
    insight.appendChild(
      el("strong", null, row.task.toUpperCase() + ": " + text),
    );
    insight.appendChild(
      el(
        "span",
        null,
        fmt(row.paired.n_both, 0) +
          " paired items · " +
          fmtPValue(row.paired.p_value) +
          ". This comparison does not establish general performance on other questions.",
      ),
    );
    host.appendChild(insight);
  });
}

async function runCompare() {
  if ($("#cmp-btn").disabled) return;
  if (!$("#cmp-threshold").reportValidity()) return;
  if ($("#cmp-baseline").value === $("#cmp-candidate").value) {
    toast(
      "Choose different runs",
      "The baseline and candidate must be different checkpoints.",
    );
    return;
  }
  const baseline = $("#cmp-baseline").value;
  const candidate = $("#cmp-candidate").value;
  if (!baseline || !candidate) {
    toast("Pick two runs", "A baseline and a candidate are both required.");
    return;
  }

  const params = new URLSearchParams({
    baseline: baseline,
    candidate: candidate,
  });
  const threshold = parseFloat($("#cmp-threshold").value);
  if (Number.isFinite(threshold))
    params.set("min_improvement", String(threshold));

  const button = $("#cmp-btn");
  const status = $("#cmp-status");
  const host = $("#compare-host");
  button.disabled = true;
  status.textContent = "comparing…";
  lockInputs(
    "#cmp-baseline, #cmp-candidate, #cmp-threshold, #runs-refresh",
    true,
  );
  showSkeleton(host, false);

  try {
    const res = await fetch("/api/compare?" + params.toString(), {
      headers: { Accept: "application/json" },
    });
    if (!res.ok) {
      await toastHttp(res, "compare");
      clear(host);
      status.textContent = "failed";
      return;
    }
    renderCompare(await res.json());
    status.textContent = "";
  } catch (err) {
    toastNetwork(err, "compare");
    clear(host);
    status.textContent = "failed";
  } finally {
    button.disabled = false;
    lockInputs(
      "#cmp-baseline, #cmp-candidate, #cmp-threshold, #runs-refresh",
      false,
    );
    if (!host.firstChild)
      host.appendChild(
        el(
          "div",
          "waiting-state",
          "Comparison unavailable. Retry to load the saved results.",
        ),
      );
    setBusy(host, false);
  }
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
function reveal(node) {
  if (window.gsap && !reducedMotion.matches) {
    gsap.killTweensOf(node);
    gsap.fromTo(
      node,
      { opacity: 0, y: 8 },
      {
        opacity: 1,
        y: 0,
        duration: 0.28,
        ease: "power2.out",
        clearProps: "opacity,transform",
      },
    );
  }
}

function initTabs() {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const narrow = window.matchMedia("(max-width: 800px)");
  const orientation = () =>
    $(".tabs").setAttribute(
      "aria-orientation",
      narrow.matches ? "horizontal" : "vertical",
    );
  orientation();
  narrow.addEventListener("change", orientation);
  function select(index) {
    tabs.forEach((tab, i) => {
      const on = i === index;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      panel.hidden = !on;
      if (on) {
        const label = tab.id.replace("tab-", "");
        $("#current-section").textContent =
          label[0].toUpperCase() + label.slice(1);
        document.title = "NanoReason · " + $("#current-section").textContent;
        reveal(panel);
      }
    });
  }
  function fromHash() {
    if (location.hash === "#main") return;
    const index = tabs.findIndex(
      (t) => "#" + t.id.replace("tab-", "") === location.hash,
    );
    select(index < 0 ? 0 : index);
  }
  function navigate(index) {
    const hash = "#" + tabs[index].id.replace("tab-", "");
    if (location.hash !== hash) history.pushState(null, "", hash);
    select(index);
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => navigate(i));
    tab.addEventListener("keydown", (event) => {
      let next = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown")
        next = (i + 1) % tabs.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp")
        next = (i - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      if (next !== null) {
        event.preventDefault();
        navigate(next);
        tabs[next].focus();
      }
    });
  });
  window.addEventListener("hashchange", fromHash);
  fromHash();
}

function initTheme() {
  const system = window.matchMedia("(prefers-color-scheme: dark)");
  function sync() {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : system.matches;
    $("#theme-toggle").setAttribute(
      "aria-label",
      "Switch to " + (dark ? "light" : "dark") + " theme",
    );
    $("#theme-icon").src = "vendor/" + (dark ? "sun" : "moon") + ".svg";
  }
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : system.matches;
    const theme = dark ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("nanoreason-theme", theme);
    } catch (_) {
      /* Optional persistence. */
    }
    sync();
  });
  system.addEventListener("change", sync);
  sync();
}

function init() {
  initTabs();
  initTheme();
  buildExamples();
  $("#solve-btn").addEventListener("click", runSolve);
  $("#verify-btn").addEventListener("click", runVerify);
  $("#cmp-btn").addEventListener("click", runCompare);
  $("#runs-refresh").addEventListener("click", loadRuns);
  $("#solve-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runSolve();
    }
  });
  ["solve-gold", "solve-tokens"].forEach((id) =>
    $("#" + id).addEventListener("input", invalidateSolve),
  );
  ["verify-input", "verify-gold"].forEach((id) =>
    $("#" + id).addEventListener("input", () => {
      clear($("#verify-analysis"));
      $("#verify-status").textContent = "Input changed · Verify to update";
    }),
  );
  const sample = $("#verify-input").value;
  $("#verify-sample").addEventListener("click", () => {
    $("#verify-input").value = sample;
    $("#verify-gold").value = "216";
    clear($("#verify-analysis"));
    $("#verify-analysis").appendChild(
      el(
        "div",
        "waiting-state",
        "Sample restored. Verify to generate a new report.",
      ),
    );
    $("#verify-status").textContent = "Ready to check";
  });
  ["cmp-baseline", "cmp-candidate", "cmp-threshold"].forEach((id) =>
    $("#" + id).addEventListener("change", () => {
      clear($("#compare-host"));
      $("#cmp-status").textContent =
        "Selection changed. Compare to update results.";
    }),
  );
  $("#copy-response").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("#solve-output").textContent);
      toast("Copied", "Model response copied to clipboard.");
    } catch (_) {
      toast(
        "Copy unavailable",
        "Select the response text and copy it manually.",
      );
    }
  });
  loadHealth();
  loadRuns();
}

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", init);
else init();
