// Content script - runs on GitHub pages and scans C/C++ files.
//
// What it draws:
//   - a risk-% badge at each function's first line (colour = band)
//   - each function's #1 risky line: strong red + an inline
//     "Likely vulnerable line" callout with the reason
//   - #2/#3 risky lines: a lighter tint
//   - a hover card with details and a 👎 "false positive" button
//   - a collapsible side panel: the most likely vulnerable line in the file,
//     then every function ranked by risk with clickable lines
//
// Safety: all text from the file/backend is inserted with textContent (never
// innerHTML), so code shown on the page can't inject HTML or scripts.

(function () {
  // bump when this file changes, so it's easy to check which version is running
  const SCRIPT_VERSION = "4";
  document.documentElement.dataset.vulnScannerVersion = SCRIPT_VERSION;

  const C_EXTS = ["c", "cpp", "cc", "cxx", "h", "hpp", "hh"];

  // -------------------------------------------------------------------------
  // THE ONE DOM-DEPENDENT FUNCTION.
  // GitHub changes its HTML now and then, so all "find the row for line N" logic
  // lives here. If highlighting ever stops working, this is the only function to
  // fix. It returns the element that holds the CODE text for a 1-based line.
  // -------------------------------------------------------------------------
  function getCodeCell(lineNo) {
    // (a) classic view, "?plain=1" view, and the current React view: id="LC5"
    let el = document.getElementById("LC" + lineNo);
    if (el) return el;

    // (b) fallback: a line-number element carries data-line-number
    const num = document.querySelector('[data-line-number="' + lineNo + '"]');
    if (num) {
      const row = num.closest("tr, .react-file-line, div");
      if (row) {
        return (
          row.querySelector(".react-code-text, .blob-code-inner, .react-file-line") ||
          row
        );
      }
    }
    return null;
  }

  // -------------------------------------------------------------------------
  // small helpers
  // -------------------------------------------------------------------------
  function fileExtension(path) {
    const name = path.split("/").pop() || "";
    return name.includes(".") ? name.split(".").pop().toLowerCase() : "";
  }

  function pct(p) {
    return Math.round(p * 100) + "%";
  }

  function bandClass(band) {
    return band === "High" ? "vs-red" : band === "Medium" ? "vs-yellow" : "vs-green";
  }

  // create an element with a class and (safe) text
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  function cweText(fn) {
    if (!fn.cwe) return null;
    return `${fn.cwe.id} (${fn.cwe.name}) · ${pct(fn.cwe.prob)}`;
  }

  // -------------------------------------------------------------------------
  // state for the current file
  // -------------------------------------------------------------------------
  let lastResult = null;          // last scan result (re-used to re-apply marks)
  const dismissed = new Set();    // "function:line" keys reported as false positive

  function lineKey(fn, tl) {
    return fn.name + ":" + tl.line;
  }

  // -------------------------------------------------------------------------
  // hover card (one shared element)
  // -------------------------------------------------------------------------
  let hideTimer = null;

  function card() {
    let c = document.getElementById("vs-card");
    if (!c) {
      c = el("div");
      c.id = "vs-card";
      c.addEventListener("mouseenter", () => clearTimeout(hideTimer));
      c.addEventListener("mouseleave", hideCard);
      document.body.appendChild(c);
    }
    return c;
  }

  function hideCard() {
    const c = document.getElementById("vs-card");
    if (c) c.style.display = "none";
  }

  function showCard(cell, result, fn, tl, rank) {
    clearTimeout(hideTimer);
    const c = card();
    c.replaceChildren();

    const title =
      rank === 0
        ? `Line ${tl.line} · most likely vulnerable line`
        : `Line ${tl.line} · suspicious line #${rank + 1}`;
    c.append(el("div", "vs-card-title", title));
    c.append(el("div", "vs-card-reason", tl.reason));
    c.append(
      el("div", "vs-card-meta", `in ${fn.name}() · ${fn.band} risk ${pct(fn.risk_score)}`)
    );
    const cwe = cweText(fn);
    if (cwe) c.append(el("div", "vs-card-meta", "Likely type: " + cwe));

    const btn = el("button", "vs-btn", "👎 Not a bug");
    btn.title = "Report this line as a false positive";
    btn.onclick = () => reportFalsePositive(result, fn, tl, btn);
    c.append(btn);

    const r = cell.getBoundingClientRect();
    c.style.left = window.scrollX + Math.max(8, r.left + 40) + "px";
    c.style.top = window.scrollY + r.bottom + 4 + "px";
    c.style.display = "block";
  }

  // -------------------------------------------------------------------------
  // feedback
  // -------------------------------------------------------------------------
  function reportFalsePositive(result, fn, tl, btn) {
    chrome.runtime.sendMessage({
      type: "feedback",
      payload: {
        file_hash: result.file_hash,
        function_name: fn.name,
        line: tl.line,
        is_false_positive: true,
      },
    });
    dismissed.add(lineKey(fn, tl));
    if (btn) {
      btn.textContent = "✓ Reported";
      btn.disabled = true;
    }
    unmarkLine(tl.line);
    if (lastResult) renderPanel(lastResult);
  }

  // -------------------------------------------------------------------------
  // inline marks on the code
  // -------------------------------------------------------------------------
  function addBadge(fn) {
    const cell = getCodeCell(fn.start_line);
    if (!cell || cell.querySelector(".vs-badge")) return; // missing, or already done
    const badge = el("span", "vs-badge " + bandClass(fn.band), pct(fn.risk_score));
    const cwe = cweText(fn);
    badge.title =
      `${fn.name}(): ${fn.band} risk (${pct(fn.risk_score)})` + (cwe ? `\nLikely type: ${cwe}` : "");
    cell.prepend(badge);
  }

  function markLine(result, fn, tl, rank) {
    const cell = getCodeCell(tl.line);
    if (!cell) return; // line not rendered by GitHub right now

    // Always (re)apply: GitHub's React code can reset classes / children when it
    // re-renders a line. Adding a class twice is harmless.
    cell.classList.add(rank === 0 ? "vs-line-top" : "vs-line-sus");

    // inline callout only on each function's #1 line
    if (rank === 0 && !cell.querySelector(".vs-callout")) {
      const callout = el("span", "vs-callout");
      callout.append(el("span", "vs-callout-icon", "⚠"));
      callout.append(el("span", null, "Likely vulnerable line · " + tl.reason));
      callout.addEventListener("click", (ev) => {
        ev.stopPropagation();
        showCard(cell, result, fn, tl, rank);
      });
      cell.append(callout);
    }

    // hover listeners only ONCE per element
    if (cell.dataset.vsMarked === "1") return;
    cell.dataset.vsMarked = "1";
    cell.addEventListener("mouseenter", () => showCard(cell, result, fn, tl, rank));
    cell.addEventListener("mouseleave", () => {
      hideTimer = setTimeout(hideCard, 300); // time to move into the card
    });
  }

  function unmarkLine(lineNo) {
    const cell = getCodeCell(lineNo);
    if (!cell) return;
    cell.classList.remove("vs-line-top", "vs-line-sus");
    cell.querySelectorAll(".vs-callout").forEach((e) => e.remove());
    hideCard();
  }

  // GitHub only keeps part of a long file in the page, and re-renders lines.
  // We keep the last result and re-apply marks on scroll and on a timer;
  // addBadge/markLine skip anything already in place.
  function applyMarks(result) {
    for (const fn of result.functions) {
      addBadge(fn);
      fn.top_lines.forEach((tl, rank) => {
        if (!dismissed.has(lineKey(fn, tl))) markLine(result, fn, tl, rank);
      });
    }
  }

  function flash(cell) {
    cell.classList.remove("vs-flash");
    void cell.offsetWidth; // restart the CSS animation
    cell.classList.add("vs-flash");
  }

  function goToLine(lineNo) {
    const cell = getCodeCell(lineNo);
    if (cell) {
      cell.scrollIntoView({ block: "center", behavior: "smooth" });
      flash(cell);
      return;
    }
    // line not in the page yet: let GitHub scroll to it via the #L<n> anchor
    location.hash = "L" + lineNo;
    setTimeout(() => {
      const c = getCodeCell(lineNo);
      if (c) {
        c.scrollIntoView({ block: "center" });
        flash(c);
      }
    }, 700);
  }

  // -------------------------------------------------------------------------
  // side panel
  // -------------------------------------------------------------------------
  let panelCollapsed = false;
  chrome.storage.local.get("vs_panel_collapsed", (s) => {
    panelCollapsed = !!s.vs_panel_collapsed;
    const p = document.getElementById("vs-panel");
    if (p) p.classList.toggle("vs-collapsed", panelCollapsed);
  });

  function panel() {
    let p = document.getElementById("vs-panel");
    if (p) return p;

    p = el("div");
    p.id = "vs-panel";
    p.classList.toggle("vs-collapsed", panelCollapsed);

    const head = el("div", "vs-panel-head");
    head.append(el("span", "vs-panel-title", "🛡 Vuln Scanner"));
    const sub = el("span", "vs-panel-sub");
    sub.id = "vs-panel-sub";
    head.append(sub);
    const toggle = el("button", "vs-icon-btn", "▾");
    toggle.title = "Collapse / expand";
    toggle.onclick = () => {
      panelCollapsed = !panelCollapsed;
      p.classList.toggle("vs-collapsed", panelCollapsed);
      chrome.storage.local.set({ vs_panel_collapsed: panelCollapsed });
    };
    head.append(toggle);
    p.append(head);

    const body = el("div", "vs-panel-body");
    body.id = "vs-panel-body";
    p.append(body);

    document.body.appendChild(p);
    return p;
  }

  function panelStatus(text, isError) {
    panel();
    document.getElementById("vs-panel-sub").textContent = "";
    const body = document.getElementById("vs-panel-body");
    body.replaceChildren(el("div", isError ? "vs-status vs-error" : "vs-status", text));
  }

  // one clickable "line" row (used in the panel)
  function lineRow(result, fn, tl, rank) {
    const row = el("div", "vs-line-row" + (rank === 0 ? " vs-rank1" : ""));
    row.append(el("span", "vs-rank", "#" + (rank + 1)));
    row.append(el("span", "vs-lineno", "L" + tl.line));
    row.append(el("span", "vs-reason", tl.reason));
    row.title = tl.text ? tl.text : "Jump to line " + tl.line;
    row.onclick = () => goToLine(tl.line);

    const fp = el("button", "vs-icon-btn vs-fp", "👎");
    fp.title = "Report as false positive";
    fp.onclick = (ev) => {
      ev.stopPropagation();
      reportFalsePositive(result, fn, tl, fp);
    };
    row.append(fp);
    return row;
  }

  function renderPanel(result) {
    panel();
    const fns = [...result.functions].sort((a, b) => b.risk_score - a.risk_score);
    const high = fns.filter((f) => f.band === "High").length;
    const medium = fns.filter((f) => f.band === "Medium").length;

    document.getElementById("vs-panel-sub").textContent =
      `${fns.length} functions · ${result.scan_time_ms} ms${result.cached ? " (cached)" : ""}`;

    const body = document.getElementById("vs-panel-body");
    body.replaceChildren();

    if (fns.length === 0) {
      body.append(el("div", "vs-status", "No functions in this file — nothing to score."));
      return;
    }

    // --- the single most likely vulnerable line in the whole file ---
    const topFn = fns.find((f) => f.top_lines.some((t) => !dismissed.has(lineKey(f, t))));
    if (topFn) {
      const topLine = topFn.top_lines.find((t) => !dismissed.has(lineKey(topFn, t)));
      const hero = el("div", "vs-hero");
      hero.append(el("div", "vs-hero-label", "Most likely vulnerable line"));
      const main = el("div", "vs-hero-main");
      main.append(el("span", "vs-hero-line", "Line " + topLine.line));
      main.append(el("span", "vs-hero-fn", `in ${topFn.name}()`));
      hero.append(main);
      if (topLine.text) hero.append(el("code", "vs-hero-code", topLine.text));
      hero.append(el("div", "vs-hero-reason", topLine.reason));
      const cwe = cweText(topFn);
      if (cwe) hero.append(el("div", "vs-hero-reason", "Likely type: " + cwe));
      hero.title = "Click to jump to this line";
      hero.onclick = () => goToLine(topLine.line);
      body.append(hero);
    }

    // --- band summary ---
    const summary = el("div", "vs-summary");
    summary.append(el("span", "vs-chip vs-red", `${high} High`));
    summary.append(el("span", "vs-chip vs-yellow", `${medium} Medium`));
    summary.append(el("span", "vs-chip vs-green", `${fns.length - high - medium} Low`));
    body.append(summary);

    // --- every function, riskiest first; top 3 open by default ---
    fns.forEach((fn, i) => {
      const item = el("div", "vs-fn" + (i < 3 ? " vs-open" : ""));
      const head = el("div", "vs-fn-head");
      head.append(el("span", "vs-badge " + bandClass(fn.band), pct(fn.risk_score)));
      head.append(el("span", "vs-fn-name", fn.name + "()"));
      head.append(el("span", "vs-fn-range", `L${fn.start_line}–${fn.end_line}`));
      head.onclick = () => item.classList.toggle("vs-open");
      item.append(head);

      const lines = el("div", "vs-fn-lines");
      const cwe = cweText(fn);
      if (cwe) lines.append(el("div", "vs-fn-cwe", "Likely type: " + cwe));
      fn.top_lines.forEach((tl, rank) => {
        if (!dismissed.has(lineKey(fn, tl))) lines.append(lineRow(result, fn, tl, rank));
      });
      item.append(lines);
      body.append(item);
    });

    body.append(
      el("div", "vs-foot", "Shows where the model looked — not proof of a bug.")
    );
  }

  // -------------------------------------------------------------------------
  // putting it together
  // -------------------------------------------------------------------------
  // GitHub renders the code a little after navigation; wait (up to ~5 s) until
  // line 1 exists so our marks have something to attach to.
  function waitForCode(callback, tries = 25) {
    if (getCodeCell(1) || tries <= 0) return callback();
    setTimeout(() => waitForCode(callback, tries - 1), 200);
  }

  // remove everything from a previous file (GitHub is a single-page app)
  function clearMarks() {
    lastResult = null;
    dismissed.clear();
    document.querySelectorAll(".vs-badge, .vs-callout").forEach((e) => e.remove());
    document.querySelectorAll(".vs-line-top, .vs-line-sus").forEach((e) => {
      e.classList.remove("vs-line-top", "vs-line-sus");
      delete e.dataset.vsMarked;
    });
    const p = document.getElementById("vs-panel");
    if (p) p.remove();
    hideCard();
  }

  function saveSummary(result) {
    const fns = [...result.functions].sort((a, b) => b.risk_score - a.risk_score);
    chrome.storage.local.set({
      vs_summary: {
        filename: result.filename,
        total: fns.length,
        high: fns.filter((f) => f.band === "High").length,
        medium: fns.filter((f) => f.band === "Medium").length,
        scan_time_ms: result.scan_time_ms,
        cached: result.cached,
        error: null,
        top_functions: fns.slice(0, 5).map((f) => ({
          name: f.name,
          risk: f.risk_score,
          band: f.band,
          line: f.top_lines.length ? f.top_lines[0].line : f.start_line,
          reason: f.top_lines.length ? f.top_lines[0].reason : "",
          cwe: f.cwe ? f.cwe.id : null,
        })),
      },
    });
  }

  function render(result) {
    lastResult = result;
    applyMarks(result);
    renderPanel(result);
    saveSummary(result);
  }

  function run() {
    // URL looks like /owner/repo/blob/branch/path/to/file.c
    const m = location.pathname.match(/^\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    if (!m) return;
    const [, owner, repo, rest] = m;
    if (!C_EXTS.includes(fileExtension(rest))) return; // only C/C++ files

    const rawUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${rest}`;
    const filename = rest.split("/").pop();

    console.log("[VulnScanner] scanning", filename);
    panelStatus("Scanning " + filename + "…", false);

    chrome.runtime.sendMessage({ type: "scan", rawUrl, filename }, (resp) => {
      if (chrome.runtime.lastError) {
        resp = { error: "Extension error: " + chrome.runtime.lastError.message };
      }
      if (!resp || resp.error) {
        const msg = (resp && resp.error) || "Scan failed.";
        console.warn("[VulnScanner]", msg);
        panelStatus(msg, true);
        chrome.storage.local.set({
          vs_summary: { filename, error: msg, total: 0, high: 0, medium: 0 },
        });
        return;
      }
      console.log("[VulnScanner] got", resp.data.functions.length, "functions");
      waitForCode(() => render(resp.data));
    });
  }

  // the popup can ask us to jump to a line
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg && msg.type === "goto" && typeof msg.line === "number") goToLine(msg.line);
  });

  window.addEventListener(
    "scroll",
    (() => {
      let t = null;
      return () => {
        if (!lastResult) return;
        clearTimeout(t);
        t = setTimeout(() => applyMarks(lastResult), 250); // debounce
      };
    })(),
    true // capture: also catches scrolling inside GitHub's inner containers
  );

  // GitHub is a single-page app: clicking from the repo page to a file does NOT
  // reload the page, so we watch the URL ourselves and re-run when it changes.
  let lastPath = "";
  function maybeRun() {
    if (location.pathname !== lastPath) {
      lastPath = location.pathname;
      clearMarks();
      setTimeout(run, 800); // let GitHub start rendering the new page
    }
  }
  maybeRun();
  setInterval(() => {
    maybeRun();
    // GitHub's React code view re-renders lines (scroll, hover, selection...)
    // and can wipe our marks; cheaply put back any that went missing.
    if (lastResult) applyMarks(lastResult);
  }, 1500);
})();
