// Popup: show the last scan summary saved by the content script.
// Clicking a line asks the page (content script) to jump to it.
// All text is inserted with textContent, never innerHTML.

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function pct(p) {
  return Math.round(p * 100) + "%";
}

function bandClass(band) {
  return band === "High" ? "red" : band === "Medium" ? "yellow" : "green";
}

// ask the content script in the current tab to scroll to a line
function goTo(line) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) chrome.tabs.sendMessage(tabs[0].id, { type: "goto", line });
    window.close();
  });
}

chrome.storage.local.get("vs_summary", (store) => {
  const s = store.vs_summary;
  const root = document.getElementById("content");
  if (!s) return;

  root.className = "";
  root.replaceChildren();
  root.append(el("div", "muted", s.filename || ""));

  if (s.error) {
    root.append(el("div", "err", s.error));
    return;
  }
  if (!s.total) {
    root.append(el("div", "muted", "No functions in this file — nothing to score."));
    return;
  }

  const chips = el("div", "chips");
  chips.append(el("span", "chip red", `${s.high} High`));
  chips.append(el("span", "chip yellow", `${s.medium} Medium`));
  chips.append(el("span", "chip green", `${s.total - s.high - s.medium} Low`));
  root.append(chips);

  const fns = s.top_functions || [];
  if (fns.length) {
    const top = fns[0];
    const hero = el("div", "hero");
    hero.append(el("div", "label", "Most likely vulnerable line"));
    hero.append(el("div", "big", `Line ${top.line}`));
    hero.append(el("div", "muted", `in ${top.name}() · ${pct(top.risk)} risk`));
    if (top.reason) hero.append(el("div", null, top.reason));
    hero.onclick = () => goTo(top.line);
    root.append(hero);

    root.append(el("div", "muted", "Riskiest functions"));
    for (const f of fns) {
      const row = el("div", "row");
      row.append(el("span", "pct " + bandClass(f.band), pct(f.risk)));
      row.append(el("span", "name", f.name + "()"));
      row.append(el("span", "ln", "L" + f.line));
      row.title = f.reason || "";
      row.onclick = () => goTo(f.line);
      root.append(row);
    }
  }

  root.append(
    el("div", "foot muted",
       `${s.total} functions · ${s.scan_time_ms ?? "-"} ms${s.cached ? " (cached)" : ""}`)
  );
});
