// Background service worker (Manifest V3).
//
// Why this file exists: GitHub pages have a strict Content-Security-Policy that
// BLOCKS the content script from fetching our local server. The service worker
// is NOT subject to the page's CSP, so we do all network calls here and send the
// results back to the content script via messages.
//
// We use 127.0.0.1 instead of "localhost" on purpose: Chrome may resolve
// "localhost" to the IPv6 address ::1, but uvicorn listens on IPv4 127.0.0.1 by
// default - that mismatch makes requests hang instead of failing fast.

const BACKEND = "http://127.0.0.1:8000";
const TIMEOUT_MS = 10000; // give up after 10 s instead of hanging

// fetch() with a timeout, so a stuck request fails quickly with a clear message
async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function describe(err) {
  return err && err.name === "AbortError"
    ? `timed out after ${TIMEOUT_MS / 1000}s`
    : `${(err && err.message) || err}`;
}

async function doScan(rawUrl, filename) {
  // 1. fetch the raw source text of the file from GitHub
  let code;
  try {
    console.log("[VulnScanner] fetching raw file:", rawUrl);
    const rawResp = await fetchWithTimeout(rawUrl);
    if (!rawResp.ok) {
      return { error: `Could not fetch raw file from GitHub (HTTP ${rawResp.status}).` };
    }
    code = await rawResp.text();
  } catch (e) {
    return { error: `Could not fetch raw file from GitHub: ${describe(e)}` };
  }

  // 2. send it to our local backend for scanning
  try {
    console.log("[VulnScanner] sending", code.length, "chars to", BACKEND);
    const scanResp = await fetchWithTimeout(`${BACKEND}/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, code }),
    });
    if (!scanResp.ok) {
      return { error: `Backend returned HTTP ${scanResp.status}.` };
    }
    return { data: await scanResp.json() };
  } catch (e) {
    return {
      error:
        `Backend not reachable at ${BACKEND} (${describe(e)}). ` +
        "Is it running? From the project root: python -m uvicorn backend.app:app --port 8000",
    };
  }
}

async function doFeedback(payload) {
  await fetchWithTimeout(`${BACKEND}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "scan") {
    doScan(msg.rawUrl, msg.filename).then(sendResponse);
    return true; // keep the message channel open for the async reply
  }

  if (msg.type === "feedback") {
    doFeedback(msg.payload)
      .then(() => sendResponse({ ok: true }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
});
