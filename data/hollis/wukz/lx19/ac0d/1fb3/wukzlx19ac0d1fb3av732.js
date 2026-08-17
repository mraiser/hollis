// hollis - the acoustic sensor's face. Status polled while open; every
// button is an existing platform command; this pane owns no state.
// Data: hollis.audio.status (loop, gate, counters, config, agent
// presence), hollis.audio.transcripts (the cortex's utterance store),
// hollis.audio.cortex (toggle), hollis.audio.inject (manual envelope).
var me = this;
var ME = document.getElementById(me.UUID);

var readyP = new Promise(function (res) { me.ready = res; }).then(async () => {
  const host = ME;
  const invokeP = (l, c, m, a) => new Promise((r) => invokeCommand(l, c, m, a, r));
  // the HTTP exec envelope nests the command's return under .data; an
  // inner status (our commands carry one) wins over the transport's
  const env = (e) => (!e || e.status !== "ok") ? null :
    ((e.data && typeof e.data === "object") ? { status: e.status, ...e.data } : e);
  const esc = (s) => { const d = document.createElement("span"); d.textContent = String(s ?? ""); return d.innerHTML; };
  const kv = (n) => host.querySelector(`[data-kv="${n}"]`);
  const kvRows = (pairs) => pairs.map(([k, v, cls]) =>
    `<span class="ho-kv-row"><span class="ho-kv-k">${esc(k)}</span>` +
    `<span class="ho-kv-v ${cls || ""}">${esc(v)}</span></span>`).join("");
  const chip = (sel, txt, cls) => { const el = host.querySelector(`[data-chip="${sel}"]`);
    el.textContent = txt; el.className = "ho-chip " + cls; };
  const note = (n, txt, cls) => { const el = host.querySelector(`[data-note="${n}"]`);
    el.textContent = txt || ""; el.className = "ho-note " + (cls || ""); };

  async function refresh() {
    const [stR, trR] = await Promise.all([
      invokeP("hollis", "audio", "status", {}),
      invokeP("hollis", "audio", "transcripts", { limit: 8 }),
    ]);
    const st = env(stR), tr = env(trR);
    if (!st) { note("pipeline", "status unavailable: " + ((stR && stR.msg) || "no response"), "err"); return; }
    chip("listening", st.listening ? "listening" : "ears off", st.listening ? "ok" : "off");
    chip("emit", st.emit ? "emit on" : "emit off", st.emit ? "ok" : "warn");
    chip("agent", st.agent_present ? "agent present" : "no agent", st.agent_present ? "ok" : "warn");

    const conf = st.config || {};
    kv("pipeline").innerHTML = kvRows([
      ["cortex loop", st.listening ? "running" : "stopped", st.listening ? "ok" : ""],
      ["mic 1 / mic 2", `${conf.mic1 || "default"} / ${conf.mic2 || "—"}`],
      ["loopback", conf.loopback || "—"],
      ["stt model", conf.model || "(default moonshine-base)"],
      ["memory file", conf.memory_file || "hollis_memory.json"],
      ["mic distance", conf.mic_distance ? `${conf.mic_distance}m` : "0.6m"],
    ]);
    host.querySelector('[data-act="cortex-toggle"]').textContent =
      st.listening ? "stop the ears" : "start the ears";

    const s = st.sensor || {};
    const ago = s.last_emit ? `${Math.round((Date.now() - s.last_emit) / 1000)}s ago` : "never";
    kv("sensor").innerHTML = kvRows([
      ["emitted", s.emitted ?? 0],
      ["skips (gate off / no agent)", s.skips ?? 0],
      ["errors", s.errors ?? 0, (s.errors ?? 0) > 0 ? "err" : ""],
      ["last emit", ago],
      ["last event", s.last_event || "—"],
    ]);

    const list = host.querySelector('[data-list="transcripts"]');
    const rows = (tr && tr.transcripts) || [];
    list.innerHTML = rows.length ? rows.map((t) =>
      `<div class="ho-tr"><span class="ho-tr-meta">` +
      `${esc(new Date(Math.round((t.timestamp || 0) / 1000)).toLocaleTimeString())}` +
      ` · entity #${esc(t.entity_id ?? "?")}</span>` +
      `<span class="ho-tr-text">${esc(t.text || "")}</span></div>`).join("")
      : '<p class="ho-lbl">none yet — the cortex writes one per utterance</p>';
  }

  host.querySelector('[data-act="cortex-toggle"]').addEventListener("click", async () => {
    note("pipeline", "toggling the cortex…");
    const r = await invokeP("hollis", "audio", "cortex", {});
    note("pipeline", (r && r.status === "ok")
      ? "toggled — hardware spin-up/down takes a moment"
      : "failed: " + ((r && r.msg) || "no response"), (r && r.status === "ok") ? "" : "err");
    setTimeout(refresh, 1500);
  });
  host.querySelector('[data-act="inject"]').addEventListener("click", async () => {
    const text = host.querySelector('[data-in="text"]').value.trim();
    const entity = host.querySelector('[data-in="entity"]').value.trim() || "operator";
    if (!text) { note("inject", "say something first", "warn"); return; }
    note("inject", "injecting…");
    const r = env(await invokeP("hollis", "audio", "inject", { text: text, entity: entity }));
    if (r && r.emitted) {
      note("inject", `perceived — queue depth ${r.queue_depth ?? "?"}`);
      host.querySelector('[data-in="text"]').value = "";
    } else {
      note("inject", "not emitted: " + ((r && r.reason) || "no response"), "warn");
    }
    setTimeout(refresh, 400);
  });

  setInterval(refresh, 5000);
  refresh();
});