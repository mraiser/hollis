use ndata::dataobject::DataObject;
use ndata::dataarray::DataArray;
use flowlang::datastore::DataStore;
use flowlang::command::Command;
use flowlang::flowlang::system::time::time;
pub fn execute(o: DataObject) -> DataObject {
    use std::panic;
    for p in ["text", "entity"] {
        if !o.has(p) {
            let mut e = DataObject::new();
            e.put_string("status", "err");
            e.put_string("msg", &format!("missing required parameter: {}", p));
            let mut result_obj = DataObject::new();
            result_obj.put_object("a", e);
            return result_obj;
        }
    }
    let ax = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let arg_0: String = o.get_string("text");
        let arg_1: String = o.get_string("entity");
        inject(arg_0, arg_1)
    }));
    match ax {
        Ok(ax) => {
            let mut result_obj = DataObject::new();
    result_obj.put_object("a", ax);
            result_obj
        }
        Err(err) => {
            let mut err_obj = DataObject::new();
            err_obj.put_string("status", "err");

            let msg = if let Some(s) = err.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = err.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic occurred".to_string()
            };

            err_obj.put_string("msg", &msg);
            // Wrapped in the same `a` envelope a successful return uses.
            // Unwrapped, callers that unpack the envelope (newbound's
            // format_result, for one) report an opaque 500 — "Not an object:
            // DString(\"err\")" — instead of this message.
            let mut result_obj = DataObject::new();
            result_obj.put_object("a", err_obj);
            result_obj
        }
    }
}

pub fn inject(text: String, entity: String) -> DataObject {
// Manual injection path for the hollis sensor (docs/perception-contract.md
// section 5): builds a transcript acoustic_event and proposes it to the
// agent executive exactly as the cortex does for a live utterance - the
// test and demo surface for boxes with no microphones attached.
// This file is also the home of emit_acoustic_perception, the single
// shared emit path (the cortex's Transcript arm calls it on a thread).
emit_acoustic_perception("transcript", &text, &entity, 0.9, "")
}

// The hollis sensor's one emit path: wrap a semantic event in a v1
// perception envelope, bind best-effort through the agent's own recall
// (binding is never a gate - contract section 2), and deliver via
// agent.executive.perceive. Call from a spawned thread when the caller
// must not block. A missing agent library is a counted skip, never an
// error - hollis runs standalone by design.
pub fn emit_acoustic_perception(event: &str, text: &str, entity: &str, hint: f64, extra: &str) -> DataObject {
  // `extra` (H4c): the contract's low-level payload fields as a JSON
  // object string - label, db, delta_db, confidence, duration_ms,
  // location - merged into the payload verbatim. A String because the
  // cortex emits from spawned threads: primitives cross, handles don't.
  let mut g = DataStore::globals();
  let mut hs = if g.has("HOLLIS_SENSOR") { g.get_object("HOLLIS_SENSOR") } else {
    let mut o = DataObject::new();
    o.put_int("emitted", 0);
    o.put_int("skips", 0);
    o.put_int("errors", 0);
    o.put_int("last_emit", 0);
    o.put_string("last_event", "");
    g.put_object("HOLLIS_SENSOR", o.clone());
    o
  };
  let mut out = DataObject::new();
  // the emit gate: system.apps.hollis.runtime emit=off (absent = on)
  let mut emit_on = true;
  if g.has("system") {
    let sys = g.get_object("system");
    if sys.has("apps") && sys.get_object("apps").has("hollis") {
      let happ = sys.get_object("apps").get_object("hollis");
      if happ.has("runtime") && happ.get_object("runtime").has("emit") {
        emit_on = happ.get_object("runtime").get_string("emit") != "off";
      }
    }
  }
  if !emit_on {
    hs.put_int("skips", hs.get_int("skips") + 1);
    out.put_string("status", "ok");
    out.put_boolean("emitted", false);
    out.put_string("reason", "emit=off");
    return out;
  }
  let now = time();
  let mut payload = DataObject::new();
  payload.put_string("event", event);
  if text != "" { payload.put_string("text", text); }
  if entity != "" { payload.put_string("entity", entity); }
  if extra != "" {
    if let Ok(x) = DataObject::try_from_string(extra) {
      for k in x.clone().keys() {
        payload.set_property(&k, x.get_property(&k));
      }
    }
  }
  let mut envl = DataObject::new();
  envl.put_int("v", 1);
  envl.put_string("kind", "acoustic_event");
  envl.put_int("time", now);
  envl.put_string("sensor", "hollis");
  envl.put_object("payload", payload);
  envl.put_float("salience_hint", hint);
  // binding (contract section 2): the claims that name this entity, via
  // the agent's recall - skipped for unnamed/technical labels, and empty
  // claims still conform
  let mut claims = DataArray::new();
  if entity != "" && !entity.starts_with("Entity #")
      && entity != "Unknown" && entity != "Speaker" && entity != "Sustained" {
    let ename = entity.to_string();
    let bound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let mut p = DataObject::new();
      p.put_string("query", &ename);
      p.put_string("domains", "");
      p.put_int("limit", 3);
      Command::lookup("agent", "archivist", "recall").execute(p)
    }));
    if let Ok(Ok(r)) = bound {
      // same wrapper envelope as the sink below: the declared return
      // value of an in-process rust command rides under "a"
      let r = if r.has("a") { r.get_object("a") } else { r };
      if r.has("claims") {
        for c in r.get_array("claims").objects() {
          let co = c.object();
          let home = if co.has("home") { co.get_string("home") } else { "".to_string() };
          let mut parts = home.splitn(2, '.');
          let mut b = DataObject::new();
          b.put_string("lib", parts.next().unwrap_or(""));
          b.put_string("ctl", parts.next().unwrap_or(""));
          b.put_string("claim", &co.get_string("claim"));
          if co.has("stale") && co.get_boolean("stale") { b.put_boolean("stale", true); }
          claims.push_object(b);
        }
      }
    }
  }
  envl.put_array("claims", claims);
  // deliver: a failed lookup means no agent executive in this process
  let sink = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    Command::lookup("agent", "executive", "perceive")
  }));
  match sink {
    Ok(sink) => {
      let mut p = DataObject::new();
      p.put_object("perception", envl);
      match sink.execute(p) {
        Ok(r) => {
          // a rust command's execute returns the generated wrapper's
          // envelope: the declared return value rides under "a"
          let r = if r.has("a") { r.get_object("a") } else { r };
          if r.has("status") && r.get_string("status") == "err" {
            hs.put_int("errors", hs.get_int("errors") + 1);
            out.put_string("status", "err");
            out.put_boolean("emitted", false);
            let msg = if r.has("msg") { r.get_string("msg") } else { "perceive rejected".to_string() };
            out.put_string("reason", &msg);
          } else {
            hs.put_int("emitted", hs.get_int("emitted") + 1);
            hs.put_int("last_emit", now);
            hs.put_string("last_event", event);
            out.put_string("status", "ok");
            out.put_boolean("emitted", true);
            if r.has("queue_depth") { out.put_int("queue_depth", r.get_int("queue_depth")); }
          }
        },
        Err(e) => {
          hs.put_int("errors", hs.get_int("errors") + 1);
          out.put_string("status", "err");
          out.put_boolean("emitted", false);
          out.put_string("reason", &format!("perceive failed: {:?}", e));
        }
      }
    },
    _ => {
      hs.put_int("skips", hs.get_int("skips") + 1);
      out.put_string("status", "ok");
      out.put_boolean("emitted", false);
      out.put_string("reason", "agent not present");
    }
  }
  out
}
