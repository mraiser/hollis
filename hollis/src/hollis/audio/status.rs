use ndata::dataobject::DataObject;
use ndata::dataarray::DataArray;
use flowlang::datastore::DataStore;
use flowlang::command::Command;
use flowlang::flowlang::system::time::time;
pub fn execute(_: DataObject) -> DataObject {
    use std::panic;
    let ax = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        status()
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

pub fn status() -> DataObject {
// Sensor-side status for the hollis plugin: the cortex loop, the emit
// gate, the emit counters, and whether an agent executive is present to
// receive. The hollis UI and verification batteries read this; it never
// blocks and never touches audio hardware.
let mut o = DataObject::new();
let mut g = DataStore::globals();
o.put_string("status", "ok");
o.put_boolean("listening", g.has("HOLLIS_CORTEX_RUNNING") && g.get_boolean("HOLLIS_CORTEX_RUNNING"));
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
o.put_boolean("emit", emit_on);
let sensor = if g.has("HOLLIS_SENSOR") { g.get_object("HOLLIS_SENSOR").deep_copy() } else {
  let mut s = DataObject::new();
  s.put_int("emitted", 0);
  s.put_int("skips", 0);
  s.put_int("errors", 0);
  s.put_int("last_emit", 0);
  s.put_string("last_event", "");
  s
};
o.put_object("sensor", sensor);
// the runtime config subset the UI shows (system.apps.hollis.runtime)
let mut conf = DataObject::new();
if g.has("system") {
  let sys = g.get_object("system");
  if sys.has("apps") && sys.get_object("apps").has("hollis") {
    let happ = sys.get_object("apps").get_object("hollis");
    if happ.has("runtime") {
      let meta = happ.get_object("runtime");
      for k in ["mic1", "mic2", "loopback", "model", "memory_file", "mic_distance", "emit", "links", "ignore"] {
        if meta.has(k) { conf.put_string(k, &meta.get_string(k)); }
      }
    }
  }
}
o.put_object("config", conf);
let agent_present = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
  let _ = Command::lookup("agent", "executive", "perceive");
})).is_ok();
o.put_boolean("agent_present", agent_present);
o
}
