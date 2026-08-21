use ndata::dataobject::DataObject;
use ndata::dataarray::DataArray;
use flowlang::datastore::DataStore;
pub fn execute(o: DataObject) -> DataObject {
    use std::panic;
    for p in ["limit"] {
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
        let arg_0: i64 = o.get_int("limit");
        transcripts(arg_0)
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

pub fn transcripts(limit: i64) -> DataObject {
// Newest transcripts from the hollis_transcripts store - the cortex
// writes one record per utterance, keyed by microsecond timestamp.
// Walks the sharded store dirs, so cost scales with total records:
// fine for a UI pane, not a hot path. limit <= 0 means 10.
let store = DataStore::new();
let mut keys: Vec<u64> = Vec::new();
let base = store.root.join("hollis_transcripts");
if base.exists() {
  fn walk(dir: &std::path::Path, keys: &mut Vec<u64>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
      for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() { walk(&path, keys); }
        else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
          if let Ok(k) = name.parse::<u64>() { keys.push(k); }
        }
      }
    }
  }
  walk(&base, &mut keys);
}
keys.sort_unstable_by(|a, b| b.cmp(a));
let n = if limit > 0 { limit as usize } else { 10 };
let mut list = DataArray::new();
for k in keys.into_iter().take(n) {
  let rec = store.get_data("hollis_transcripts", &k.to_string());
  if rec.has("data") { list.push_object(rec.get_object("data")); }
}
let mut o = DataObject::new();
o.put_string("status", "ok");
o.put_array("transcripts", list);
o
}
