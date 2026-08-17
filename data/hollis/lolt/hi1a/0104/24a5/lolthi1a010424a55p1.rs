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