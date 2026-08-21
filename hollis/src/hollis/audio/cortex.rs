use crate::hollis::audio::data::{EventKind, SemanticEvent, WorldState, Entity, Point3D, Vector3D};
use crossbeam_channel::Receiver;
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use crate::hollis::audio::data::cosine_similarity;
use crate::hollis::audio::perception::PerceptionEngine;
use crate::hollis::audio::data::AcousticMap;
use flowlang::flowlang::system::time::time;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::io::BufWriter;
use std::fs::OpenOptions;
use flowlang::datastore::DataStore;
use ndata::data::Data::DString;
use crate::hollis::audio::data::ContextBriefing;
use crate::hollis::audio::data::SynchronizedArrayFrame;
use crate::hollis::audio::transcribe::TranscriberMessage;
use ndata::dataobject::DataObject;
use ndata::dataarray::DataArray;
pub fn execute(_: DataObject) -> DataObject {
    use std::panic;
    let ax = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        cortex()
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

pub fn cortex() -> DataObject {
let mut g = DataStore::globals();
let RUNPARAM = "HOLLIS_CORTEX_RUNNING";
match g.has(RUNPARAM) && g.get_boolean(RUNPARAM) {
  true => { 
    g.put_boolean(RUNPARAM, false); 
  },
  _ => {
    let meta = g.get_object("system").get_object("apps").get_object("hollis").get_object("runtime");
    let ear_left = match meta.has("mic1") { true => meta.get_string("mic1"), _ => "default".to_string() };
    let ear_right = match meta.has("mic2") { true => Some(meta.get_string("mic2")), _ => None };
    let loopback_dev = match meta.has("loopback") { true => Some(meta.get_string("loopback")), _ => None };
    let model_path = match meta.has("model") { true => meta.get_string("model"), _ => "/newbound/models/moonshine-base".to_string() };
    let memory_file = match meta.has("memory_file") { true => meta.get_string("memory_file"), _ => "hollis_memory.json".to_string() };
    let mic_distance = match meta.has("mic_distance") { true => meta.get_string("mic_distance").parse().unwrap(), _ => 0.6 };

    g.put_boolean(RUNPARAM, true);

   // 1. The Raw Audio Pipe
    let (sync_tx, sync_rx) = crossbeam_channel::bounded::<SynchronizedArrayFrame>(50);
    
    // 2. The Brain's Event Pipe
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<SemanticEvent>();

    // 3. The Transcriber's Audio Pipe
    let (transcribe_tx, transcribe_rx) = crossbeam_channel::unbounded::<TranscriberMessage>();

    println!("Initializing Hollis...");
    println!("Config: Model='{}', Memory='{}', MicDist={:.2}m", model_path, memory_file, mic_distance);

    // 1. Spawn Transcriber
    // It listens to Perception and sends results to Cortex (via event_tx)
    let transcriber = crate::hollis::audio::transcribe::Transcriber::new(model_path);
    let t_handle = transcriber.spawn(transcribe_rx, event_tx.clone());

    // 2. Spawn Cortex
    let c_handle = Cortex::spawn(event_rx, event_tx.clone(), memory_file);

    // 3. Spawn Perception
    let p_handle = PerceptionEngine::spawn(
      sync_rx,
      event_tx, 
      Some(transcribe_tx),
      mic_distance as f32,
    );

    // 4. Spawn Sensors

    // 1. Determine Channel Strategy
    // If user didn't specify a distinct second mic, or if the names match, assume Stereo Pair.
    let is_stereo_pair = match &ear_right {
      Some(right_name) => ear_left == *right_name || right_name == "default", // Heuristic
      None => false,
    };

    // 4. Spawn Sensors 
    
    // Step A: Build the 6-channel Radar Dish and patch the physical hardware into it
    crate::hollis::audio::sensor::setup_hardware_routing();

    // Step C: Spawn the SINGLE Master Sensor Array (2 Loopback, 5 Physical Mics)
    let _master_sensor = crate::hollis::audio::sensor::MasterSensorArray::new(
        "default", 
        2, 
        5, // <--- CHANGED TO 5
        sync_tx
    ).expect("Failed to initialize Master Sensor Array");
    
    println!("Hollis Cortex Active.");
    
    println!("Initiating Acoustic Calibration...");
    // Fire an asynchronous 50ms square wave "Pop". 
    // The sharp leading edge acts as an acoustic needle for alignment.
    std::thread::spawn(|| {
      std::thread::sleep(std::time::Duration::from_millis(1500)); 
      let _ = std::process::Command::new("play")
      .args(["-n", "synth", "0.05", "square", "2000"])
      .output();
    });

    thread::spawn(move || {
      let beat = std::time::Duration::from_secs(1);
      while g.get_boolean(RUNPARAM) { std::thread::sleep(beat); }

      println!("Stopping threads...");

      // 1. Drop the sensor array and loopback to close audio_tx
      // This causes Perception to finish -> Closes transcribe_tx -> Transcriber finishes
      drop(_master_sensor);

      // 2. Wait for the threads to unwind and finish
      if let Err(e) = p_handle.join() { eprintln!("Error joining Perception: {:?}", e); }
      if let Err(e) = t_handle.join() { eprintln!("Error joining Transcriber: {:?}", e); }
      if let Err(e) = c_handle.join() { eprintln!("Error joining Cortex: {:?}", e); }

      println!("Hollis Cortex Shut Down.");
    });
  }
}

DataObject::new()
}

const ENTITY_TIMEOUT_MS: u64 = 30_000; // Forget entities after 30s of silence

pub struct Cortex {
  state: WorldState,
  history: HashMap<u64, Entity>,
  next_entity_id: u64, 
  store: DataStore,
  event_tx: crossbeam_channel::Sender<SemanticEvent>,  
  // H4c coalescing state: one perception per state SHIFT, never per
  // frame (contract section 6 rubric). Keyed "event:entity:label" ->
  // last-emit timestamp (us); cadence tracks its last state string.
  recent_emits: HashMap<String, u64>,
  last_cadence: String,
}

impl Cortex {
  pub fn spawn(rx: Receiver<SemanticEvent>, event_tx: crossbeam_channel::Sender<SemanticEvent>, memory_file: String) -> thread::JoinHandle<()> {
    thread::spawn(move || {
      // 1. Try to load existing memory from DataStore
      let store = DataStore::new();
      let (loaded_history, next_id) = Self::load_memory(&store);

      let mut cortex = Cortex {
        state: WorldState {
          entities: HashMap::new(),
          map: AcousticMap { zones: vec![] }, // Empty map for now
          sensors: HashMap::new(),
          context: HashMap::new(),
        },
        history: loaded_history,
        next_entity_id: next_id,
        store,
        event_tx,
        recent_emits: HashMap::new(),
        last_cadence: String::new(),
      };

      println!("Cortex Online. Known Entities: {}", cortex.history.len());
      cortex.run(rx);
    })
  }

  fn run(&mut self, rx: Receiver<SemanticEvent>) {
    let mut last_save = SystemTime::now();

    let g = DataStore::globals();
    loop {
      if !g.get_boolean("HOLLIS_CORTEX_RUNNING") {
          break;
      }
      
      // 1. Wait for events (with a timeout so we can run maintenance tasks)
      match rx.recv_timeout(Duration::from_millis(100)) {
        Ok(event) => self.handle_event(event),
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
          // No events? Good time to clean up memory.
          self.prune_stale_entities();
          //self.print_world_state();
          
          self.cluster_identities();

          // Auto-save every 10 seconds
          if last_save.elapsed().unwrap_or_default() > Duration::from_secs(10) {
            self.save_memory();
            last_save = SystemTime::now();
          }
        },
        Err(_) => break,
      }
    }
    // Final save on exit
    println!("Cortex loop ending, saving memory...");
    self.save_memory();
  }

  fn handle_event(&mut self, event: SemanticEvent) {
    let now = event.start_timestamp;
    
    // Do not assign 3D physical entities to internal LLM context updates
    if let EventKind::ContextUpdate(briefing) = event.kind {
        self.state.context.insert(briefing.domain.clone(), briefing);
        self.update_occupancy_context(); // Optional, but keeps state fresh
        return; 
    }
    
    let entity_id = self.identify_entity(&event); // Renamed from find_or_create

    // Update the entity (whether it's Active or revived from History)
    // Check Active list first
    if let Some(entity) = self.state.entities.get_mut(&entity_id) {
      Self::update_entity_stats(entity, &event);
    } else if let Some(mut entity) = self.history.remove(&entity_id) {
      // REVIVAL! It was in history, move it back to Active
      ///println!(">>> Welcome back, Entity #{} ({})", entity.id, entity.label);
      Self::update_entity_stats(&mut entity, &event);
      self.state.entities.insert(entity_id, entity);
    }

    match event.kind {
      EventKind::Transcript { text } => {
        println!(">> Entity #{} said: \"{}\"", entity_id, text);
        let text_content = text.clone();

        // 1. Build the inner DataObject
        let mut transcript_data = ndata::dataobject::DataObject::new();
        transcript_data.put_int("timestamp", now as i64);
        transcript_data.put_int("entity_id", entity_id as i64);
        transcript_data.put_string("text", &text_content);

        // 2. Wrap it in the required "data" envelope
        let mut envelope = ndata::dataobject::DataObject::new();
        envelope.put_object("data", transcript_data); 

        // 3. Save to DataStore! 
        self.store.set_data("hollis_transcripts", &now.to_string(), envelope);

        // Understanding is the executive's job (perception-contract:
        // ContextUpdate briefings retired) - the sensor stores the
        // transcript and proposes it below; it draws no conclusions.

        // Phase 9 (docs/perception-contract.md section 5): propose the
        // utterance to the agent executive - fire-and-forget on a thread
        // so the audio loop never blocks on the plugin host; a missing
        // agent library is a counted skip inside the emit path.
        {
          let etext = text_content.clone();
          let elabel = self.state.entities.get(&entity_id)
            .map(|e| e.label.clone())
            .unwrap_or_else(|| format!("Entity #{}", entity_id));
          let elabel = if elabel == "Sustained" || elabel == "Speaker" || elabel == "Unknown" {
            format!("Entity #{}", entity_id)
          } else { elabel };
          thread::spawn(move || {
            let _ = crate::hollis::audio::inject::emit_acoustic_perception("transcript", &etext, &elabel, 0.7, "");
          });
        }
      },
      EventKind::ContextUpdate(briefing) => {
        // [NEW] The Situation Room Update
        // We map the briefing by its domain (e.g., "Atmosphere")
        self.state.context.insert(briefing.domain.clone(), briefing);
      },
      other => {
        // H4c: the low levels reach the executive at the granularity
        // of MEANING, not volume (perception-contract section 6
        // rubric). Every arm goes through the one emit path with a
        // coalescing rule and a per-event runtime gate
        // (emit_<event>=on|off beside the master emit switch;
        // defaults are owner call 6's proposal). Payloads are
        // observations only; hints are honest priors.
        self.emit_low_level(entity_id, now, &other);
      }
    }

    self.update_occupancy_context();
  }

  // per-event emit gate; defaults per owner call 6: ambience_shift,
  // state_change, transient ON (located enforced at the emit site);
  // continuous ON but start-only by construction; micro_transient and
  // cadence OFF until turned on.
  fn event_gate(event: &str) -> bool {
    let dflt = match event {
      "ambience_shift" | "state_change" | "transient" | "continuous" => true,
      _ => false,
    };
    let g = DataStore::globals();
    if g.has("system") {
      let sys = g.get_object("system");
      if sys.has("apps") && sys.get_object("apps").has("hollis") {
        let happ = sys.get_object("apps").get_object("hollis");
        if happ.has("runtime") {
          let rt = happ.get_object("runtime");
          let key = format!("emit_{}", event);
          if rt.has(&key) { return rt.get_string(&key) != "off"; }
        }
      }
    }
    dflt
  }

  fn emit_low_level(&mut self, entity_id: u64, now: u64, kind: &EventKind) {
    // the entity's label and location, if it has earned them; the
    // origin means "not located" (the tracker's default position)
    let (elabel, loc) = match self.state.entities.get(&entity_id) {
      Some(e) => {
        let l = if e.label.starts_with("Entity #") || e.label == "Sustained"
            || e.label == "Speaker" || e.label == "Unknown" {
          format!("Entity #{}", entity_id)
        } else { e.label.clone() };
        let located = e.position.x != 0.0 || e.position.y != 0.0 || e.position.z != 0.0;
        (l, if located { Some((e.position.x, e.position.y, e.position.z)) } else { None })
      }
      None => (format!("Entity #{}", entity_id), None),
    };
    let loc_json = |loc: &Option<(f32, f32, f32)>| -> String {
      match loc {
        Some((x, y, z)) => format!(", \"location\": [{:.2}, {:.2}, {:.2}]", x, y, z),
        None => String::new(),
      }
    };
    // (event name, coalesce key, window us, hint, extra json) - or a
    // silent return where the rule says zero perceptions
    let (event, key, window_us, hint, extra): (&str, String, u64, f64, String) = match kind {
      EventKind::Transient { label, confidence, peak_db } => {
        // located transients only (call 6): an unlocated bang is
        // indistinguishable from sensor noise at perception grade
        if loc.is_none() { return; }
        ("transient", format!("transient:{}:{}", entity_id, label), 30_000_000, 0.35,
         format!("{{\"label\": \"{}\", \"confidence\": {:.2}, \"db\": {:.1}{}}}",
                 label.replace('"', ""), confidence, peak_db, loc_json(&loc)))
      }
      EventKind::MicroTransient { label, peak_db, margin } => {
        ("micro_transient", format!("micro:{}:{}", entity_id, label), 60_000_000, 0.15,
         format!("{{\"label\": \"{}\", \"db\": {:.1}, \"confidence\": {:.2}{}}}",
                 label.replace('"', ""), peak_db, 1.0 - margin, loc_json(&loc)))
      }
      EventKind::Continuous { label, is_speech } => {
        // a source STARTING is one event; a source running is zero -
        // the long window is the re-arm, not a repeat cadence
        ("continuous", format!("continuous:{}:{}", entity_id, label), 300_000_000, 0.35,
         format!("{{\"label\": \"{}\", \"confidence\": {}{}}}",
                 label.replace('"', ""), if *is_speech { "0.9" } else { "0.6" }, loc_json(&loc)))
      }
      EventKind::AmbienceShift { delta_db, new_floor_db } => {
        // already a shift by construction; the short window only
        // guards a flapping detector
        ("ambience_shift", "ambience".to_string(), 10_000_000, 0.3,
         format!("{{\"delta_db\": {:.1}, \"db\": {:.1}}}", delta_db, new_floor_db))
      }
      EventKind::StateChange { previous_db, current_db } => {
        ("state_change", "state".to_string(), 10_000_000, 0.3,
         format!("{{\"delta_db\": {:.1}, \"db\": {:.1}}}",
                 current_db - previous_db, current_db))
      }
      EventKind::CadenceUpdate { state, duration_ms } => {
        // one perception per cadence STATE CHANGE, never per update
        let s = format!("{:?}", state);
        if s == self.last_cadence { return; }
        self.last_cadence = s.clone();
        ("cadence", "cadence".to_string(), 0, 0.2,
         format!("{{\"label\": \"{}\", \"duration_ms\": {}}}", s, duration_ms))
      }
      _ => return,
    };
    if !Self::event_gate(event) { return; }
    if window_us > 0 {
      if let Some(last) = self.recent_emits.get(&key) {
        if now.saturating_sub(*last) < window_us { return; }
      }
    }
    self.recent_emits.insert(key, now);
    if self.recent_emits.len() > 512 { self.recent_emits.clear(); }
    let ev = event.to_string();
    let el = elabel.clone();
    thread::spawn(move || {
      let _ = crate::hollis::audio::inject::emit_acoustic_perception(&ev, "", &el, 0.0f64.max(hint), &extra);
    });
  }

  fn update_entity_stats(entity: &mut Entity, event: &SemanticEvent) {
    let dt = (event.start_timestamp as i64 - entity.last_heard as i64) as f32 / 1_000_000.0;
    entity.last_heard = event.start_timestamp;

    match event.kind {
      EventKind::Transient { ref label, .. } => {
        println!("Entity #{} made a noise: {}", entity.id, label);
      },
      EventKind::Continuous { ref label, .. } => {
        if label != "Sustained" {
          if entity.label.starts_with("Entity #") || entity.label == "Speaker" || entity.label == "Unknown" {
            entity.label = label.clone();
          }
        }
      }
      _ => {}
    }

    // LEARN: 3D Kalman / Alpha-Beta Filter
    if !event.fingerprint.is_empty() {
      let loc = event.location.unwrap_or(entity.position);

      if dt > 0.0 && dt < 2.0 {
        // Alpha-Beta Filter for 3D Space
        let alpha = 0.85;
        let beta = 0.20;

        let xp = entity.position.x + entity.velocity.dx * dt;
        let yp = entity.position.y + entity.velocity.dy * dt;
        let zp = entity.position.z + entity.velocity.dz * dt;

        let rx = loc.x - xp;
        let ry = loc.y - yp;
        let rz = loc.z - zp;

        entity.position.x = xp + alpha * rx;
        entity.position.y = yp + alpha * ry;
        entity.position.z = zp + alpha * rz;

        entity.velocity.dx = entity.velocity.dx + (beta / dt) * rx;
        entity.velocity.dy = entity.velocity.dy + (beta / dt) * ry;
        entity.velocity.dz = entity.velocity.dz + (beta / dt) * rz;
      } else {
        entity.position = loc;
        entity.velocity.dx *= 0.1; 
        entity.velocity.dy *= 0.1;
        entity.velocity.dz *= 0.1;
      }

      // Update Signature
      let sig_len = entity.signature.len();
      if sig_len > 0 && event.fingerprint.len() == sig_len {
        for i in 0..sig_len {
          entity.signature[i] = (entity.signature[i] * 0.9) + (event.fingerprint[i] * 0.1);
        }
      } else {
        entity.signature = event.fingerprint.clone();
      }
    }
  }

  fn identify_entity(&mut self, event: &SemanticEvent) -> u64 {
    let mut best_match = None;
    let mut best_score = 0.0;
    let mut best_components = (0.0, 0.0, 0.0); 
    let mut debug_dt = 0.0;

    let input_spectrum = &event.fingerprint;
    // Default to origin if no location provided
    let input_loc = event.location.unwrap_or(Point3D::new(0.0, 0.0, 0.0));

    let mut calc_bio = |entity: &Entity| -> f32 {
      if !entity.signature.is_empty() && entity.signature.len() == input_spectrum.len() {
          cosine_similarity(&entity.signature, input_spectrum)
      } else { 0.0 }
    };

    // 1. Active Entities
    for entity in self.state.entities.values() {
      let dt = (event.start_timestamp as i64 - entity.last_heard as i64) as f32 / 1_000_000.0;

      let pred_loc = if dt > 0.0 && dt < 5.0 {
        Point3D::new(
          entity.position.x + entity.velocity.dx * dt,
          entity.position.y + entity.velocity.dy * dt,
          entity.position.z + entity.velocity.dz * dt
        )
      } else { entity.position };

      let bio_score = calc_bio(entity);
      let spatial_diff = input_loc.distance(&pred_loc); 

      // Hard Gate: Relaxed to 1.5 since ratio-space max distance is ~3.4
      if spatial_diff > 1.5 { continue; }

      let spatial_bonus = if spatial_diff < 0.5 { 0.5 } else { 0.0 };
      let time_bonus = if dt < 3.0 { 0.2 } else { 0.0 };
      let total_score = bio_score + spatial_bonus + time_bonus;

      if total_score > best_score {
        best_score = total_score;
        best_match = Some(entity.id);
        best_components = (bio_score, spatial_bonus, time_bonus);
        debug_dt = dt;
      }
    }

    // 2. Historical Entities
    for entity in self.history.values() {
      let bio_score = calc_bio(entity);
      if bio_score > best_score {
        best_score = bio_score;
        best_match = Some(entity.id);
        best_components = (bio_score, 0.0, 0.0);
        debug_dt = 999.0;
      }
    }

    if best_score > 0.45 {
      let id = best_match.unwrap();
      // RESTORED DEBUG LOG
      println!("> Matched #{} (Score: {:.2} [Bio:{:.2} Spc:{:.2} Tim:{:.2}], dt:{:.2}s)", 
        id, best_score, best_components.0, best_components.1, best_components.2, debug_dt);
      return id; 
    }

    // 3. New Entity Creation
    let id = self.next_entity_id;
    self.next_entity_id += 1;

    let new_entity = Entity {
      id,
      label: format!("Entity #{}", id),
      position: input_loc,
      velocity: Vector3D {dx:0.0, dy:0.0, dz:0.0},
      last_heard: event.start_timestamp,
      signature: event.fingerprint.clone(), 
      signature_count: 1,
    };
    self.state.entities.insert(id, new_entity);

    // RESTORED DEBUG LOG
    println!(">>> New Entity Detected: #{} (Score: {:.2}, Loc: [{:.2}, {:.2}, {:.2}])", 
      id, best_score, input_loc.x, input_loc.y, input_loc.z);

    id
  }

  fn prune_stale_entities(&mut self) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;
    let active_timeout = 30_000_000; // 30s
    
    let stale_ids: Vec<u64> = self.state.entities.iter()
    .filter(|(_, e)| (now - e.last_heard) > active_timeout)
    .map(|(&id, _)| id)
    .collect();

    for id in stale_ids {
      if let Some(e) = self.state.entities.remove(&id) {
        // FLUSH TO DISK AS IT GOES TO SLEEP
        save_entity_to_datastore(&self.store, &e); 
        self.history.insert(id, e);
      }
    }

    // 2. Permanent Garbage Collection (The Memory Leak Fix)
    let history_timeout = 86_400_000_000; // 24 hours in microseconds
    
    self.history.retain(|_, e| {
      let time_in_history = now - e.last_heard;
      
      // Delete if it has been silent for more than 24 hours
      if time_in_history > history_timeout { return false; }
      
      // Delete if it has been silent for over an hour AND only ever made 1 sound 
      // (This purges random chair squeaks and dropped books)
      if time_in_history > 3_600_000_000 && e.signature_count <= 1 { return false; }
      
      true // Keep it!
    });
  } 

  fn cluster_identities(&mut self) {
    // 1. Gather all entities (Active + History) to compare
    let mut all_entities: Vec<Entity> = self.state.entities.values().cloned().collect();
    all_entities.extend(self.history.values().cloned());
    
    let mut merges: Vec<(u64, u64)> = Vec::new();
    
    // 2. Compare everyone against everyone
    for i in 0..all_entities.len() {
        for j in (i + 1)..all_entities.len() {
            let a = &all_entities[i];
            let b = &all_entities[j];
            
            // Biometric Similarity (MFCC Voiceprint)
            let bio_score = cosine_similarity(&a.signature, &b.signature);
            // Spatial Similarity (Distance in meters)
            let spatial_diff = a.position.distance(&b.position);
            
            // THE THRESHOLD: High voice match (> 88%) AND physically close (< 0.5m)
            if bio_score > 0.88 && spatial_diff < 0.5 {
                let survivor = std::cmp::min(a.id, b.id);
                let duplicate = std::cmp::max(a.id, b.id);
                
                if !merges.contains(&(survivor, duplicate)) {
                    merges.push((survivor, duplicate));
                }
            }
        }
    }
    
    if merges.is_empty() { return; }
    
    // 3. Apply the merges!
    for (survivor_id, duplicate_id) in merges {
        println!(">>> CLUSTERING: Merging fragmented Entity #{} into true Entity #{}", duplicate_id, survivor_id);
        
        // Grab the duplicate's latest data before we delete it
        let duplicate_last_heard = self.state.entities.get(&duplicate_id)
            .or(self.history.get(&duplicate_id))
            .map(|e| e.last_heard).unwrap_or(0);
            
        let duplicate_pos = self.state.entities.get(&duplicate_id)
            .or(self.history.get(&duplicate_id))
            .map(|e| e.position.clone());

        // Remove the duplicate from live memory and history
        self.state.entities.remove(&duplicate_id);
        self.history.remove(&duplicate_id);

        // EXORCISE THE GHOST: Delete the orphan file from the DataStore
        let duplicate_path = self.store.get_data_file("hollis_entities", &duplicate_id.to_string());
        if duplicate_path.exists() {
            let _ = std::fs::remove_file(duplicate_path);
        }

        // Update the survivor's timestamps and location so it stays alive!
        if let Some(survivor) = self.state.entities.get_mut(&survivor_id).or_else(|| self.history.get_mut(&survivor_id)) {
            if duplicate_last_heard > survivor.last_heard {
                survivor.last_heard = duplicate_last_heard;
                if let Some(pos) = duplicate_pos {
                    survivor.position = pos;
                }
            }
            // Flush the updated survivor to disk immediately
            save_entity_to_datastore(&self.store, survivor);
        }

    }
  }
  
  fn update_occupancy_context(&mut self) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;
    let timeout = 60_000_000; // 60 seconds

    // 1. Find who is actually here right now
    let active_entities: Vec<&Entity> = self.state.entities.values()
    .filter(|e| now > e.last_heard && (now - e.last_heard) < timeout)
    .collect();

    // 2. Draft the summary
    let summary = if active_entities.is_empty() {
      "The area is currently vacant.".to_string()
    } else {
      let count = active_entities.len();

      // Name Dropping Logic (Sanitized)
      let names: Vec<String> = active_entities.iter()
      .map(|e| {
        // FILTER: If the label is technical garbage, use the ID instead
        if e.label == "Sustained" || e.label == "Speaker" || e.label == "Unknown" {
          format!("Entity #{}", e.id)
        } else {
          e.label.clone()
        }
      })
      // Filter out "Entity #123" if you only want to list "Real Names"
      // For now, let's keep them so you see who is there.
      .collect();

      // Create the string (e.g., "Marc, Entity #309")
      let name_list = names.join(", ");
      format!("{} is present.", name_list)
    };

    // 3. Update the Situation Room
    let old_summary = self.state.context.get("Occupancy")
    .map(|b| b.summary.clone())
    .unwrap_or_default();

    if summary != old_summary {
      println!(">>> CONTEXT SHIFT: {}", summary);
      self.state.context.insert("Occupancy".to_string(), ContextBriefing {
        domain: "Occupancy".to_string(),
        summary,
        confidence: 1.0,
        urgency: 1,
        timestamp: now,
      });
    }
  }

  fn load_memory(store: &DataStore) -> (HashMap<u64, Entity>, u64) {
    use std::fs;
    use std::path::Path;

    let mut history = HashMap::new();
    let mut max_id = 0;
    
    // The base directory where DataStore keeps our sharded entities
    let base_dir = store.root.join("hollis_entities");
    
    if base_dir.exists() {
      // Helper function to recursively walk the sharded DataStore directories
      fn walk_dir(dir: &Path, store: &DataStore, history: &mut HashMap<u64, Entity>, max_id: &mut u64) {
        if let Ok(entries) = fs::read_dir(dir) {
          for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
              walk_dir(&path, store, history, max_id);
            } else if path.is_file() {
              // In DataStore, the deepest file is named exactly the ID
              if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if let Ok(id) = filename.parse::<u64>() {
                  // Track the highest ID so we don't accidentally overwrite someone
                  if id > *max_id { *max_id = id; }
                  
                  // Use our new DataStore loader!
                  if let Some(entity) = load_entity_from_datastore(store, id) {
                    history.insert(id, entity);
                  }
                }
              }
            }
          }
        }
      }
      
      walk_dir(&base_dir, store, &mut history, &mut max_id);
    }
    
    // Next ID is always one higher than the highest we found
    (history, max_id + 1)
  }

  fn save_memory(&self) {
    // 1. Save all active entities
    for entity in self.state.entities.values() {
        save_entity_to_datastore(&self.store, entity);
    }
    
    // 2. Save all historical entities
    for entity in self.history.values() {
        save_entity_to_datastore(&self.store, entity);
    }
  }
}

fn save_entity_to_datastore(store: &DataStore, entity: &Entity) {
    let mut entity_data = DataObject::new();
    entity_data.put_int("id", entity.id as i64);
    entity_data.put_string("label", &entity.label);
    entity_data.put_int("last_heard", entity.last_heard as i64);

    // 1. Serialize Position
    let mut pos_data = DataObject::new();
    pos_data.put_float("x", entity.position.x as f64);
    pos_data.put_float("y", entity.position.y as f64);
    pos_data.put_float("z", entity.position.z as f64);
    entity_data.put_object("position", pos_data); // Moves pos_data

    // 2. Serialize Velocity
    let mut vel_data = DataObject::new();
    vel_data.put_float("dx", entity.velocity.dx as f64);
    vel_data.put_float("dy", entity.velocity.dy as f64);
    vel_data.put_float("dz", entity.velocity.dz as f64);
    entity_data.put_object("velocity", vel_data); // Moves vel_data

    // 3. Serialize Signature (FIXED: f32 to f64 cast)
    let mut signature_array = DataArray::new();
    for &val in &entity.signature {
        signature_array.push_float(val as f64); 
    }
    entity_data.put_array("signature", signature_array); // Moves signature_array

    // 4. Wrap in the required "data" envelope
    let mut envelope = DataObject::new();
    envelope.put_object("data", entity_data);

    // 5. Write to disk
    let id_str = entity.id.to_string();
    store.set_data("hollis_entities", &id_str, envelope);
}

fn load_entity_from_datastore(store: &DataStore, id: u64) -> Option<Entity> {
    let id_str = id.to_string();
    
    // Check if the file exists using the hashed path
    if !store.exists("hollis_entities", &id_str) {
        return None;
    }

    // Retrieve the DataObject
    let envelope = store.get_data("hollis_entities", &id_str);
    let entity_data = envelope.get_object("data");

    // 1. Extract Signature (FIXED: f64 to f32 cast)
    let sig_array = entity_data.get_array("signature");
    let mut signature = Vec::new();
    for val in sig_array.objects() {
        signature.push(val.float() as f32); 
    }

    // 2. Extract Position (with a safe fallback to origin)
    let position = if entity_data.has("position") {
        let p = entity_data.get_object("position");
        Point3D {
            x: p.get_float("x") as f32,
            y: p.get_float("y") as f32,
            z: p.get_float("z") as f32,
        }
    } else {
        Point3D::new(0.0, 0.0, 0.0)
    };

    // 3. Extract Velocity (with a safe fallback to zero)
    let velocity = if entity_data.has("velocity") {
        let v = entity_data.get_object("velocity");
        Vector3D {
            dx: v.get_float("dx") as f32,
            dy: v.get_float("dy") as f32,
            dz: v.get_float("dz") as f32,
        }
    } else {
        Vector3D { dx: 0.0, dy: 0.0, dz: 0.0 }
    };

    // 4. Fully initialize the struct (FIXED: Missing fields)
    Some(Entity {
        id: entity_data.get_int("id") as u64,
        label: entity_data.get_string("label"),
        last_heard: entity_data.get_int("last_heard") as u64,
        signature,
        signature_count: 1, 
        position, 
        velocity, 
    })
}

fn qwert(){
}
