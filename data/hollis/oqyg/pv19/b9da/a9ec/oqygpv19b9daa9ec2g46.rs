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
    let model_path = match meta.has("model") { true => meta.get_string("model"), _ => "models/ggml-medium.en.bin".to_string() };
    let memory_file = match meta.has("memory_file") { true => meta.get_string("memory_file"), _ => "hollis_memory.json".to_string() };
    let mic_distance = match meta.has("mic_distance") { true => meta.get_string("mic_distance").parse().unwrap(), _ => 0.6 };

    g.put_boolean(RUNPARAM, true);

    let (audio_tx, audio_rx) = crossbeam_channel::bounded(50);
    // Event channel: Both Perception AND Transcriber write to this
    let (event_tx, event_rx) = crossbeam_channel::unbounded();

    // Transcriber channel: Perception writes to this
    let (transcribe_tx, transcribe_rx) = crossbeam_channel::unbounded();

    println!("Initializing Hollis...");
    println!("Config: Model='{}', Memory='{}', MicDist={:.2}m", model_path, memory_file, mic_distance);

    // 1. Spawn Transcriber
    // It listens to Perception and sends results to Cortex (via event_tx)
    let transcriber = crate::hollis::audio::transcribe::Transcriber::new(model_path);
    let t_handle = transcriber.spawn(transcribe_rx, event_tx.clone());

    // 2. Spawn Cortex
    let c_handle = Cortex::spawn(event_rx, memory_file);

    // 3. Spawn Perception
    // Note: We pass Some(transcribe_tx) now
    let p_handle = PerceptionEngine::spawn(
      audio_rx, 
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

    // Ear 1 (Left / Ch 0)
    let _sensor_left = crate::hollis::audio::sensor::AudioSensor::new(
      "ear_left".to_string(), 
      ear_left.clone(),
      if is_stereo_pair { Some(0) } else { None }, // Force Ch0 if stereo pair
      audio_tx.clone()
    ).expect("Left Ear failed");

    // Ear 2 (Right / Ch 1)
    let _sensor_right = match ear_right {
      Some(name) => crate::hollis::audio::sensor::AudioSensor::new(
        "ear_right".to_string(), 
        name,
        if is_stereo_pair { Some(1) } else { None }, // Force Ch1 if stereo pair
        audio_tx
      ).ok(),
      _ => None
    };

    println!("Hollis Cortex Active.");

    thread::spawn(move || {
      let beat = std::time::Duration::from_secs(1);
      while g.get_boolean(RUNPARAM) { std::thread::sleep(beat); }

      println!("Stopping threads...");

      // 1. Drop the sensor to close audio_tx
      // This causes Perception to finish -> Closes transcribe_tx -> Transcriber finishes
      drop(_sensor_left);
      drop(_sensor_right);

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
  conversation_buffer: Vec<(u64, u64, String)>,
  last_transcript_time: u64,
  memory_file: String,  
}

impl Cortex {
  pub fn spawn(rx: Receiver<SemanticEvent>, memory_file: String) -> thread::JoinHandle<()> {
    thread::spawn(move || {
      // 1. Try to load existing memory from disk
      let (loaded_history, next_id) = Self::load_memory(&memory_file);

      let mut cortex = Cortex {
        state: WorldState {
          entities: HashMap::new(),
          map: AcousticMap { zones: vec![] }, // Empty map for now
          sensors: HashMap::new(),
          context: HashMap::new(),
        },
        history: loaded_history,
        next_entity_id: next_id,
        conversation_buffer: Vec::new(),
        last_transcript_time: 0,
        memory_file,
      };

      println!("Cortex Online. Known Entities: {}", cortex.history.len());
      cortex.run(rx);
    })
  }

  fn run(&mut self, rx: Receiver<SemanticEvent>) {
    let mut last_save = SystemTime::now();

    loop {
      // 1. Wait for events (with a timeout so we can run maintenance tasks)
      match rx.recv_timeout(Duration::from_millis(100)) {
        Ok(event) => self.handle_event(event),
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
          // No events? Good time to clean up memory.
          self.prune_stale_entities();
          //self.print_world_state();

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

        // Write to log (Existing logic)
        let logfile = DataStore::new().root.parent().unwrap().join("runtime").join("hollis").join("transcript.txt");
        match OpenOptions::new().write(true).create(true).append(true).open(&logfile) {
          Ok(mut file) => {
            if let Err(e) = file.write_all((text_content + "\n").as_bytes()) {
              eprintln!("Failed to write log entry: {}", e);
            }
          },
          Err(e) => eprintln!("Failed to open log file: {}", e),
        };

        self.analyze_transcript(entity_id, text.clone());
      },
      EventKind::ContextUpdate(briefing) => {
        // [NEW] The Situation Room Update
        // We map the briefing by its domain (e.g., "Atmosphere")
        self.state.context.insert(briefing.domain.clone(), briefing);
      },
      _ => {
        // Catch-all for Transients/Continuous (or just log them)
        println!("{}: {:?}", entity_id, event.kind);
      }
    }

    self.update_occupancy_context();
  }

  fn analyze_transcript(&mut self, entity_id: u64, text: String) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;
    let window_duration = 300_000_000; // 5 Minutes

    // 1. Add to buffer (Raw Data)
    self.conversation_buffer.push((now, entity_id, text));

    // 2. Sliding Window Pruning
    self.conversation_buffer.retain(|(ts, _, _)| now - *ts < window_duration);

    // 3. Generate Briefing
    self.generate_discourse_briefing();
  }

  fn generate_discourse_briefing(&mut self) {
    if self.conversation_buffer.is_empty() { return; }

    // 1. Analyze Dynamics (Who is here?)
    let mut participants: Vec<u64> = self.conversation_buffer.iter().map(|(_, id, _)| *id).collect();
    participants.sort();
    participants.dedup();

    let speaker_count = participants.len();
    let dynamic_type = match speaker_count {
      1 => "Monologue (Single Speaker)",
      2 => "Dialogue (Two Speakers)",
      _ => "Group Discussion",
    };

    // 2. Reconstruct Transcript with LATEST Labels
    let transcript_block = self.conversation_buffer.iter()
    .map(|(_, id, text)| {
      let label = self.state.entities.get(id)
      .map(|e| {
        if e.label == "Sustained" || e.label == "Speaker" || e.label == "Unknown" {
          format!("Entity #{}", id)
        } else {
          e.label.clone()
        }
      })
      .unwrap_or(format!("Entity #{}", id));
      format!("{}: {}", label, text)
    })
    .collect::<Vec<String>>()
    .join("\n");

    // println!(">>> GENERATING DISCOURSE BRIEFING ({}, {} lines)...", dynamic_type, self.conversation_buffer.len());

    // 3. Ask LLM
    let system_prompt = format!(
      "You are the cognitive center of an intelligent system. \
      Analyze the following recent conversation (last 5 minutes). \
      Context: This is a {}. \
      Summarize the active topic, intent, or ongoing activity in one concise sentence. \
      If the conversation seems fragmented or over, say 'No active topic'.", 
      dynamic_type
    );

    let prompt = format!("TRANSCRIPT:\n{}\n\nCURRENT CONTEXT:", transcript_block);

    if let Ok(api) = std::panic::catch_unwind(|| crate::api::new()) {
      let response = ask_llm(prompt, Data::DString(system_prompt)); // Note: system_prompt is now a String, not &str
      let summary = response.trim().to_string();

      // 4. Update Situation Room
      let old_summary = self.state.context.get("Discourse")
      .map(|b| b.summary.clone())
      .unwrap_or_default();

      if summary != old_summary {
        println!(">>> CONTEXT SHIFT (Discourse): {}", summary);
        self.state.context.insert("Discourse".to_string(), ContextBriefing {
          domain: "Discourse".to_string(),
          summary,
          confidence: 0.9,
          urgency: 1,
          timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64,
        });
      }
    }
  }

  fn update_entity_stats(entity: &mut Entity, event: &SemanticEvent) {
    let dt = (event.start_timestamp as i64 - entity.last_heard as i64) as f32 / 1_000_000.0;
    entity.last_heard = event.start_timestamp;

    match event.kind {
      // [RESTORED] This was missing in the previous update!
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

    // LEARN: Kalman / Alpha-Beta Filter
    if !event.fingerprint.is_empty() {
      // [Cite: 282, 283] Extract Measured Angle (Last element)
      let measured_angle = event.angle.unwrap_or(0.0);
      let xm = measured_angle.sin();
      let ym = measured_angle.cos();

      if dt > 0.0 && dt < 2.0 {
        // Alpha-Beta Filter
        let alpha = 0.85;
        let beta = 0.20;

        let xp = entity.position.x + entity.velocity.dx * dt;
        let yp = entity.position.y + entity.velocity.dy * dt;

        let rx = xm - xp;
        let ry = ym - yp;

        entity.position.x = xp + alpha * rx;
        entity.position.y = yp + alpha * ry;

        entity.velocity.dx = entity.velocity.dx + (beta / dt) * rx;
        entity.velocity.dy = entity.velocity.dy + (beta / dt) * ry;
      } else {
        entity.position.x = xm;
        entity.position.y = ym;
        entity.velocity.dx *= 0.1; 
        entity.velocity.dy *= 0.1;
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

    // Debug info
    let mut best_components = (0.0, 0.0, 0.0); 
    let mut debug_dt = 0.0;
    let mut debug_bio_err = String::new();

    let input_spectrum = &event.fingerprint;
    let input_angle = event.angle.unwrap_or(0.0);

    // Helper closure to calculate bio score
    let mut calc_bio = |entity: &Entity| -> f32 {
      if !entity.signature.is_empty() {
        if entity.signature.len() != input_spectrum.len() {
          if debug_bio_err.is_empty() { 
            debug_bio_err = format!("Len Mismatch: Ent {} vs In {}", entity.signature.len(), input_spectrum.len());
          }
          0.0
        } else {
          cosine_similarity(&entity.signature, input_spectrum)
        }
      } else {
        0.0
      }
    };

    // 1. Active Entities (Full Spatial + Temporal Logic)
    for entity in self.state.entities.values() {
      let dt = (event.start_timestamp as i64 - entity.last_heard as i64) as f32 / 1_000_000.0;

      // Prediction
      let (pred_x, pred_y) = if dt > 0.0 && dt < 5.0 {
        (
          entity.position.x + entity.velocity.dx * dt,
          entity.position.y + entity.velocity.dy * dt
        )
      } else {
        (entity.position.x, entity.position.y)
      };
      let pred_angle = pred_x.atan2(pred_y);

      // Biometrics
      let bio_score = calc_bio(entity);

      // Spatial
      let angle_diff = (input_angle - pred_angle).abs();

      // Hard Gate: If an active entity is physically far away, it's not them.
      if angle_diff > 0.8 { continue; }

      let spatial_bonus = if angle_diff < 0.3 { 0.5 } else { 0.0 };
      let time_bonus = if dt < 3.0 { 0.2 } else { 0.0 };

      let total_score = bio_score + spatial_bonus + time_bonus;

      if total_score > best_score {
        best_score = total_score;
        best_match = Some(entity.id);
        best_components = (bio_score, spatial_bonus, time_bonus);
        debug_dt = dt;
      }
    }

    // 2. Historical Entities (Biometrics Only)
    // We ignore position because they could have moved while "sleeping".
    for entity in self.history.values() {
      let bio_score = calc_bio(entity);

      // Score is purely biometric (0.0 to 1.0)
      let total_score = bio_score; 

      if total_score > best_score {
        best_score = total_score;
        best_match = Some(entity.id);
        best_components = (bio_score, 0.0, 0.0);
        debug_dt = 999.0; // Stale
      }
    }

    if best_score > 0.45 {
      let id = best_match.unwrap();
      // [DEBUG] Now printing Time Bonus and DT
      println!("> Matched #{} (Score: {:.2} [Bio:{:.2} Spc:{:.2} Tim:{:.2}], dt:{:.2}s)", 
        id, best_score, best_components.0, best_components.1, best_components.2, debug_dt);
      return id;
    }

    // New Entity Creation
    let id = self.next_entity_id;
    self.next_entity_id += 1;
    let x = input_angle.sin();
    let y = input_angle.cos();

    let new_entity = Entity {
      id,
      label: format!("Entity #{}", id),
      position: Point3D::new(x, y, 0.0),
      velocity: Vector3D {dx:0.0, dy:0.0, dz:0.0},
      last_heard: event.start_timestamp,
      signature: event.fingerprint.clone(), 
      signature_count: 1,
    };
    self.state.entities.insert(id, new_entity);

    // [DEBUG] Print Bio Error if it exists
    let err_msg = if !debug_bio_err.is_empty() { format!(" ({})", debug_bio_err) } else { "".to_string() };

    println!(">>> New Entity Detected: #{} (Score: {:.2}, Angle: {:.2}){}", 
      id, best_score, input_angle, err_msg);

    id
  }

  fn prune_stale_entities(&mut self) {
    // Move from Active -> History
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;
    let timeout = 30_000_000; // 30s

    let stale_ids: Vec<u64> = self.state.entities.iter()
    .filter(|(_, e)| (now - e.last_heard) > timeout)
    .map(|(&id, _)| id)
    .collect();

    for id in stale_ids {
      if let Some(e) = self.state.entities.remove(&id) {
        //println!(">>> Entity #{} stored in memory.", id);
        self.history.insert(id, e);
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

  fn load_memory(filename: &str) -> (HashMap<u64, Entity>, u64) {
    let memfile = DataStore::new().root.parent().unwrap().join("runtime").join("hollis").join(filename);
    if let Ok(file) = File::open(memfile) {
      let reader = BufReader::new(file);
      if let Ok(history) = serde_json::from_reader::<_, HashMap<u64, Entity>>(reader) {
        // Find the highest ID so we don't reuse IDs
        let max_id = history.keys().max().unwrap_or(&0) + 1;
        return (history, max_id);
      }
    }
    (HashMap::new(), 1)
  }

  fn save_memory(&self) {
    // We save BOTH active entities and history to ensure nothing is lost on crash
    let mut full_roster = self.history.clone();
    for (id, entity) in &self.state.entities {
      full_roster.insert(*id, entity.clone());
    }

    let memfile = DataStore::new().root.parent().unwrap().join("runtime").join("hollis").join(&self.memory_file);
    if let Ok(file) = OpenOptions::new()
    .write(true)
    .create(true)
    .truncate(true)
    .open(memfile) 
    {
      let writer = BufWriter::new(file);
      let _ = serde_json::to_writer(writer, &full_roster);
    }
  }
}

fn qwert(){