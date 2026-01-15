DataObject::new()
}

// --- Tuning Parameters ---
const NOISE_LEARNING_RATE: f32 = 0.05; 
const TRANSIENT_THRESHOLD: f32 = 4.0; // Needs to be 4x louder than floor
const MIN_ABSOLUTE_RMS: f32 = 0.005;  // ~ -46dB. Signals below this are ignored completely.
const WARMUP_FRAMES: usize = 100;     // Spend the first ~5 seconds just listening
const HANGOVER_MS: u64 = 1500; // Wait 1.5 seconds of silence before cutting the stream

// Constants for Binaural Math
const SPEED_OF_SOUND: f32 = 343.0;
const SAMPLE_RATE: f32 = 44100.0;     // Matches sensor config

pub struct PerceptionEngine {
  noise_floor_rms: f32,
  is_tracking_event: bool,
  event_start_time: Option<u64>,
  consecutive_loud_frames: usize,
  frame_count: usize, // Track total lifetime to handle warmup
  transcriber_tx: Option<Sender<TranscriberMessage>>, // Channel to talk to the Transcriber thread
  silence_start_time: Option<u64>, // Track when silence began
  last_briefing_summary: String, // To prevent spam
  stereo_buffer: HashMap<u64, Vec<AcousticFrame>>,
  mic_distance: f32,
}

impl PerceptionEngine {
  pub fn spawn(
    rx: Receiver<AcousticFrame>,
    tx: Sender<SemanticEvent>,
    transcriber_tx: Option<Sender<TranscriberMessage>>,
    mic_distance: f32, 
  ) -> thread::JoinHandle<()> {
    thread::spawn(move || {
      let mut engine = PerceptionEngine {
        noise_floor_rms: 0.0,
        is_tracking_event: false,
        event_start_time: None,
        consecutive_loud_frames: 0,
        frame_count: 0,
        transcriber_tx,
        silence_start_time: None,
        last_briefing_summary: String::new(),
        stereo_buffer: HashMap::new(),
        mic_distance,
      };

      while let Ok(frame) = rx.recv() {
        engine.ingest_frame(frame, &tx);
      }
    })
  }

  // [NEW] The Entry Point: Handles buffering and pairing
  fn ingest_frame(&mut self, frame: AcousticFrame, tx: &Sender<SemanticEvent>) {
    // 1. Quantize timestamp to 50ms buckets to catch slightly out-of-sync frames
    let bucket = frame.timestamp_micros / 50_000;

    let entry = self.stereo_buffer.entry(bucket).or_insert_with(Vec::new);
    entry.push(frame);

    // 2. Check if we have a stereo pair
    if entry.len() >= 2 {
      // Clone the pair to process (assuming index 0 is left, 1 is right is NOT safe, 
      // but for fusion averaging it doesn't matter yet).
      let frames = entry.clone();
      self.stereo_buffer.remove(&bucket); // Clear buffer

      // 3. Fuse the two frames into one "Super Frame"
      if let Some(fused) = self.fuse_stereo_pair(frames) {
        // 4. Pass the fused frame to your ORIGINAL analysis logic
        self.analyze_frame(&fused, tx);
      }
    }

    // Cleanup old buckets (> 200ms) to prevent memory leaks
    let oldest_allowed = bucket.saturating_sub(4);
    self.stereo_buffer.retain(|&k, _| k >= oldest_allowed);
  }

  // The Math: Combines two ears into one brain
  fn fuse_stereo_pair(&self, frames: Vec<AcousticFrame>) -> Option<AcousticFrame> {
    if frames.len() < 2 { return None; }

    let f1 = &frames[0];
    let f2 = &frames[1];

    // [CHANGED] Calculate Angle via Loudness (ILD) instead of Phase (TDOA)
    // TDOA fails on unsynced USB devices. ILD works on relative volume.
    let total_pwr = f1.rms_power + f2.rms_power;
    let angle = if total_pwr > 0.0001 {
        // Pan ratio: -1.0 (Right Louder) to +1.0 (Left Louder)
        // [UPDATED] Use dynamic mic_distance if we were doing TDOA, 
        // but for ILD we map -1.0..1.0 to -90..90 degrees.
        // However, if we ever switch back to TDOA (via calculate_doa below),
        // we need to pass self.mic_distance.
        let pan = (f1.rms_power - f2.rms_power) / total_pwr;
        // Map to -90 to +90 degrees (approx -1.57 to +1.57 radians)
        pan * 1.5708
    } else {
        0.0
    };

    // B. Average the Metrics
    let avg_rms = (f1.rms_power + f2.rms_power) / 2.0;
    let avg_zcr = (f1.zcr + f2.zcr) / 2.0;
    let avg_centroid = (f1.spectral_centroid + f2.spectral_centroid) / 2.0;
    
    // C. Average the Spectrum
    let len = f1.spectrum.len().min(f2.spectrum.len());
    let avg_spectrum: Vec<f32> = (0..len)
        .map(|i| (f1.spectrum[i] + f2.spectrum[i]) / 2.0)
        .collect();

    // D. Pick the best audio
    let best_samples = if f1.rms_power > f2.rms_power { &f1.raw_samples } else { &f2.raw_samples };

    Some(AcousticFrame {
      sensor_id: "fused_array".to_string(),
      timestamp_micros: f1.timestamp_micros,
      rms_power: avg_rms,
      dominant_freq_hz: f1.dominant_freq_hz,
      spectrum: avg_spectrum,
      zcr: avg_zcr,
      spectral_centroid: avg_centroid,
      direction_of_arrival: Some((angle, 0.0)), // Now using the Loudness Angle
      raw_samples: best_samples.clone(),
    })
  }

  fn analyze_frame(&mut self, frame: &AcousticFrame, tx: &Sender<SemanticEvent>) {
    self.frame_count += 1;
    let current_rms = frame.rms_power;

    // --- Phase 1: Warmup / Calibration ---
    if self.frame_count < WARMUP_FRAMES {
      if self.frame_count == 1 {
        self.noise_floor_rms = current_rms;
      } else {
        self.noise_floor_rms = lerp(self.noise_floor_rms, current_rms, 0.2);
      }
      if self.frame_count % 20 == 0 {
        println!("Calibrating... Floor: {:.5} (Current: {:.5})", self.noise_floor_rms, current_rms);
      }
      return;
    }

    // --- Phase 2: Active Monitoring ---

    // Condition A: It is quieter than our floor (Ambient noise dropped)
    if current_rms < self.noise_floor_rms {
      self.noise_floor_rms = lerp(self.noise_floor_rms, current_rms, NOISE_LEARNING_RATE);
      // If we were tracking, the Hangover logic below will handle the finish.
    }

    // Condition B: Is it loud enough to be an event?
    let is_loud = current_rms > MIN_ABSOLUTE_RMS 
    && current_rms > (self.noise_floor_rms * TRANSIENT_THRESHOLD);

    if is_loud {
      self.consecutive_loud_frames += 1;
      self.silence_start_time = None; // RESET timer (we heard sound)

      if !self.is_tracking_event {
        self.is_tracking_event = true;
        self.event_start_time = Some(frame.timestamp_micros);

        // Instant Trigger for Transients
        if self.consecutive_loud_frames == 1 {
          let label = classify_sound(frame.zcr, frame.spectral_centroid);
          //println!("[DEBUG] ZCR: {:.2} | Centroid: {:.0}Hz -> {}", frame.zcr, frame.spectral_centroid, label);

          let fingerprint = frame.spectrum.clone();
          let angle = frame.direction_of_arrival.map(|(a, _)| a);

          tx.send(SemanticEvent {
            start_timestamp: frame.timestamp_micros,
            end_timestamp: None,
            sources: vec![frame.sensor_id.clone()],
            kind: EventKind::Transient {
              label,
              confidence: 0.8,
              peak_db: to_db(frame.rms_power),
            },
            fingerprint,
            angle,
          }).unwrap();
        }
      }

      // STREAMING LOGIC (Send Audio)
      if let Some(ref t_tx) = self.transcriber_tx {
        let _ = t_tx.send(TranscriberMessage::AudioChunk {
          samples: frame.raw_samples.clone(),
          sample_rate: 44100, 
        });
      }

    } else {
      // --- SILENCE DETECTED ---

      if self.is_tracking_event {
        // We are inside an event, but it just went quiet.

        // 1. Start the timer if not started
        if self.silence_start_time.is_none() {
          self.silence_start_time = Some(frame.timestamp_micros);
        }

        // 2. Check how long it has been silent
        let silence_duration = frame.timestamp_micros - self.silence_start_time.unwrap();

        if silence_duration < (HANGOVER_MS * 1000) {
          // HANGOVER ACTIVE: Keep recording silence so the audio flows naturally.
          if let Some(ref t_tx) = self.transcriber_tx {
            let _ = t_tx.send(TranscriberMessage::AudioChunk {
              samples: frame.raw_samples.clone(),
              sample_rate: 44100, 
            });
          }
        } else {
          // TIMEOUT EXCEEDED: The user actually stopped talking.
          self.finish_event(frame, tx);
        }
      }

      // Slowly drift noise floor up if it's just ambient noise
      self.noise_floor_rms = lerp(self.noise_floor_rms, current_rms, 0.001);
    }

    if self.frame_count % 50 == 0 { // Check every ~500ms
      let briefing = self.generate_atmosphere_briefing();

      // Only send if the description has changed
      if briefing.summary != self.last_briefing_summary {
        println!(">>> CONTEXT SHIFT: {}", briefing.summary);
        self.last_briefing_summary = briefing.summary.clone();

        // Send to Cortex
        tx.send(SemanticEvent {
          start_timestamp: frame.timestamp_micros,
          end_timestamp: None,
          sources: vec!["perception_engine".to_string()],
          kind: EventKind::ContextUpdate(briefing),
          fingerprint: vec![],
          angle: None,
        }).unwrap();
      }
    }    
  }

  fn finish_event(&mut self, frame: &AcousticFrame, tx: &Sender<SemanticEvent>) {
    // 1. Tell Transcriber to process what it just heard
    if let Some(ref t_tx) = self.transcriber_tx {
      let _ = t_tx.send(TranscriberMessage::EndOfSpeech { 
        sensor_id: "binaural_mix".to_string(),
        timestamp: self.event_start_time.unwrap_or(0),
      });
    }

    self.is_tracking_event = false;
    self.consecutive_loud_frames = 0;

    if let Some(start) = self.event_start_time {
      // Optional: Filter out super short glitches (< 100ms) here if desired
      let duration = frame.timestamp_micros - start;
      if duration > 200_000 { // > 200ms
        tx.send(SemanticEvent {
          start_timestamp: start,
          end_timestamp: Some(frame.timestamp_micros),
          sources: vec![frame.sensor_id.clone()],
          kind: EventKind::Continuous {
            label: "Sustained".to_string(),
            is_speech: false,
          },
          fingerprint: frame.spectrum.clone(),
          angle: None,
        }).unwrap();
      }
    }
    self.event_start_time = None;
  }

  fn generate_atmosphere_briefing(&self) -> ContextBriefing {
    // 1. Convert technical RMS to human "Vibe"
    let loudness_desc = if self.noise_floor_rms < 0.001 {
      "dead silent"
    } else if self.noise_floor_rms < 0.01 {
      "quiet (library levels)"
    } else if self.noise_floor_rms < 0.05 {
      "humming with ambient noise"
    } else {
      "loud and chaotic"
    };

    // 2. Determine activity level
    let activity_desc = if self.is_tracking_event {
      "active acoustic events occurring"
    } else {
      "stable background state"
    };

    ContextBriefing {
      domain: "Atmosphere".to_string(),
      summary: format!("The environment is {} with {}.", loudness_desc, activity_desc),
      confidence: 0.9,
      urgency: 1,
      timestamp: time() as u64,
    }
  }
}

// TDOA Math Helper (Generalized Cross Correlation)
fn calculate_doa(samples_a: &[f32], samples_b: &[f32], mic_distance: f32) -> f32 {
  // 1. Find the best lag
  // We slide signal B across Signal A to see where they match best.
  let max_lag = 30; // Max samples to check (depends on mic distance & sample rate)
  let mut best_lag = 0;
  let mut max_corr = 0.0;

  for lag in -(max_lag as i32)..=(max_lag as i32) {
    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..samples_a.len() {
      let j = i as i32 + lag;
      if j >= 0 && j < samples_b.len() as i32 {
        sum += samples_a[i] * samples_b[j as usize];
        count += 1;
      }
    }

    if count > 0 {
      let corr = sum / count as f32;
      if corr > max_corr {
        max_corr = corr;
        best_lag = lag;
      }
    }
  }

  // 2. Convert Lag to Angle
  // Time Delay = Lag / SampleRate
  // Distance = Time * SpeedOfSound
  // sin(theta) = Distance / MicDistance
  let delay_sec = best_lag as f32 / SAMPLE_RATE;
  let dist_diff = delay_sec * SPEED_OF_SOUND;

  // Clamp to valid range for asin (-1.0 to 1.0)
  let ratio = (dist_diff / mic_distance).clamp(-1.0, 1.0);

  // Return angle in radians (approx -1.57 to +1.57)
  ratio.asin() 
}

fn lerp(start: f32, end: f32, amount: f32) -> f32 {
  start + (end - start) * amount
}

fn to_db(rms: f32) -> f32 {
  20.0 * rms.log10()
}

fn classify_sound(zcr: f32, centroid: f32) -> String {
  if centroid > 2500.0 { if zcr < 0.1 { "Whistle/Alarm".to_string() } else { "Click/Clap/Sibilance".to_string() } }
  else if centroid >= 400.0 && centroid <= 2500.0 { "Speech/Vocal".to_string() }
  else if centroid >= 100.0 && centroid < 400.0 { if zcr > 0.02 { "Deep Vocal".to_string() } else { "Thud/Rumble".to_string() } }
  else if centroid < 100.0 { "Low Rumble".to_string() }
  else { "Noise".to_string() }
}

fn qwert() {