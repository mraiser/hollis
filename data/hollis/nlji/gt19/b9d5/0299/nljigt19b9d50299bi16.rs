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
  is_tracking_voice: bool,
  active_track_id: Option<String>,
  active_location: Option<Point3D>,
  event_start_time: Option<u64>,
  consecutive_loud_frames: usize,
  frame_count: usize, // Track total lifetime to handle warmup
  transcriber_tx: Option<Sender<TranscriberMessage>>, // Channel to talk to the Transcriber thread
  silence_start_time: Option<u64>, // Track when silence began
  last_briefing_summary: String, // To prevent spam
  stereo_buffer: HashMap<u64, Vec<AcousticFrame>>,
  mic_distance: f32,
  vad_session: Session,
  event_audio_buffer: Vec<f32>,
  mel_extractor: MelExtractor,
}

impl PerceptionEngine {
  pub fn spawn(
    rx: Receiver<SynchronizedArrayFrame>,
    tx: Sender<SemanticEvent>,
    transcriber_tx: Option<Sender<TranscriberMessage>>,
    mic_distance: f32, 
  ) -> thread::JoinHandle<()> {

    let vad_session = Session::builder()
    .expect("Failed to create SessionBuilder")
    .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
    .commit_from_file("models/smart-turn-v3.2-cpu.onnx")
    .expect("Failed to load Smart Turn v3 ONNX model");

    thread::spawn(move || {
      let mut engine = PerceptionEngine {
        noise_floor_rms: 0.0,
        is_tracking_event: false,
        is_tracking_voice: false,
        active_track_id: None,
        active_location: None,
        event_start_time: None,
        consecutive_loud_frames: 0,
        frame_count: 0,
        transcriber_tx,
        silence_start_time: None,
        last_briefing_summary: String::new(),
        stereo_buffer: HashMap::new(),
        mic_distance,
        vad_session,
        event_audio_buffer: Vec::with_capacity(44100 * 5),
        mel_extractor: MelExtractor::new(44100.0, 1024, 441, 80),
      };

      // TAPS: 1024 taps at 44.1kHz covers about 23 milliseconds of acoustic reflection.
      let mut echo_cancellers = vec![NlmsFilter::new(1024, 0.1); 5]; // <--- 5 Mics

      // --- SETUP STATE VARIABLES BEFORE THE LOOP ---
      let mut is_calibrated = false;
      let mut calibration_buffer: Vec<SynchronizedArrayFrame> = Vec::new();
      let mut radar = SpatialRadar::new(44100, mic_distance, 0.30, 0.40); 
      let mut silence_frames = 0;
      let mut beamformer: Option<StreamingBeamformer> = None;
      
      // The Acoustic SLAM Engine
      let mut geometry_mapper = GeometryMapper::new(5, 44100.0);
      let mut hardware_latency_samples = vec![0; 5]; // Store SAMPLES, not meters
      let mut active_mic_weights = vec![1.0; 5];

      println!("NLMS Adaptive Echo Cancellation Online.");
      println!("Waiting for Calibration Chirp...");

      while let Ok(mut sync_frame) = rx.recv() {
        let frame_len = sync_frame.loopback_reference.len();
        let mic_count = sync_frame.mic_channels.len();

        if !is_calibrated {
          let loopback_rms = (sync_frame.loopback_reference.iter().map(|&x| x * x).sum::<f32>() / frame_len as f32).sqrt();
          if loopback_rms > 0.002 || !calibration_buffer.is_empty() {
            calibration_buffer.push(sync_frame.clone());

            if calibration_buffer.len() >= 25 {
              println!("--- Anchoring Acoustic Array ---");
              
              let mut flat_mics = vec![Vec::new(); mic_count];
              let mut flat_loopback = Vec::new(); 
              for frame in &calibration_buffer {
                  flat_loopback.extend_from_slice(&frame.loopback_reference);
                  for m in 0..mic_count { flat_mics[m].extend_from_slice(&frame.mic_channels[m]); }
              }

              let mut raw_delays = vec![0; mic_count];
              let mut corr_scores = vec![0.0; mic_count]; 
              
              for m in 0..mic_count {
                  let (delay, score) = find_hardware_delay(&flat_loopback, &flat_mics[m]);
                  raw_delays[m] = delay;
                  corr_scores[m] = score; 
                  println!("  [DEBUG] Mic {} -> Delay: {:>4} | Corr Score: {:.5}", m, delay, score);
              }
              
              // --- THE STRICT MATH CONFIDENCE GATE ---
              let confidence_threshold = 0.85; 
              let mut current_weights = vec![1.0; mic_count];
              for m in 0..mic_count {
                  if corr_scores[m] < confidence_threshold { 
                      current_weights[m] = 0.0; 
                      println!("  [WEIGHT] Mic {} failed strict calibration (Score: {:.2}). Muting in beamformer.", m, corr_scores[m]);
                  }
              }

              // Dynamically find the anchor: the absolute fastest HEALTHY mic. No hardcoding.
              let fastest_mic = raw_delays.iter().enumerate()
                  .filter(|&(m, _)| current_weights[m] > 0.0)
                  .map(|(_, &d)| d)
                  .min()
                  .unwrap_or_else(|| {
                      println!("  [CRITICAL] All mics failed strict confidence check! Defaulting anchor to 0.");
                      0
                  });
              
              hardware_latency_samples = raw_delays.iter().enumerate().map(|(m, &o)| {
                  if current_weights[m] > 0.0 { if o > fastest_mic { o - fastest_mic } else { 0 } } else { 0 }
              }).collect();

              active_mic_weights = current_weights;
              let delay_meters: Vec<f32> = hardware_latency_samples.iter().map(|&s| (s as f32 / 44100.0) * 343.0).collect();

              beamformer = Some(StreamingBeamformer::new(
                  geometry_mapper.positions.clone(), 
                  &delay_meters, 
                  active_mic_weights.clone(),
                  44100
              ));

              is_calibrated = true;
              calibration_buffer.clear(); 
              println!("Array Anchored. Radar Auto-Expansion Active.");
            }
          }
          continue; 
        }
        // --- END CALIBRATION ROUTINE ---

        // Scrub the frame sample-by-sample
        for i in 0..frame_len {
          let ref_sample = sync_frame.loopback_reference[i];

          for m in 0..mic_count {
            let raw_mic = sync_frame.mic_channels[m][i];
            let clean_mic = echo_cancellers[m].process(raw_mic, ref_sample);
            sync_frame.mic_channels[m][i] = clean_mic;
          }
        }

        // 1. Get the directional unit vector from the radar, using the USB latency offsets!
        let (dx, dy, dz) = radar.locate(&sync_frame.mic_channels, &hardware_latency_samples);
        
        // 2. Find the raw volume of the loudest microphone to estimate distance
        let mut max_raw_rms = 0.0;
        for channel in &sync_frame.mic_channels {
            let sum_sq: f32 = channel.iter().map(|&val| val * val).sum();
            let channel_rms = (sum_sq / channel.len() as f32).sqrt();
            if channel_rms > max_raw_rms { max_raw_rms = channel_rms; }
        }

        // --- AUTONOMOUS ACOUSTIC SLAM ---
        if is_calibrated && max_raw_rms > 0.005 {
            // Pass the raw samples, the USB latency offsets, and the mic weights!
            let did_expand = geometry_mapper.observe(&sync_frame.mic_channels, &hardware_latency_samples, &active_mic_weights);
            
            if did_expand {
                println!("  [SLAM UPDATE] Array geometry expanded! Re-calibrating spatial lens...");
                for (i, pos) in geometry_mapper.positions.iter().enumerate() {
                    // Only print healthy mics
                    if active_mic_weights[i] > 0.0 {
                        println!("    Mic {}: [X: {:.3}m, Y: {:.3}m, Z: {:.3}m]", i, pos.x, pos.y, pos.z);
                    }
                }
                
                let delay_meters: Vec<f32> = hardware_latency_samples.iter().map(|&s| (s as f32 / 44100.0) * 343.0).collect();
                beamformer = Some(StreamingBeamformer::new(
                    geometry_mapper.positions.clone(), 
                    &delay_meters, 
                    active_mic_weights.clone(),
                    44100
                ));
            }
        }
        // --- END SLAM ---

        // 3. Inverse Square Law Heuristic (Amplitude drops with distance)
        // A standard conversational voice at 1 meter produces an RMS of roughly 0.015 in this array.
        let ref_rms_at_1m = 0.015;
        
        // Calculate estimated depth (radius). 
        let mut estimated_distance = if max_raw_rms > 0.0001 {
            ref_rms_at_1m / max_raw_rms
        } else {
            2.0 // Fallback for near-silence
        };

        // Clamp the distance so it doesn't hallucinate sounds 50 meters away (too quiet) 
        // or inside the microphone (too loud). 
        estimated_distance = estimated_distance.clamp(0.4, 6.0);

        // 4. Project the directional vector out to the dynamically estimated distance
        let target_loc = Point3D::new(
            dx * estimated_distance, 
            dy * estimated_distance, 
            dz * estimated_distance
        );

        // 2. Only process if the Beamformer is online
        if let Some(ref mut bf) = beamformer {
          let focused_audio = bf.process(&sync_frame.mic_channels, target_loc);

          let sum_squares: f32 = focused_audio.iter().map(|&val| val * val).sum();
          let rms = (sum_squares / focused_audio.len() as f32).sqrt();

          // UNCOMMENT THIS LINE if it's still deaf, so we can see your volume levels!
          // if rms > 0.001 { println!("Beamformer RMS: {:.4}", rms); }

          // --- THE UNIFIED 3D STREAMING PIPELINE ---
          if rms > 0.002 {
            engine.event_audio_buffer.extend_from_slice(&focused_audio);
            silence_frames = 0;

            if !engine.is_tracking_event {
              engine.is_tracking_event = true;
              engine.is_tracking_voice = false; // Reset the voice flag
              engine.event_start_time = Some(sync_frame.timestamp_micros);
              
              // 1. Lock the TRUE 3D Coordinate
              engine.active_track_id = Some(format!("loc_{:.1}_{:.1}", target_loc.x, target_loc.y));
              engine.active_location = Some(target_loc);

              // 2. Classify the initial sound transient
              let mut zero_crossings = 0;
              for i in 1..focused_audio.len() {
                  if (focused_audio[i-1] >= 0.0 && focused_audio[i] < 0.0) || (focused_audio[i-1] < 0.0 && focused_audio[i] >= 0.0) { zero_crossings += 1; }
              }
              let zcr = zero_crossings as f32 / focused_audio.len() as f32;
              let label = if zcr > 0.10 { "Click/Clap/Sibilance" } else if zcr > 0.02 { "Speech/Vocal" } else { "Thud/Rumble" };

              // Generate the fingerprint right before sending the event
              let (_, fingerprint) = engine.mel_extractor.extract(&focused_audio, 800);

              let _ = tx.send(SemanticEvent {
                start_timestamp: sync_frame.timestamp_micros,
                end_timestamp: None,
                sources: vec!["Radar Array".to_string()],
                kind: EventKind::Transient { label: label.to_string(), confidence: 0.8, peak_db: to_db(rms) },
                fingerprint, // <--- NO LONGER EMPTY!
                location: Some(target_loc),
              });
            }

            // --- 3. THE VAD GATE ---
            // If we aren't tracking a voice yet, check if we have enough data to make a decision
            if !engine.is_tracking_voice && engine.event_audio_buffer.len() > (44100 / 4) { // ~250ms backlog
                engine.is_tracking_voice = engine.evaluate_turn_completion();
                
                // If it JUST triggered as a voice, flush the entire backlog to the transcriber so we don't chop the first word
                if engine.is_tracking_voice {
                    if let Some(ref t_tx) = engine.transcriber_tx {
                        if let Some(ref track_id) = engine.active_track_id {
                            let _ = t_tx.send(crate::hollis::audio::transcribe::TranscriberMessage::AudioChunk {
                                track_id: track_id.clone(),
                                samples: engine.event_audio_buffer.clone(), 
                                sample_rate: 44100, 
                            });
                        }
                    }
                }
            } else if engine.is_tracking_voice {
                // We already confirmed it's a voice; stream the live chunk directly
                if let Some(ref t_tx) = engine.transcriber_tx {
                    if let Some(ref track_id) = engine.active_track_id {
                        let _ = t_tx.send(crate::hollis::audio::transcribe::TranscriberMessage::AudioChunk {
                            track_id: track_id.clone(),
                            samples: focused_audio.clone(), 
                            sample_rate: 44100, 
                        });
                    }
                }
            }

          } else if engine.is_tracking_event {
            // --- THE HANGOVER ---
            silence_frames += 1;
            engine.event_audio_buffer.extend_from_slice(&focused_audio);

            // If it IS a voice, we MUST keep sending the hangover frames to STT.
            // Otherwise, we accidentally chop off the final trailing syllables of a sentence!
            if engine.is_tracking_voice {
                if let Some(ref t_tx) = engine.transcriber_tx {
                    if let Some(ref track_id) = engine.active_track_id {
                        let _ = t_tx.send(crate::hollis::audio::transcribe::TranscriberMessage::AudioChunk {
                            track_id: track_id.clone(),
                            samples: focused_audio.clone(), 
                            sample_rate: 44100, 
                        });
                    }
                }
            }

            if silence_frames > 45 { 
                // Only send EndOfSpeech to Moonshine if it was actually a voice
                if engine.is_tracking_voice {
                    if let (Some(track_id), Some(loc)) = (&engine.active_track_id, &engine.active_location) {
                        if let Some(ref t_tx) = engine.transcriber_tx {
                            let _ = t_tx.send(crate::hollis::audio::transcribe::TranscriberMessage::EndOfSpeech {
                                track_id: track_id.clone(), 
                                timestamp: engine.event_start_time.unwrap_or(sync_frame.timestamp_micros), 
                                location: *loc,
                            });
                        }
                        
                        // --- DUMP THE FINAL BEAMFORMED AUDIO TO DISK ---
                        save_debug_audio(&engine.event_audio_buffer, track_id);
                    }
                }

                // Clean up state
                engine.event_audio_buffer.clear();
                engine.is_tracking_event = false;
                engine.is_tracking_voice = false; 
                engine.active_track_id = None;
                engine.active_location = None;
            }
          }
        }
      }
    })
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

  fn evaluate_turn_completion(&mut self) -> bool {
    if self.event_audio_buffer.is_empty() { return false; }

    // 1. Generate the [80 x 800] tensor flat array (max frames = 800)
    let (tensor_data, _) = self.mel_extractor.extract(&self.event_audio_buffer, 800);

    // 2. Feed it to the ONNX model
    let shape = [1_usize, 80_usize, 800_usize];
    let tensor = ort::value::TensorRef::from_array_view((shape, tensor_data.as_slice()))
        .expect("Failed to build tensor");

    let input_name = self.vad_session.inputs()[0].name().to_string();
    let inputs = ort::inputs![input_name => tensor];

    match self.vad_session.run(inputs) {
      Ok(outputs) => {
        if let Ok((_shape, data)) = outputs[0].try_extract_tensor::<f32>() {
          return data[0] > 0.5; // Prob > 50% means Speech
        }
      },
      Err(e) => eprintln!("VAD Inference Error: {}", e),
    }

    false // Fail closed on error
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

pub struct SpatialRadar {
  sample_rate: f32,
  speed_of_sound: f32,
  mic_distance_x: f32, 
  mic_distance_y: f32, 
  mic_distance_z: f32, 
}

impl SpatialRadar {
  pub fn new(sample_rate: u32, mic_distance_x: f32, mic_distance_y: f32, mic_distance_z: f32) -> Self {
    Self {
      sample_rate: sample_rate as f32,
      speed_of_sound: 343.0,
      mic_distance_x,
      mic_distance_y,
      mic_distance_z,
    }
  }

  pub fn locate(&mut self, channels: &[Vec<f32>], hw_delays: &[usize]) -> (f32, f32, f32) { 
    if channels.len() < 5 { return (0.0, 0.0, 0.0); }

    // STRICT ALIGNMENT: Match the SLAM universe axes perfectly
    let x_lag = self.cross_correlate(&channels[0], &channels[1], hw_delays[0], hw_delays[1]); 
    let y_lag = self.cross_correlate(&channels[0], &channels[2], hw_delays[0], hw_delays[2]); 
    let z_lag = self.cross_correlate(&channels[0], &channels[4], hw_delays[0], hw_delays[4]); 

    let (x_pos, new_x) = self.lag_to_position(x_lag, self.mic_distance_x);
    self.mic_distance_x = new_x;
    let (y_pos, new_y) = self.lag_to_position(y_lag, self.mic_distance_y);
    self.mic_distance_y = new_y;
    let (z_pos, new_z) = self.lag_to_position(z_lag, self.mic_distance_z);
    self.mic_distance_z = new_z;

    (x_pos, y_pos, z_pos)
  }

  fn cross_correlate(&self, sig_a: &[f32], sig_b: &[f32], delay_a: usize, delay_b: usize) -> i32 {
    let max_acoustic_search = 60; 
    let usb_offset = delay_b as i32 - delay_a as i32; // <--- Target the center of the USB gap!
    
    let mut best_acoustic_lag = 0;
    let mut max_corr = -1.0;

    let mean_a = sig_a.iter().sum::<f32>() / sig_a.len() as f32;
    let mean_b = sig_b.iter().sum::<f32>() / sig_b.len() as f32;
    let var_a = sig_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f32>();
    let var_b = sig_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f32>();

    if var_a == 0.0 || var_b == 0.0 { return 0; }
    let denominator = (var_a * var_b).sqrt();

    for lag in (usb_offset - max_acoustic_search)..=(usb_offset + max_acoustic_search) {
      let mut sum = 0.0;
      for i in 0..sig_a.len() {
        let j = i as i32 + lag;
        if j >= 0 && j < sig_b.len() as i32 { sum += (sig_a[i] - mean_a) * (sig_b[j as usize] - mean_b); }
      }
      let corr = sum / denominator;
      if corr > max_corr { 
          max_corr = corr; 
          best_acoustic_lag = lag - usb_offset; // Extract the true directional audio lag
      }
    }
    best_acoustic_lag
  }

  fn lag_to_position(&self, lag_samples: i32, current_bound: f32) -> (f32, f32) {
    let delay_sec = lag_samples.abs() as f32 / self.sample_rate;
    let acoustic_distance = delay_sec * self.speed_of_sound;
    
    // THE AUTO-EXPANDER: Calculate the stretched boundary
    let new_bound = if acoustic_distance > current_bound {
        acoustic_distance + 0.05 // Stretch with a 5cm buffer
    } else {
        current_bound
    };
    
    let ratio = (acoustic_distance / new_bound).clamp(-1.0, 1.0);
    let final_ratio = if lag_samples < 0 { -ratio } else { ratio };
    
    (final_ratio, new_bound)
  }
}

/// Normalized Least Mean Squares (NLMS) Echo Canceller
#[derive(Clone)]
pub struct NlmsFilter {
  weights: Vec<f32>,
  history: Vec<f32>,
  head: usize,
  taps: usize,
  mu: f32,
}

impl NlmsFilter {
  pub fn new(taps: usize, mu: f32) -> Self {
    Self {
      weights: vec![0.0; taps],
      history: vec![0.0; taps],
      head: 0,
      taps,
      mu,
    }
  }

  /// Feeds one sample of the mic and one sample of the speaker into the filter.
  /// Returns the cleaned, echo-free microphone sample.
  pub fn process(&mut self, mic_in: f32, ref_in: f32) -> f32 {
    // 1. Add the new speaker sample to our rolling history
    self.history[self.head] = ref_in;

    // 2. Calculate the estimated echo based on our current learned weights
    let mut echo_est = 0.0;
    let mut ref_power = 0.0001; // Small epsilon to prevent division by zero

    for i in 0..self.taps {
      let idx = (self.head + self.taps - i) % self.taps;
      let x = self.history[idx];
      echo_est += self.weights[i] * x;
      ref_power += x * x;
    }

    // 3. Subtract the echo from the actual microphone signal
    let clean_signal = mic_in - echo_est;

    // 4. Learn: Update the weights so we are more accurate next time
    let norm_mu = self.mu / ref_power;
    for i in 0..self.taps {
      let idx = (self.head + self.taps - i) % self.taps;
      self.weights[i] += norm_mu * clean_signal * self.history[idx];
    }

    // Move the ring buffer head forward
    self.head = (self.head + 1) % self.taps;

    clean_signal
  }
}

fn find_hardware_delay(loopback: &[f32], mic: &[f32]) -> (usize, f32) {
    let window_size = 1024;
    
    // 1. Find where the chirp ACTUALLY starts in the loopback channel
    let mut best_loopback_start = 0;
    let mut max_loopback_energy = 0.0;
    
    for i in 0..(loopback.len() - window_size) {
        let mut energy = 0.0;
        for j in 0..window_size {
            energy += loopback[i + j] * loopback[i + j]; // True energy (squared)
        }
        if energy > max_loopback_energy {
            max_loopback_energy = energy;
            best_loopback_start = i;
        }
    }
    
    let chirp_signature = &loopback[best_loopback_start..(best_loopback_start + window_size)];
    let sig_rms = max_loopback_energy.sqrt();

    // 2. Normalized Cross-Correlation
    let mut max_corr = 0.0;
    let mut best_delay = 0;

    for delay in 0..(mic.len() - window_size) {
        let mut sum = 0.0;
        let mut mic_energy = 0.0;
        
        for i in 0..window_size {
            let m = mic[delay + i];
            sum += chirp_signature[i] * m;
            mic_energy += m * m;
        }

        let mic_rms = mic_energy.sqrt();
        
        // Pearson-style normalization (returns 0.0 to 1.0)
        let norm_corr = if sig_rms > 0.0 && mic_rms > 0.0 {
            (sum / (sig_rms * mic_rms)).abs() 
        } else {
            0.0
        };

        if norm_corr > max_corr {
            max_corr = norm_corr;
            best_delay = delay;
        }
    }
    
    let true_delay = if best_delay > best_loopback_start {
        best_delay - best_loopback_start
    } else {
        0 
    };
    
    (true_delay, max_corr)
}

pub struct StreamingBeamformer {
  mic_positions: Vec<Point3D>,
  hardware_delays_samples: Vec<usize>, 
  mic_weights: Vec<f32>,
  sample_rate: f32,
  speed_of_sound: f32,
  history_buffers: Vec<Vec<f32>>,
  max_possible_delay: usize,
}

impl StreamingBeamformer {
  pub fn new(mic_positions: Vec<Point3D>, hardware_latency_meters: &[f32], mic_weights: Vec<f32>, sample_rate: u32) -> Self {
    let mut max_dist = 0.0;
    for m1 in &mic_positions {
      for m2 in &mic_positions {
        let d = m1.distance(m2);
        if d > max_dist { max_dist = d; }
      }
    }

    let max_hw_latency = hardware_latency_meters.iter().cloned().fold(0.0, f32::max);
    let hardware_delays_samples: Vec<usize> = hardware_latency_meters.iter()
        .map(|&m| ((max_hw_latency - m) / 343.0 * sample_rate as f32).round() as usize).collect();

    let max_delay_sec = ((max_dist + max_hw_latency) / 343.0) * 1.2; 
    let max_possible_delay = (max_delay_sec * sample_rate as f32).ceil() as usize;

    Self {
      mic_positions: mic_positions.clone(), 
      hardware_delays_samples,
      mic_weights,
      sample_rate: sample_rate as f32,
      speed_of_sound: 343.0,
      history_buffers: vec![vec![0.0; max_possible_delay]; mic_positions.len()], 
      max_possible_delay,
    }
  }

  pub fn process(&mut self, channels: &[Vec<f32>], target: Point3D) -> Vec<f32> {
    if channels.is_empty() || channels.len() != self.mic_positions.len() { return vec![]; }

    let frame_len = channels[0].len();
    let mut output = vec![0.0; frame_len];
    
    let mut active_mic_count = 0.0;
    for &w in &self.mic_weights { active_mic_count += w; }
    if active_mic_count == 0.0 { active_mic_count = 1.0; } 

    let mut distances = Vec::with_capacity(self.mic_positions.len());
    let mut max_distance = 0.0;
    for mic in &self.mic_positions {
      let dist = target.distance(mic);
      distances.push(dist);
      if dist > max_distance { max_distance = dist; }
    }

    let delays: Vec<usize> = distances.iter().enumerate().map(|(mic_idx, &d)| {
      let acoustic_delay_samples = (((max_distance - d) / self.speed_of_sound) * self.sample_rate).round() as usize;
      acoustic_delay_samples + self.hardware_delays_samples[mic_idx]
    }).collect();

    for i in 0..frame_len {
      let mut sum = 0.0;
      for (mic_idx, channel) in channels.iter().enumerate() {
        let weight = self.mic_weights[mic_idx];
        if weight == 0.0 { continue; } // Completely ignore poisoned mics

        let delay = delays[mic_idx];
        let sample = if i < delay {
          self.history_buffers[mic_idx][self.max_possible_delay - (delay - i)]
        } else {
          channel[i - delay]
        };
        sum += sample * weight;
      }
      output[i] = sum / active_mic_count;
    }

    for (mic_idx, channel) in channels.iter().enumerate() {
      if frame_len >= self.max_possible_delay {
        let tail_start = frame_len - self.max_possible_delay;
        self.history_buffers[mic_idx].copy_from_slice(&channel[tail_start..]);
      } else {
        self.history_buffers[mic_idx].rotate_left(frame_len);
        let start_idx = self.max_possible_delay - frame_len;
        self.history_buffers[mic_idx][start_idx..].copy_from_slice(channel);
      }
    }

    // --- DIGITAL GAIN STAGE ---
    let gain_multiplier = 6.0; 
    for val in output.iter_mut() {
        *val = (*val * gain_multiplier).clamp(-1.0, 1.0);
    }

    output
  }
}

pub struct MelExtractor {
    fft_size: usize,
    hop_size: usize,
    mel_bins: usize,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
    planner: FftPlanner<f32>,
}

impl MelExtractor {
    pub fn new(sample_rate: f32, fft_size: usize, hop_size: usize, mel_bins: usize) -> Self {
        // 1. Create Hann Window
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size as f32 - 1.0)).cos()))
            .collect();

        // 2. Build Mel Filterbank
        let min_mel = Self::hz_to_mel(0.0);
        let max_mel = Self::hz_to_mel(sample_rate / 2.0);
        let mel_step = (max_mel - min_mel) / (mel_bins as f32 + 1.0);

        let mut mel_points = Vec::new();
        for i in 0..=(mel_bins + 1) {
            mel_points.push(Self::mel_to_hz(min_mel + i as f32 * mel_step));
        }

        let hz_per_bin = sample_rate / fft_size as f32;
        let mut filterbank = vec![vec![0.0; fft_size / 2 + 1]; mel_bins];

        for m in 0..mel_bins {
            let left = mel_points[m];
            let center = mel_points[m + 1];
            let right = mel_points[m + 2];

            for k in 0..=(fft_size / 2) {
                let freq = k as f32 * hz_per_bin;
                if freq > left && freq < right {
                    if freq <= center {
                        filterbank[m][k] = (freq - left) / (center - left);
                    } else {
                        filterbank[m][k] = (right - freq) / (right - center);
                    }
                }
            }
        }

        Self {
            fft_size, hop_size, mel_bins, window, filterbank,
            planner: FftPlanner::new(),
        }
    }

    fn hz_to_mel(hz: f32) -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() }
    fn mel_to_hz(mel: f32) -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) }

    /// Returns (Tensor [80 x 800], Biometric Fingerprint [80])
    pub fn extract(&mut self, audio: &[f32], max_frames: usize) -> (Vec<f32>, Vec<f32>) {
        let fft = self.planner.plan_fft_forward(self.fft_size);
        let num_frames = if audio.len() < self.fft_size { 0 } else { (audio.len() - self.fft_size) / self.hop_size + 1 };
        
        // The ONNX tensor format: [mel_bins, time_frames]
        let mut tensor_flat = vec![0.0; self.mel_bins * max_frames];
        // The Biometric average
        let mut fingerprint = vec![0.0; self.mel_bins];
        let mut valid_frames = 0;

        for frame_idx in 0..num_frames {
            if frame_idx >= max_frames { break; }
            let start = frame_idx * self.hop_size;
            
            // Apply Window & FFT
            let mut complex_buffer: Vec<Complex<f32>> = (0..self.fft_size).map(|i| {
                Complex { re: audio[start + i] * self.window[i], im: 0.0 }
            }).collect();
            fft.process(&mut complex_buffer);

            // Compute Power Spectrum
            let power_spec: Vec<f32> = complex_buffer.iter().take(self.fft_size / 2 + 1)
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();

            // Apply Filters and Log
            for m in 0..self.mel_bins {
                let mut mel_energy = 0.0;
                for k in 0..=(self.fft_size / 2) {
                    mel_energy += power_spec[k] * self.filterbank[m][k];
                }
                
                let log_mel = (mel_energy + 1e-9).log10();
                
                // Write to Tensor: [m][frame_idx] -> m * max_frames + frame_idx
                tensor_flat[m * max_frames + frame_idx] = log_mel;
                fingerprint[m] += log_mel;
            }
            valid_frames += 1;
        }

        // Average the fingerprint for the Cortex
        if valid_frames > 0 {
            for m in 0..self.mel_bins { fingerprint[m] /= valid_frames as f32; }
        }

        (tensor_flat, fingerprint)
    }
}

pub struct GeometryMapper {
    mic_count: usize,
    sample_rate: f32,
    speed_of_sound: f32,
    max_lags: Vec<Vec<usize>>,
    pub positions: Vec<Point3D>,
    pub is_locked: bool,
    updates_without_expansion: usize,
}

impl GeometryMapper {
    pub fn new(mic_count: usize, sample_rate: f32) -> Self {
        let mut positions = vec![Point3D::new(0.0, 0.0, 0.0); mic_count];
        
        // The "Big Bang" Seed to give the force solver a directional vector
        for i in 0..mic_count {
            positions[i] = Point3D::new((i as f32 * 0.001), ((i % 2) as f32 * 0.001), ((i % 3) as f32 * 0.001));
        }

        Self {
            mic_count,
            sample_rate,
            speed_of_sound: 343.0,
            max_lags: vec![vec![0; mic_count]; mic_count],
            positions, 
            is_locked: false,
            updates_without_expansion: 0,
        }
    }

    /// Feeds a loud frame into the matrix, completely bypassing quarantined mics
    pub fn observe(&mut self, channels: &[Vec<f32>], hw_delays: &[usize], weights: &[f32]) -> bool {
        if self.is_locked { return false; } 

        let mut geometry_expanded = false;

        for i in 0..self.mic_count {
            for j in (i + 1)..self.mic_count {
                // Ignore "poison" mics that failed calibration
                if weights[i] == 0.0 || weights[j] == 0.0 { continue; }

                let lag = self.cross_correlate_pair(&channels[i], &channels[j], hw_delays[i], hw_delays[j]);
                
                if lag > self.max_lags[i][j] {
                    self.max_lags[i][j] = lag;
                    self.max_lags[j][i] = lag; 
                    geometry_expanded = true;
                }
            }
        }

        if geometry_expanded {
            self.solve_spring_matrix();
            self.updates_without_expansion = 0; 
        } else {
            self.updates_without_expansion += 1;
            if self.updates_without_expansion > 10 {
                self.is_locked = true;
                println!("  [SLAM LOCKED] Array geometry permanently anchored for this session.");
            }
        }
        
        geometry_expanded
    }

    fn cross_correlate_pair(&self, sig_a: &[f32], sig_b: &[f32], delay_a: usize, delay_b: usize) -> usize {
        let max_acoustic_search = 150; // Search up to ~1.1 physical meters
        
        // THE FIX: Shift the search window by the known USB latency gap!
        let usb_offset = delay_b as i32 - delay_a as i32;

        let mut best_acoustic_lag = 0;
        let mut max_corr = 0.0;

        // Center the cross-correlation search directly over the USB gap
        for lag in (usb_offset - max_acoustic_search)..=(usb_offset + max_acoustic_search) {
            let mut sum = 0.0;
            for i in 0..sig_a.len() {
                let j = i as i32 + lag;
                if j >= 0 && j < sig_b.len() as i32 {
                    sum += sig_a[i] * sig_b[j as usize];
                }
            }
            if sum.abs() > max_corr {
                max_corr = sum.abs();
                // The true acoustic distance is the total lag minus the USB lag
                best_acoustic_lag = (lag - usb_offset).abs() as usize;
            }
        }
        best_acoustic_lag
    }

    fn solve_spring_matrix(&mut self) {
        let learning_rate = 0.05;
        let iterations = 100;

        for _ in 0..iterations {
            let mut forces = vec![Point3D::new(0.0, 0.0, 0.0); self.mic_count];

            for i in 0..self.mic_count {
                for j in (i + 1)..self.mic_count {
                    let target_dist = (self.max_lags[i][j] as f32 / self.sample_rate) * self.speed_of_sound;
                    
                    let p1 = &self.positions[i];
                    let p2 = &self.positions[j];
                    let mut current_dist = p1.distance(p2);
                    if current_dist < 0.001 { current_dist = 0.001; } 

                    let error = current_dist - target_dist;
                    let push_force = error * learning_rate;

                    let dx = (p1.x - p2.x) / current_dist;
                    let dy = (p1.y - p2.y) / current_dist;
                    let dz = (p1.z - p2.z) / current_dist;

                    forces[i].x -= dx * push_force; forces[i].y -= dy * push_force; forces[i].z -= dz * push_force;
                    forces[j].x += dx * push_force; forces[j].y += dy * push_force; forces[j].z += dz * push_force;
                }
            }

            for i in 0..self.mic_count {
                self.positions[i].x += forces[i].x; self.positions[i].y += forces[i].y; self.positions[i].z += forces[i].z;
            }

            // THE TETRAHEDRON ANCHORS
            self.positions[0] = Point3D::new(0.0, 0.0, 0.0); // Origin
            self.positions[1].y = 0.0; self.positions[1].z = 0.0; // X-Axis
            self.positions[2].z = 0.0; // XY Desk Plane
        }
    }
}

fn save_debug_audio(samples: &[f32], track_id: &str) {
    use std::fs::File;
    use std::io::Write;
    
    // Sanitize the track_id so it makes a valid filename
    let safe_id = track_id.replace(|c: char| !c.is_alphanumeric(), "_");
    let filename = format!("debug_{}.wav", safe_id);
    
    if let Ok(mut file) = File::create(&filename) {
        let sample_rate: u32 = 44100;
        let channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate * channels as u32 * (bits_per_sample / 8) as u32;
        let block_align = channels * (bits_per_sample / 8);
        let data_size = (samples.len() * 2) as u32;
        let chunk_size = 36 + data_size;

        // Write the standard WAV header
        let _ = file.write_all(b"RIFF");
        let _ = file.write_all(&chunk_size.to_le_bytes());
        let _ = file.write_all(b"WAVE");
        let _ = file.write_all(b"fmt ");
        let _ = file.write_all(&16u32.to_le_bytes()); // Subchunk1Size
        let _ = file.write_all(&1u16.to_le_bytes());  // AudioFormat (1 = PCM)
        let _ = file.write_all(&channels.to_le_bytes());
        let _ = file.write_all(&sample_rate.to_le_bytes());
        let _ = file.write_all(&byte_rate.to_le_bytes());
        let _ = file.write_all(&block_align.to_le_bytes());
        let _ = file.write_all(&bits_per_sample.to_le_bytes());
        let _ = file.write_all(b"data");
        let _ = file.write_all(&data_size.to_le_bytes());

        // Convert f32 to i16 and write the audio
        for &sample in samples {
            let pcm_sample = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            let _ = file.write_all(&pcm_sample.to_le_bytes());
        }
        println!("  [AUDIO SAVED] Wrote clear audio track to {}", filename);
    }
}

fn qwert() {