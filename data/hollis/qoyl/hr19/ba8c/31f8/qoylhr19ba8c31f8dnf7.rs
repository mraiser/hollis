DataObject::new()
}

pub enum TranscriberMessage {
  AudioChunk { 
    samples: Vec<f32>, 
    sample_rate: u32 
  },
  EndOfSpeech { 
    sensor_id: String, 
    timestamp: u64 
  },
}

pub struct Transcriber {
  ctx: Arc<WhisperContext>,
}

impl Transcriber {
  // Constructor loads the model synchronously
  pub fn new(model_path: String) -> Self {
    let ctx = Arc::new(WhisperContext::new_with_params(
      &model_path, 
      whisper_rs::WhisperContextParameters::default()
      ).expect("failed to load whisper model"));
    
    whisper_rs::install_logging_hooks();

    Transcriber { ctx }
  }

  // Spawn takes 'self' to move the loaded context into the thread
  pub fn spawn(self, rx: Receiver<TranscriberMessage>, event_tx: Sender<SemanticEvent>) -> thread::JoinHandle<()> {
    let ctx = self.ctx; // Move Arc into the thread

    thread::spawn(move || {
      let mut audio_buffer: Vec<f32> = Vec::new();

      while let Ok(msg) = rx.recv() {
        match msg {
          TranscriberMessage::AudioChunk { samples, sample_rate } => {
            if sample_rate != 16000 {
              let resampled = linear_resample(&samples, sample_rate, 16000);
              audio_buffer.extend(resampled);
            } else {
              audio_buffer.extend(samples);
            }
          },
          TranscriberMessage::EndOfSpeech { sensor_id, timestamp } => {
            if !audio_buffer.is_empty() {
              //println!("Transcribing {} samples...", audio_buffer.len());
              if let Some(text) = process_audio(&audio_buffer, &ctx) {
                //println!(">>> Transcript: \"{}\"", text);
                if let Err(_) = event_tx.send(SemanticEvent {
                  start_timestamp: timestamp,
                  end_timestamp: None,
                  sources: vec![sensor_id],
                  kind: EventKind::Transcript { text },
                  fingerprint: vec![],
                  angle: None,
                }) {
                  eprintln!("[Transcriber] Warning: Cortex channel closed. Transcript dropped.");
                  break; // Stop the thread if there is nobody listening
                }
              }
              audio_buffer.clear();
            }
          }
        }
      }
    })
  }
}

fn process_audio(samples: &[f32], ctx: &Arc<WhisperContext>) -> Option<String> {
  let mut state = ctx.create_state().ok()?;
  let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

  if samples.len() < 320 { return None; } 
  state.full(params, samples).ok()?;

  let num_segments = state.full_n_segments(); 
  let mut full_text = String::new();

  for i in 0..num_segments {
    if let Some(segment) = state.get_segment(i) {
      full_text.push_str(&segment.to_string()); 
      full_text.push(' ');
    }
  }

  let text = full_text.trim().to_string();
  if text.is_empty() || text == "Thanks for watching!" || text.starts_with("[") {
    return None;
  }
  Some(text)
}

fn linear_resample(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
  if source_rate == target_rate { return input.to_vec(); }

  let ratio = source_rate as f32 / target_rate as f32;
  let new_len = (input.len() as f32 / ratio).ceil() as usize;
  let mut output = Vec::with_capacity(new_len);

  for i in 0..new_len {
    let src_idx_f = i as f32 * ratio;
    let src_idx = src_idx_f as usize;
    if src_idx >= input.len() - 1 { break; }
    let frac = src_idx_f - src_idx as f32;
    output.push(input[src_idx] + (input[src_idx + 1] - input[src_idx]) * frac);
  }
  output
}

fn qwert(){