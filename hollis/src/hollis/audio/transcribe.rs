use ndata::dataobject::DataObject;
use crossbeam_channel::{Receiver, Sender};
use std::thread;
use std::path::PathBuf;

use crate::hollis::audio::data::SemanticEvent;
use crate::hollis::audio::data::EventKind;

use transcribe_rs::onnx::moonshine::{MoonshineModel, MoonshineVariant, MoonshineParams};
use transcribe_rs::onnx::Quantization;
use crate::hollis::audio::data::Point3D;
pub fn execute(_: DataObject) -> DataObject {
    use std::panic;
    let ax = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        transcribe()
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

pub fn transcribe() -> DataObject {
DataObject::new()
}

pub enum TranscriberMessage {
  AudioChunk { track_id: String, samples: Vec<f32>, sample_rate: u32 },
  EndOfSpeech { track_id: String, timestamp: u64, location: Point3D }, // <--- USE 3D POINT
}
pub struct Transcriber {
  model_path: String,
}

impl Transcriber {
  // We no longer need to load the heavy context here. 
  // We just store the path and load it in the worker thread.
  pub fn new(model_path: String) -> Self {
    Transcriber { model_path }
  }

  pub fn spawn(self, rx: Receiver<TranscriberMessage>, event_tx: Sender<SemanticEvent>) -> thread::JoinHandle<()> {

    thread::spawn(move || {
      println!("Loading Moonshine STT from {}...", self.model_path);
      let mut model = MoonshineModel::load(
        &PathBuf::from(&self.model_path),
        MoonshineVariant::Base, 
        &Quantization::default() 
      ).expect("Failed to load Moonshine model");

      // --- THE MULTI-TRACK MIXER ---
      let mut active_tracks: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
      let options = MoonshineParams::default();

      // 2. The Transcription Loop
      while let Ok(msg) = rx.recv() {
        match msg {
          TranscriberMessage::AudioChunk { track_id, samples, sample_rate } => {
            // Resample to 16kHz if necessary
            let resampled = if sample_rate != 16000 {
              linear_resample(&samples, sample_rate, 16000)
            } else {
              samples
            };

            // Append the audio to the correct spatial track!
            active_tracks.entry(track_id).or_insert_with(Vec::new).extend(resampled);
          },

          TranscriberMessage::EndOfSpeech { track_id, timestamp, location } => { // <--- GRAB IT
            if let Some(audio_buffer) = active_tracks.remove(&track_id) {
              if audio_buffer.len() >= 320 {
                if let Ok(result) = model.transcribe_with(&audio_buffer, &options) {
                  let text = result.text.trim().to_string();
                  if !text.is_empty() && !text.starts_with("[") {
                    let _ = event_tx.send(SemanticEvent {
                      start_timestamp: timestamp,
                      end_timestamp: None,
                      sources: vec![track_id], 
                      kind: EventKind::Transcript { text },
                      fingerprint: vec![],
                      location: Some(location), // <--- HAND TO CORTEX
                    });
                  }
                }
              }
            }
          
          }
        }
      }
    })
  }
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
}
