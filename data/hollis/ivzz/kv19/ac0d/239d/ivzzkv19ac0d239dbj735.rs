cortex()
/*
let mut top_node = CognitiveNode::new(DataObject::new());


let mut raw_node = DataObject::new();
let mut audio_sensor = CognitiveNode::new(raw_node.clone());
let input_buffer = raw_node.get_array("input_buffer");
//let output_buffer = raw_node.get_array("output_buffer");



start_listening(&mut audio_sensor);


while raw_node.get_boolean("listening") {
  if input_buffer.len() > 0 {
    audio_sensor.pulse();
    
    
    
    raw_node.put_boolean("listening", false);
  }
  else {
    thread::sleep(Duration::from_millis(100));
  }
}
println!("Stopping listening process...");


DataObject::new()
*/
}

pub fn start_listening(audio_sensor: &mut CognitiveNode){
  let mut raw_node = audio_sensor.state.clone();
  raw_node.put_boolean("listening", true);
  
  let mut listen_buffer = DataArray::new();
  
  let mut transcribe_buffer = listen_buffer.clone();
  let mut transcribe_node = raw_node.clone();
  thread::spawn(move ||{
    let ctx = WhisperContext::new_with_params(
      "models/ggml-medium.en.bin", 
      whisper_rs::WhisperContextParameters::default()
    ).expect("Failed to load Whisper Model");
    let shared_model = Arc::new(ctx);

    while transcribe_node.get_boolean("listening") {
      if transcribe_buffer.len() > 0 {
        let mut event = transcribe_buffer.get_object(0);
        let chunk = event.get_bytes("bytes");
        transcribe_buffer.remove_property(0);
        if let Some(text) = process_audio_chunk(chunk, shared_model.clone()) {
          event.put_string("content", &text);
          transcribe_node.get_array("input_buffer").push_object(event);
        }
      }
      else {
        thread::sleep(Duration::from_millis(100));
      }
    }
    
    println!("Stopping transcription process...");
  });  
  
  thread::spawn(move ||{
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");

    let config = cpal::StreamConfig {
      channels: 1,
      sample_rate: cpal::SampleRate(16000),
      buffer_size: cpal::BufferSize::Default, 
    };
    
    let mut is_speaking = false;
    let mut speech_start_time = Instant::now();
    let mut last_speech_time = Instant::now();
    let mut current_buffer = DataBytes::new();
    let err_fn = move |err| eprintln!("an error occurred on stream: {}", err);
    let stream = device.build_input_stream(
      &config,
      move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let rms = calculate_rms(data);

        // VAD Logic
        if rms > 0.01 {
          last_speech_time = Instant::now();
          if !is_speaking {
            is_speaking = true;
            speech_start_time = Instant::now();
            println!("Speech detected (RMS: {:.4})", rms);
            // Start fresh buffer
            current_buffer = DataBytes::new();
          }
        }

        // Silence Timeout (0.8s)
        if is_speaking && last_speech_time.elapsed().as_millis() > 800 {
          is_speaking = false;
          let duration = speech_start_time.elapsed().as_secs_f32() - 0.8;

          if duration > 0.5 {
            println!("Speech segment ended ({:.2}s). Processing...", duration);
            let mut event = DataObject::new();
            event.put_string("source", "audio");
            event.put_string("type", "speech_detected");
            event.put_bytes("bytes", current_buffer.clone());
            listen_buffer.push_object(event);
          }

          // Prepare new buffer for next utterance
          current_buffer = DataBytes::new();
        }

        // Record Audio
        if is_speaking {
          let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
              data.as_ptr() as *const u8,
              data.len() * 4 
            )
          };
          current_buffer.write(bytes);
        }
      },
      err_fn,
      None 
    ).expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");
    
    while raw_node.get_boolean("listening") {
      thread::sleep(Duration::from_millis(100));
    }
    
    println!("Stopping listening process...");
  });  
}

fn calculate_rms(samples: &[f32]) -> f32 {
  let sum: f32 = samples.iter().map(|x| x * x).sum();
  (sum / samples.len() as f32).sqrt()
}

fn process_audio_chunk(audio_bytes: DataBytes, ctx: Arc<WhisperContext>) -> Option<String> {
  // 1. Reconstruct Floating Point Audio from Bytes
  let len_bytes = audio_bytes.current_len(); 
  if len_bytes == 0 { return None; }

  let raw_audio = audio_bytes.get_data(); 
  let samples: Vec<f32> = raw_audio
  .chunks_exact(4)
  .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
  .collect();

  // 2. Load Model
  let mut state_ctx = ctx.create_state().expect("Failed to create state");

  let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
  params.set_language(Some("en"));
  params.set_print_special(false);
  params.set_print_progress(false); 
  params.set_print_realtime(false);
  params.set_print_timestamps(false);

  if let Err(e) = state_ctx.full(params, &samples) {
    eprintln!("Whisper inference failed: {}", e);
    return None;
  }

  // 3. Extract Text
  let num_segments = state_ctx.full_n_segments(); 
  let mut full_text = String::new();

  for i in 0..num_segments {
    if let Some(segment) = state_ctx.get_segment(i) {
      full_text.push_str(&segment.to_string());
      full_text.push(' ');
    }
  }

  let text = full_text.trim().to_string();

  if !text.is_empty() && text != "Thanks for watching!" {
    println!("Native Transcription: {}", text);
    return Some(text);
  }
  None
}

/// The CognitiveNode is the fundamental atomic unit of Hollis.
/// It wraps a DataObject (the state) and provides the metabolic 'pulse' logic.
///
/// State Structure (DataObject):
/// {
///   "id": "node_name",
///   "level": 1,
///   "input_buffer": [ ... ],       // Signals coming from below (or sensory I/O)
///   "output_upstream": [ ... ],    // Insights pushing up to the parent
///   "directives_buffer": [ ... ],  // Commands coming from above
///   "output_downstream": [ ... ],  // Commands pushing down to the child
///   "memory": { ... },             // Local context/vector store
///   "meta": { ... }                // Configuration (prompts, thresholds)
/// }
#[derive(Debug)]
pub struct CognitiveNode {
    pub state: DataObject,
}

impl CognitiveNode {
    /// Connects to an existing state object or initializes a fresh one.
    pub fn new(mut state: DataObject) -> Self {
        // Ensure the skeleton structure exists
        if !state.has("id") { state.put_string("id", &unique_session_id()); }
        if !state.has("input_buffer") { state.put_array("input_buffer", DataArray::new()); }
        if !state.has("output_upstream") { state.put_array("output_upstream", DataArray::new()); }
        if !state.has("directives_buffer") { state.put_array("directives_buffer", DataArray::new()); }
        if !state.has("output_downstream") { state.put_array("output_downstream", DataArray::new()); }
        if !state.has("memory") { state.put_object("memory", DataObject::new()); }
        
        CognitiveNode { state }
    }

    /// The Metabolic Pulse
    /// This is the single function that runs the node's life cycle.
    pub fn pulse(&mut self) {
        self.ingest();
        self.synthesize();
        self.emit();
    }

    /// Step 1: Ingest
    /// Reads from buffers. In a real implementation, this might also
    /// pull from the child node's output_upstream if connected explicitly here,
    /// though we prefer the parent to push into our buffer or a synapse to move it.
    fn ingest(&mut self) {
        // Placeholder: Log if we have input to process
        let inputs = self.state.get_array("input_buffer");
        if inputs.len() > 0 {
            // In Phase 1, we just acknowledge receipt.
            // println!("Holon [{}] Ingesting {} inputs.", self.state.get_string("id"), inputs.len());
        }
    }

    /// Step 2: Synthesize
    /// This is where the "Thinking" happens.
    /// It transforms Inputs -> Upstream Insights
    /// AND Directives -> Downstream Commands.
    fn synthesize(&mut self) {
        let mut inputs = self.state.get_array("input_buffer");
        let mut upstream = self.state.get_array("output_upstream");
        
        // 2a. Bottom-Up Processing (Abstraction)
        // For the Skeleton Phase, we act as a pass-through filter.
        // We move items from Input to Upstream, emptying input.
        // In Phase 2, this is where the LLM/Narrative Engine lives.
        while inputs.len() > 0 {
            let mut data = inputs.get_object(0);
            inputs.remove_property(0);
            
            // "Digesting" the data - Adding a processed timestamp
            data.put_int("processed_at", time());
            data.put_string("processed_by", &self.state.get_string("id"));
            
            // Push to output
            upstream.push_object(data);
        }

        // 2b. Top-Down Processing (Control)
        let mut directives = self.state.get_array("directives_buffer");
        let mut downstream = self.state.get_array("output_downstream");
        
        while directives.len() > 0 {
            let cmd = directives.get_object(0);
            directives.remove_property(0);
            // Pass the command down
            downstream.push_object(cmd);
        }
    }

    /// Step 3: Emit
    /// Finalizes state changes.
    /// In a threaded environment, this might be where we explicitly sync or yield.
    fn emit(&mut self) {
        // In this architecture using ndata, the writes in synthesize() are 
        // already committed to the shared memory view. 
        // This slot is reserved for side effects (e.g., hardware I/O for Level 1).
        
        let id = self.state.get_string("id");
        if id == "sensation" {
            // Special case: Level 1 actually emits to the outside world?
            // Or maybe it just clears its buffers to avoid overflow.
        }
    }
