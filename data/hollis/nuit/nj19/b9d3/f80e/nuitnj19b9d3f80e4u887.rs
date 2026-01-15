DataObject::new()
}

const FFT_SIZE: usize = 2048;
const WINDOW_CORRECTION: f32 = 2.0;

pub struct AudioSensor {
  _stream: cpal::Stream,
  pub sample_rate: u32,
}

impl AudioSensor {
  pub fn new(
    sensor_id: String,
    device_name_query: String, // [NEW] e.g., "Webcam" or "default"
    channel_selector: Option<usize>, // [NEW] None = Mix, Some(0) = Left, Some(1) = Right
    tx: Sender<AcousticFrame>,
  ) -> Result<Self, Box<dyn std::error::Error>> {

    let host = cpal::default_host();
    let devices = host.input_devices()?;

    // 1. Find the requested device
    let device = if device_name_query == "default" {
      host.default_input_device().ok_or("No default device")?
    } else {
      let mut found = None;
      for d in devices {
        if let Ok(name) = d.name() {
          if name.contains(&device_name_query) {
            found = Some(d);
            break;
          }
        }
      }
      if let Some(d) = found {
        d
      } else {
        // [DEBUG] List devices if not found
        println!("Could not find device containing '{}'. Available devices:", device_name_query);
        for d in host.input_devices()? {
          println!(" - {}", d.name().unwrap_or("Unknown".to_string()));
        }
        return Err("Device not found".into());
      }
    };

    println!("Initializing Sensor '{}' on device: {}", sensor_id, device.name()?);

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let buffer = Arc::new(Mutex::new(Vec::with_capacity(FFT_SIZE * channels as usize)));

    let tx = tx.clone();
    let sensor_id = sensor_id.clone();
    let bin_width = sample_rate as f32 / FFT_SIZE as f32;

    let err_fn = |err| eprintln!("Audio stream error: {}", err);

    let stream = match config.sample_format() {
      cpal::SampleFormat::F32 => device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
          process_audio_chunk(
            data, 
            &buffer, 
            FFT_SIZE, 
            &fft, 
            &tx, 
            &sensor_id,
            bin_width,
            channels,
            channel_selector 
          );
        },
        err_fn,
        None,
      )?,
      _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;

    Ok(AudioSensor { 
      _stream: stream,
      sample_rate 
    })
  }
}

fn process_audio_chunk(
  input: &[f32],
  buffer_mutex: &Arc<Mutex<Vec<f32>>>,
  fft_size: usize,
  fft: &Arc<dyn rustfft::Fft<f32>>,
  tx: &Sender<AcousticFrame>,
  sensor_id: &str,
  bin_width: f32,
  channels: u16,
  channel_selector: Option<usize>,
) {
  let mut buffer = match buffer_mutex.lock() {
    Ok(guard) => guard,
    Err(_) => return,
  };

  // 1. Channel Selection Logic
  // If we have a selector (e.g., 0 for Left), grab that. Otherwise, Downmix.
  if let Some(target_ch) = channel_selector {
      if (channels as usize) > target_ch {
          let selected_chunk: Vec<f32> = input.chunks(channels as usize)
              .map(|chunk| chunk[target_ch]) // Pick the specific channel
              .collect();
          buffer.extend(selected_chunk);
      } else {
          // Fallback: If requested channel doesn't exist, just use Ch 0
          let fallback: Vec<f32> = input.chunks(channels as usize)
              .map(|chunk| chunk[0])
              .collect();
          buffer.extend(fallback);
      }
  } else {
    // 1. Stereo to Mono Downmix
    // If we receive stereo [L, R, L, R], we average them to [M, M].
    if channels > 1 {
      // This is a simple iterator approach to averaging channels
      let mono_chunk: Vec<f32> = input.chunks(channels as usize)
      .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
      .collect();
      buffer.extend(mono_chunk);
    } else {
      buffer.extend_from_slice(input);
    }
  }
  
  if buffer.len() >= fft_size {
    let mut raw_samples: Vec<f32> = buffer.drain(0..fft_size).collect();

    // 2. DC Offset Removal (Centering the wave)
    let mean: f32 = raw_samples.iter().sum::<f32>() / fft_size as f32;
    for sample in &mut raw_samples {
      *sample -= mean;
    }

    // Zero-Crossing Rate (Time Domain)
    // Count how many times the signal sign changes (positive <-> negative)
    let mut zero_crossings = 0;
    for i in 1..raw_samples.len() {
      if (raw_samples[i-1] > 0.0 && raw_samples[i] <= 0.0) || 
      (raw_samples[i-1] <= 0.0 && raw_samples[i] > 0.0) {
        zero_crossings += 1;
      }
    }
    // Normalize: ZCR of 1.0 means it crossed every single sample (Nyquist noise)
    let zcr = zero_crossings as f32 / raw_samples.len() as f32;      



    // 3. RMS Calculation (Time Domain)
    let sum_squares: f32 = raw_samples.iter().map(|&x| x * x).sum();
    let rms = (sum_squares / fft_size as f32).sqrt();

    // 4. Windowing
    let mut plan_buffer: Vec<Complex<f32>> = raw_samples
    .iter()
    .enumerate()
    .map(|(i, &sample)| {
      let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size as f32 - 1.0)).cos());
      Complex::new(sample * window, 0.0)
    })
    .collect();

    // 5. FFT
    fft.process(&mut plan_buffer);

    // 6. Spectrum Analysis
    let spectrum_len = fft_size / 2;
    let spectrum: Vec<f32> = plan_buffer[0..spectrum_len]
    .iter()
    .map(|c| (c.norm() / fft_size as f32) * WINDOW_CORRECTION) 
    .collect();




    // Spectral Centroid (Frequency Domain)
    // Formula: Sum(Freq * Magnitude) / Sum(Magnitude)
    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    // Calculate a dynamic threshold for "significant energy"
    // We use the RMS we just calculated to ignore quiet bands.
    // If a specific frequency bin is much quieter than the overall loudness, ignore it.
    let mag_threshold = rms * 0.1; 

    for (i, &magnitude) in spectrum.iter().enumerate() {
      // Only count this frequency if it's actually audible above the mix
      if magnitude > mag_threshold {
        let freq = i as f32 * bin_width;
        weighted_sum += freq * magnitude;
        total_magnitude += magnitude;
      }
    }

    let spectral_centroid = if total_magnitude > 0.0 {
      weighted_sum / total_magnitude
    } else {
      0.0
    };    


    // 7. Find Dominant Frequency (Ignoring DC/Index 0)
    let mut max_val = 0.0;
    let mut max_idx = 0;

    // Skip index 0 (0Hz) and start looking from index 1 (approx 21Hz)
    for (i, &val) in spectrum.iter().enumerate().skip(1) {
      if val > max_val {
        max_val = val;
        max_idx = i;
      }
    }

    let dominant_freq = max_idx as f32 * bin_width;

    // 8. Timestamp & Send
    let start = SystemTime::now();
    let timestamp = start.duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_micros() as u64;

    let frame = AcousticFrame {
      sensor_id: sensor_id.to_string(),
      timestamp_micros: timestamp,
      rms_power: rms,
      dominant_freq_hz: Some(dominant_freq),
      spectrum,
      zcr,               // <--- Added
      spectral_centroid, // <--- Added
      direction_of_arrival: None,
      raw_samples: raw_samples.clone(),
    };

    let _ = tx.try_send(frame);
  }
}

fn qwert() {