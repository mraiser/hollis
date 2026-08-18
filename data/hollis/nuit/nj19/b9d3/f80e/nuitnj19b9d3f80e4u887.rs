DataObject::new()
}

// ── Tier 1 ear discovery ─────────────────────────────────────────────
// The link map is DERIVED, not hardcoded: the AEC reference is the
// default sink's monitor (that is where the calibration chirp and all
// playback goes, so it is what the canceller must subtract), and every
// alsa_input capture port becomes a mic in stable sorted order. Two
// botd keys (runtime/hollis/botd.properties) override it:
//   links=src>dst;src>dst;...  the full explicit map. dst may be a bare
//     aggregate channel (FL FR FC LFE RL RR SL) or a full port name.
//   ignore=substr,substr,...   capture ports to skip in discovery.
// Only LOCALIZATION cares which mic lands in which slot - transcripts
// and voiceprints work in any order. Pin with links= when geometry
// matters. Pure and separable so the mapping is testable without audio
// hardware.
pub fn resolve_links(ports: &str, default_sink: &str, links_cfg: &str, ignore_cfg: &str) -> Vec<(String, String)> {
    let agg = |ch: &str| format!("hollis_aggregate:playback_{}", ch);
    if !links_cfg.trim().is_empty() {
        return links_cfg.split(';')
            .filter_map(|pair| {
                let (src, dst) = pair.split_once('>')?;
                let (src, dst) = (src.trim(), dst.trim());
                if src.is_empty() || dst.is_empty() { return None; }
                Some((src.to_string(),
                      if dst.contains(':') { dst.to_string() } else { agg(dst) }))
            })
            .collect();
    }
    let mut out = Vec::new();
    let mut monitors: Vec<&str> = ports.lines().map(str::trim)
        .filter(|l| !default_sink.is_empty()
            && l.starts_with(&format!("{}:monitor_", default_sink)))
        .collect();
    monitors.sort();
    for (port, ch) in monitors.iter().zip(["FL", "FR"]) {
        out.push((port.to_string(), agg(ch)));
    }
    if monitors.is_empty() {
        println!("[ARRAY] no monitor ports on default sink {:?} - echo cancellation runs without a reference", default_sink);
    }
    let ignores: Vec<&str> = ignore_cfg.split(',').map(str::trim)
        .filter(|s| !s.is_empty()).collect();
    let mut mics: Vec<&str> = ports.lines().map(str::trim)
        .filter(|l| l.starts_with("alsa_input.") && l.contains(":capture_"))
        .filter(|l| !ignores.iter().any(|ig| l.contains(ig)))
        .collect();
    mics.sort();
    let slots = ["FC", "LFE", "RL", "RR", "SL"];
    for (i, port) in mics.iter().enumerate() {
        if i < slots.len() {
            out.push((port.to_string(), agg(slots[i])));
        } else {
            println!("[ARRAY] array full - not assigned: {}", port);
        }
    }
    if mics.len() < slots.len() {
        println!("[ARRAY] {} of {} mic slots filled; the rest stay silent", mics.len(), slots.len());
    }
    out
}

pub fn setup_hardware_routing() {
    println!("--- Initializing Headless Hardware Routing ---");

    let ports = match std::process::Command::new("pw-link").arg("-o").output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => {
            println!("[ARRAY] pw-link not runnable ({}) - install pipewire's tools; the array gets no inputs", e);
            return;
        }
    };

    if !ports.contains("hollis_aggregate") {
        println!("Building 7-channel 3D Array...");
        let ok = std::process::Command::new("pactl")
            .args([
                "load-module",
                "module-null-sink",
                "media.class=Audio/Sink",
                "sink_name=hollis_aggregate",
                "channel_map=front-left,front-right,front-center,lfe,rear-left,rear-right,side-left"
            ])
            .status();
        match ok {
            Ok(s) if s.success() => {}
            Ok(s) => println!("[ARRAY] pactl load-module failed (exit {:?}) - is the PipeWire pulse shim up? (pactl info)", s.code()),
            Err(e) => println!("[ARRAY] pactl not runnable ({}) - install the pulseaudio client tools; the array cannot exist", e),
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }

    // the overrides ride runtime/hollis/botd.properties
    let (links_cfg, ignore_cfg) = (|| -> Option<(String, String)> {
        let g = flowlang::datastore::DataStore::globals();
        let meta = g.try_get_object("system").ok()?
            .try_get_object("apps").ok()?
            .try_get_object("hollis").ok()?
            .try_get_object("runtime").ok()?;
        Some((meta.try_get_string("links").unwrap_or_default(),
              meta.try_get_string("ignore").unwrap_or_default()))
    })().unwrap_or_default();

    let default_sink = std::process::Command::new("pactl")
        .args(["get-default-sink"]).output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let links = resolve_links(&ports, &default_sink, &links_cfg, &ignore_cfg);
    println!("Patching {} inputs to the array{}...", links.len(),
        if links_cfg.trim().is_empty() { " (discovered; pin with links= in botd)" } else { " (links= from botd)" });
    for (src, dest) in &links {
        match std::process::Command::new("pw-link").arg(src).arg(dest).output() {
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                if !err.is_empty() && !err.contains("File exists") {
                    println!("  [FAIL] {} -> {}: {}", src, dest, err.trim());
                } else {
                    println!("  [ OK ] {} -> {}", src, dest);
                }
            }
            Err(e) => println!("  [FAIL] {} -> {}: pw-link: {}", src, dest, e),
        }
    }
}

pub struct MasterSensorArray {
    _process: std::process::Child,
    pub sample_rate: u32,
    pub total_channels: u16,
}

impl MasterSensorArray {
    pub fn new(
        _device_name_query: &str, // No longer needed, we hardcode the dish
        loopback_count: usize, 
        mic_count: usize,      
        tx: crossbeam_channel::Sender<SynchronizedArrayFrame>,
    ) -> Result<Self, Box<dyn std::error::Error>> {

        let required_channels = (loopback_count + mic_count) as u16;
        let sample_rate = 44100;

        println!("Bypassing ALSA completely. Piping raw data directly from Radar Dish...");

        // 1. The Raw Audio Pipeline
        let (raw_tx, raw_rx) = crossbeam_channel::bounded::<Vec<f32>>(1000);

        // 2. The DSP Worker Thread
        std::thread::spawn(move || {
            let mut buffer = Vec::new();
            let chunk_size = 2048 * required_channels as usize; 

            while let Ok(raw_data) = raw_rx.recv() {
                buffer.extend(raw_data);

                while buffer.len() >= chunk_size {
                    let chunk: Vec<f32> = buffer.drain(0..chunk_size).collect();
                    
                    let mut loopback_mix = vec![0.0; 2048];
                    let mut isolated_mics = vec![vec![0.0; 2048]; mic_count];

                    for frame_idx in 0..2048 {
                        let offset = frame_idx * required_channels as usize;
                        
                        if loopback_count >= 2 {
                            loopback_mix[frame_idx] = (chunk[offset] + chunk[offset + 1]) / 2.0;
                        } else if loopback_count == 1 {
                            loopback_mix[frame_idx] = chunk[offset];
                        }

                        for mic_idx in 0..mic_count {
                            isolated_mics[mic_idx][frame_idx] = chunk[offset + loopback_count + mic_idx];
                        }
                    }

                    let _ = tx.try_send(SynchronizedArrayFrame {
                        timestamp_micros: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as u64,
                        loopback_reference: loopback_mix,
                        mic_channels: isolated_mics,
                    });
                }
            }
        });

        // 3. The OS-Level Audio Pipe
        let mut child = std::process::Command::new("parec")
            .args([
                "--device=hollis_aggregate.monitor", 
                "--format=float32le", 
                &format!("--channels={}", required_channels), 
                "--channel-map=front-left,front-right,front-center,lfe,rear-left,rear-right,side-left", // <--- FORCE MAP
                &format!("--rate={}", sample_rate)
            ])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!(
                "could not spawn parec ({}) - install the pulseaudio client \
                 tools (NixOS: pulseaudio in the shell env; Debian/Ubuntu: \
                 pulseaudio-utils)", e))?;
            
        let mut stdout = child.stdout.take().unwrap();
        
        std::thread::spawn(move || {
            use std::io::Read;
            // 6 channels * 4 bytes per float * 1024 frames = 24,576 bytes per read
            let chunk_bytes = required_channels as usize * 4 * 1024;
            let mut buf = vec![0u8; chunk_bytes]; 
            
            loop {
                // read_exact ensures we never misalign our float conversion!
                match stdout.read_exact(&mut buf) {
                    Ok(_) => {
                        let floats: Vec<f32> = buf.chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        
                        let _ = raw_tx.try_send(floats);
                    }
                    Err(_) => {
                        eprintln!("Audio pipe disconnected.");
                        break;
                    }
                }
            }
        });

        Ok(MasterSensorArray { _process: child, sample_rate, total_channels: required_channels })
    }
}

impl Drop for MasterSensorArray {
    fn drop(&mut self) {
        println!("Shutting down hardware audio pipe (parec)...");
        // Forcefully kill the background Linux process
        let _ = self._process.kill();
        // Wait for it to close to prevent a "zombie" process in htop
        let _ = self._process.wait(); 
    }
