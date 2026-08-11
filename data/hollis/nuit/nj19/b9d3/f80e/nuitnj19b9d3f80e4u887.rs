DataObject::new()
}

pub fn setup_hardware_routing() {
    println!("--- Initializing Headless Hardware Routing ---");
    
    let check = std::process::Command::new("pw-link").arg("-o").output().unwrap();
    let output = String::from_utf8_lossy(&check.stdout);
    
    if !output.contains("hollis_aggregate") {
        println!("Building 7-channel 3D Array...");
        let _ = std::process::Command::new("pactl")
            .args([
                "load-module", 
                "module-null-sink", 
                "media.class=Audio/Sink", 
                "sink_name=hollis_aggregate", 
                "channel_map=front-left,front-right,front-center,lfe,rear-left,rear-right,side-left"
            ])
            .status();
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }
    
    println!("Patching physical hardware to the array...");
    let links = vec![
        ("alsa_output.usb-Generic_USB_Audio-00.HiFi_7_1__Speaker__sink:monitor_FL", "hollis_aggregate:playback_FL"),
        ("alsa_output.usb-Generic_USB_Audio-00.HiFi_7_1__Speaker__sink:monitor_FR", "hollis_aggregate:playback_FR"),
        ("alsa_input.usb-Generic_Blue_Microphones_LT_210915032721AD02002E_111000-00.analog-stereo:capture_FL", "hollis_aggregate:playback_FC"),
        ("alsa_input.usb-Generic_Blue_Microphones_LT_210915032721AD02002E_111000-00.analog-stereo:capture_FR", "hollis_aggregate:playback_LFE"),
        ("alsa_input.usb-Andrea_Electronics_Andrea_PureAudio-00.analog-stereo:capture_FL", "hollis_aggregate:playback_RL"),
        ("alsa_input.usb-Andrea_Electronics_Andrea_PureAudio-00.analog-stereo:capture_FR", "hollis_aggregate:playback_RR"),
        ("alsa_input.usb-HD_Camera_Manufacturer_HD_USB_Camera_SN0047-03.mono-fallback:capture_MONO", "hollis_aggregate:playback_SL"), 
    ];

    println!("Patching physical hardware to the array...");
    for (src, dest) in links {
        let out = std::process::Command::new("pw-link").arg(src).arg(dest).output().unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        
        // If there's an error, and it's NOT just telling us it's already plugged in, scream about it!
        if !err.is_empty() && !err.contains("File exists") {
            println!("  [FAIL] {} -> {}: {}", src, dest, err.trim());
        } else if err.is_empty() {
            println!("  [ OK ] {} -> {}", src, dest);
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
            .expect("CRITICAL: Failed to spawn parec.");
            
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
