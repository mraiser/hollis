DataObject::new()
}

// Math Helper
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }

    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

// ==========================================
// 1. FUNDAMENTAL MATH TYPES
// ==========================================

/// A point in 3D space (meters), relative to a shared origin (e.g., center of the house).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    
    /// Calculate Euclidean distance to another point
    pub fn distance(&self, other: &Point3D) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

/// A vector representing direction and magnitude (velocity or orientation).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3D {
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
}

// ==========================================
// 2. THE SENSOR LAYER (The "Ear")
// ==========================================

/// The atomic unit of data passed from the Sensor Loop to the Perception Loop.
/// This represents a short slice of time (e.g., 50ms - 100ms).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticFrame {
    /// Unique identifier of the physical device/microphone
    pub sensor_id: String,
    
    /// Microseconds since UNIX EPOCH. Critical for synchronizing multiple sensors.
    pub timestamp_micros: u64,
    
    /// Root Mean Square amplitude (loudness)
    pub rms_power: f32,
    
    /// The dominant frequency found via FFT (optional, optimization for quick filtering)
    pub dominant_freq_hz: Option<f32>,
    
    /// The raw frequency spectrum bins (FFT output). 
    /// We store this so the Perception Layer can do advanced analysis (MFCCs).
    pub spectrum: Vec<f32>,
  
    // Zero-Crossing Rate (0.0 to 1.0)
    pub zcr: f32,
    
    // Center of mass in Hz
    pub spectral_centroid: f32,

    /// If the sensor is an array, it might pre-calculate a Direction of Arrival.
    /// (Azimuth, Elevation) in radians.
    pub direction_of_arrival: Option<(f32, f32)>, 
  
    /// We need to pass the raw audio to the Transcriber
    pub raw_samples: Vec<f32>,
}

// ==========================================
// 3. THE PERCEPTION LAYER (The "Cortex")
// ==========================================

/// ContextBriefing - The Atomic Unit of Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBriefing {
    pub domain: String,       // e.g., "Atmosphere", "Occupancy", "Discourse"
    pub summary: String,      // e.g., "The room is library-quiet."
    pub confidence: f32,      // 0.0 to 1.0
    pub urgency: u8,          // 1 (Background) to 10 (Critical Alert)
    pub timestamp: u64,
}

/// A high-level conclusion drawn from analyzing a stream of AcousticFrames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEvent {
    /// When the event started
    pub start_timestamp: u64,
    
    /// When it ended (None if currently ongoing)
    pub end_timestamp: Option<u64>,
    
    /// Which sensors contributed to this conclusion?
    pub sources: Vec<String>,
    
    /// The nature of the event
    pub kind: EventKind,
  
    /// The raw spectrum so Cortex can learn
    pub fingerprint: Vec<f32>,
    
    // Angle of arrival
    pub angle: Option<f32>, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventKind {
    /// A short, sharp sound (Clap, Drop, Gunshot)
    Transient { 
        label: String, 
        confidence: f32,
        peak_db: f32 
    },
    
    /// A sustained sound (Fan, Voice, Music)
    Continuous { 
        label: String, 
        is_speech: bool 
    },
    
    /// Transcription of detected voice audio
    Transcript {
        text: String,
    },
    
    /// A detected change in the noise floor (e.g., "The room got quieter")
    StateChange {
        previous_db: f32,
        current_db: f32,
    },
  
    ContextUpdate(ContextBriefing),
}

// ==========================================
// 4. THE WORLD MODEL (State Management)
// ==========================================

/// The "Global State" of the system.
/// This is what your main loop maintains and updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    /// Active tracking of sound sources
    pub entities: HashMap<u64, Entity>,
    
    /// The physical layout of the environment
    pub map: AcousticMap,
    
    /// Registry of all connected sensors
    pub sensors: HashMap<String, SensorConfig>,
  
    /// Stores the latest briefing from each domain
    pub context: HashMap<String, ContextBriefing>,
}

/// A dynamic object in the world (Person, machine, pet)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: u64,
    pub label: String,       // "Unknown Person", "HVAC"
    pub position: Point3D,   // Estimated location
    pub velocity: Vector3D,  // Movement vector
    pub last_heard: u64,     // Timestamp
  
	// The acoustic signature of this entity
    // We average the spectrums of all events linked to this entity
    pub signature: Vec<f32>,
    pub signature_count: usize, // To calculate rolling average
}

/// The static environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticMap {
    pub zones: Vec<Zone>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zone {
    pub name: String,        // "Kitchen"
    pub center: Point3D,
    pub radius: f32,         // Simple spherical zones for now
    pub noise_floor_db: f32, // Learned background noise level
}

// ==========================================
// 5. CONFIGURATION & MESSAGING
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    pub id: String,
    pub location: Point3D,   // Where is this sensor installed?
    pub sample_rate: u32,
    pub is_array: bool,      // Does it support directionality?
}

/// Enums for message passing between threads or network nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemMessage {
    /// Sensor -> Perception
    AudioData(AcousticFrame),
    
    /// Perception -> World
    Inference(SemanticEvent),
    
    /// Network -> World (Discovery)
    RegisterSensor(SensorConfig),
    
    /// World -> Network (Sync)
    StateUpdate(WorldState),
}

fn qwert() {