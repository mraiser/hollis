std::fs::write(&path, content.as_bytes()).unwrap();
format!("wrote {} bytes to {}", content.len(), path)