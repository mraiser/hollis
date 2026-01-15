let f = DataStore::new().root.join("RAW_LLM");
std::fs::create_dir_all(&f);
let mut x = count_files(f.clone());

let out = {
  let (system_prompt, captured_prompt) = match system_prompt.is_string() {
    true => (Some(system_prompt.string()), system_prompt.string()),
    _ => (None, "".to_string())
  };

  let captured_prompt = format!("{}\n\n{}", &captured_prompt, &prompt);
  let p = f.join(format!("Q{}.txt", x));
  std::fs::write(p, &captured_prompt).ok();

//println!("LLM SYSTEM PROMPT:\n{:?}", system_prompt);
//println!("LLM PROMPT:\n{}", &prompt);

  let meta = DataStore::globals().get_object("system").get_object("apps").get_object("hollis").get_object("runtime");

  let llm_type = match meta.has("LLM") {
    true => meta.get_string("LLM"),
    _ => "GEMINI".to_string()
  };

  match llm_type.as_str() {
    "GEMINI" => {
      // Call the helper function for Gemini
      ask_gemini(prompt, system_prompt)
    },
    "OLLAMA" => {
      // Call the helper function for Ollama
      ask_ollama(prompt, system_prompt)
    },
    _ => {
      // Fallback for other LLMs (e.g., Mistral-RS)
      let whole = meta.get_string("LLM_CTL");
      let parts: Vec<&str> = whole.split(':').collect();
      if parts.len() < 3 {
        // Return an error string if format is wrong
        return "Error: LLM_CTL is not in the format 'lib:ctl:cmd'".to_string();
      }
      let lib = parts[0];
      let ctl = parts[1];
      let cmd = parts[2];
      
      let mut params = DataObject::new();
      params.put_string("prompt", &prompt);
      if let Some(system_prompt) = system_prompt {
        params.put_string("system_prompt", &system_prompt);
      }
      let command = Command::lookup(lib, ctl, cmd);
      let result = command.execute(params);
      if let Ok(actual_result) = result {
        let msg = actual_result.get_string("msg");
        println!("RAW RESULT: {}", &msg);
        return msg;
      }
      else {
        format!("ERROR: {:?}", result.err().unwrap())
      }
    }
  }
};
x += 1;
let p = f.join(format!("A{}.txt", x));
std::fs::write(p, &out).ok();
out
}

fn count_files(base_dir:PathBuf) -> usize{
  match std::fs::read_dir(&base_dir) {
    Ok(entries) => {
      entries
      .filter_map(|entry| entry.ok()) // Filter out errors
      .filter(|entry| entry.metadata().map(|m| m.is_file()).unwrap_or(false)) // Keep only files
      .count()
    }
    Err(e) => {
      0
    }
  }
}

pub fn ask_gemini(prompt:String, system_prompt: Option<String>) -> String {
  let meta = DataStore::globals().get_object("system").get_object("apps").get_object("hollis").get_object("runtime");
  
  let api_key = &meta.get_string("GEMINI_API_KEY");
  let _model = &meta.get_string("GEMINI_MODEL");
  let url = format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}", api_key);

  // --- 1. Build the complete payload using ndata::DataObject ---
  let mut payload = DataObject::new();

  // Add "system_instruction" if a system_prompt is provided
  if let Some(sp_text) = system_prompt {
    let mut system_instruction_part = DataObject::new();
    system_instruction_part.put_string("text", &sp_text);
    let mut system_instruction_parts_array = DataArray::new();
    system_instruction_parts_array.push_object(system_instruction_part);
    let mut system_instruction = DataObject::new();
    system_instruction.put_array("parts", system_instruction_parts_array);
    payload.put_object("system_instruction", system_instruction);
  }

  // Add "contents" array
  let mut part = DataObject::new();
  part.put_string("text", &prompt);
  let mut parts_array = DataArray::new();
  parts_array.push_object(part);
  let mut content = DataObject::new();
  content.put_array("parts", parts_array);
  let mut contents_array = DataArray::new();
  contents_array.push_object(content);
  payload.put_array("contents", contents_array);
 
  // Add "generationConfig" for controlling the output
  let mut gen_config = DataObject::new();
  gen_config.put_float("temperature", 0.9);
  gen_config.put_int("maxOutputTokens", 200000);
  payload.put_object("generationConfig", gen_config);
 
  // Add "safetySettings" to avoid default blocking
  let mut safety_settings = DataArray::new();
  let categories = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
  ];
  for category in categories.iter() {
      let mut setting = DataObject::new();
      setting.put_string("category", category);
      setting.put_string("threshold", "BLOCK_NONE"); // Set all categories to BLOCK_NONE
      safety_settings.push_object(setting);
  }
  payload.put_array("safetySettings", safety_settings);
  
  println!("PAYLOAD: {}", payload.to_string());

  // --- Retry mechanism for API calls ---
  let mut final_response_text = "{\"Unsupported\":null}".to_string(); // Default if all retries fail
  for _ in 0..5 { // Attempt up to 5 times
    // --- 2. Make the HTTP Request ---
    let response_result = attohttpc::post(&url)
        .header("Content-Type", "application/json")
        .json(&payload.to_json())
        .unwrap() 
        .send();

    // --- 3. Process the API response ---
    println!("RAW RESPONSE: {:?}", response_result);
    match response_result {
        Ok(response) => {
            if response.is_success() {
                match response.text() {
                    Ok(body) => {
                        println!("RESPONSE FROM GEMINI: {}", body);
                        let root = DataObject::from_string(&body);
                        let text_opt = root.try_get_array("candidates").ok()
                            .and_then(|candidates| candidates.try_get_object(0).ok())
                            .and_then(|candidate| candidate.try_get_object("content").ok())
                            .and_then(|content| content.try_get_array("parts").ok())
                            .and_then(|parts| parts.try_get_object(0).ok())
                            .and_then(|part| part.try_get_string("text").ok());

                        match text_opt {
                            Some(text) => {
                                return text;
                            }
                            None => {
                              final_response_text = format!("Error: Failed to parse Gemini response structure. Body: {}", body);
                            }
                        }
                    }
                    Err(e) => {
                      final_response_text = format!("Error: Failed to read Gemini response body: {}", e);
                    }
                }
            } else {
                final_response_text = format!("Error: Gemini API request failed with status: {}. Body: {}", response.status(), response.text().unwrap_or_default());
            }
        }
        Err(e) => {
            final_response_text = format!("Error: HTTP request to Gemini API failed: {}", e);
        }
    }
    println!("LLM call failed, retrying... Error: {}", final_response_text);
    std::thread::sleep(std::time::Duration::from_secs(1));
  }
  final_response_text
}


pub fn ask_ollama(prompt: String, system_prompt: Option<String>) -> String {
  let meta = DataStore::globals().get_object("system").get_object("apps").get_object("hollis").get_object("runtime");
  
  // Assumes OLLAMA_URL (e.g., "http://localhost:11434/api/generate") and OLLAMA_MODEL are configured
  let url = &meta.get_string("OLLAMA_URL");
  let model = &meta.get_string("OLLAMA_MODEL");

  // --- 1. Build the payload for Ollama ---
  let mut payload = DataObject::new();
  payload.put_string("model", model);
  payload.put_string("prompt", &prompt);
  payload.put_bool("stream", false); // Request a single, complete response
  payload.put_int("keep_alive", 0);

  if let Some(sp_text) = system_prompt {
    payload.put_string("system", &sp_text);
  }

  //println!("OLLAMA PAYLOAD: {}", payload.to_string());

  // --- Retry mechanism for API calls ---
  let mut final_response_text = "{\"Unsupported\":null}".to_string(); // Default if all retries fail
  for _ in 0..5 { // Attempt up to 5 times
    // --- 2. Make the HTTP Request ---
    let response_result = attohttpc::post(url)
        .header("Content-Type", "application/json")
        .json(&payload.to_json())
        .unwrap()
        .send();

    // --- 3. Process the API response ---
    //println!("RAW OLLAMA RESPONSE: {:?}", response_result);
    match response_result {
        Ok(response) => {
            if response.is_success() {
                match response.text() {
                    Ok(body) => {
                        //println!("RESPONSE FROM OLLAMA: {}", body);
                        let root = DataObject::from_string(&body);
                        // For a non-streaming response, the text is in the "response" field
                        match root.try_get_string("response") {
                            Ok(text) => {
                                // NEW: Trim the result before returning
                                return text;
                            }
                            Err(_) => {
                              final_response_text = format!("Error: Failed to parse Ollama response structure. Body: {}", body);
                            }
                        }
                    }
                    Err(e) => {
                      final_response_text = format!("Error: Failed to read Ollama response body: {}", e);
                    }
                }
            } else {
                final_response_text = format!("Error: Ollama API request failed with status: {}. Body: {}", response.status(), response.text().unwrap_or_default());
            }
        }
        Err(e) => {
            final_response_text = format!("Error: HTTP request to Ollama API failed: {}", e);
        }
    }
    println!("LLM call failed, retrying... Error: {}", final_response_text);
    std::thread::sleep(std::time::Duration::from_secs(1));
  }
  final_response_text
