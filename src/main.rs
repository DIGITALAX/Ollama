use dirs;
use dotenv::dotenv;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use std::{
    env, error::Error, fs, path::Path, process::{Command, Stdio}, thread, time::Duration
};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv().ok();
    let ollama_path = Path::new("ollama");
    let model_dir = dirs::home_dir().unwrap().join(".ollama/models");

    if let Some(parent) = ollama_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let output = Command::new("curl")
        .arg("-L")
        .arg("https://ollama.com/download/ollama-linux-amd64")
        .arg("-o")
        .arg(ollama_path.to_str().unwrap())
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "Failed to download ollama: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    Command::new("chmod")
        .arg("+x")
        .arg(ollama_path.to_str().unwrap())
        .output()?;

    println!("Ollama installed successfully at {:?}", ollama_path);
    fs::create_dir_all(&model_dir)?;
    env::set_var("OLLAMA_MODELS", model_dir.to_str().unwrap());

    Command::new("./ollama")
        .arg("serve")
        .env("OLLAMA_MODELS", model_dir.to_str().unwrap())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to start ollama server");

    thread::sleep(Duration::from_secs(10));

    let pull_output = Command::new("./ollama")
        .arg("pull")
        .arg("llama3:8b")
        .output()?;

    if !pull_output.status.success() {
        return Err(format!(
            "Failed to pull model llama3:8b: {}",
            String::from_utf8_lossy(&pull_output.stderr)
        )
        .into());
    }

    println!("Model llama3:8b installed successfully");

    let list_models_output = Command::new("./ollama").arg("list").output()?;

    if !list_models_output.status.success() {
        return Err(format!(
            "Failed to list models: {}",
            String::from_utf8_lossy(&list_models_output.stderr)
        )
        .into());
    }

    let models_list = String::from_utf8_lossy(&list_models_output.stdout);
    println!(
        "Lista de modelos antes de iniciar el servidor: {}",
        models_list
    );

    Ok(())
}


pub async fn llamar_llama(prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let ollama = Ollama::default();
    let model = "llama3:8b".to_string();

    let res = ollama
        .generate(GenerationRequest::new(model, prompt.to_string()))
        .await;

    match res {
        Ok(response) => Ok(response.response),
        Err(e) => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Error con Ollama {}", e),
        ))),
    }
}