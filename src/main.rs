use actix_service::Service;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use dirs;
use dotenv::dotenv;
use futures_util::future::ok;
use futures_util::future::Either;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use std::{
    env,
    error::Error,
    fs,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Duration,
};
use tokio;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();

    let render_clave = std::env::var("RENDER_KEY").expect("Sin Clave");
    let ollama_path = Path::new("ollama");
    let model_dir = dirs::home_dir().unwrap().join(".ollama/models");

    if let Some(parent) = ollama_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    let output = Command::new("curl")
        .arg("-L")
        .arg("https://ollama.com/download/ollama-linux-amd64")
        .arg("-o")
        .arg(ollama_path.to_str().unwrap())
        .output()
        .unwrap();

    if !output.status.success() {
        panic!(
            "Failed to download ollama: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Command::new("chmod")
        .arg("+x")
        .arg(ollama_path.to_str().unwrap())
        .output()
        .unwrap();

    println!("Ollama installed successfully at {:?}", ollama_path);
    fs::create_dir_all(&model_dir).unwrap();
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
        .output()
        .unwrap();

    if !pull_output.status.success() {
        panic!(
            "Failed to pull model llama3:8b: {}",
            String::from_utf8_lossy(&pull_output.stderr)
        );
    }

    println!("Model llama3:8b installed successfully");

    let list_models_output = Command::new("./ollama").arg("list").output().unwrap();

    if !list_models_output.status.success() {
        panic!(
            "Failed to list models: {}",
            String::from_utf8_lossy(&list_models_output.stderr)
        );
    }

    let models_list = String::from_utf8_lossy(&list_models_output.stdout);
    println!(
        "Lista de modelos antes de iniciar el servidor: {}",
        models_list
    );

    HttpServer::new(move || {
        let render_clave_clone = render_clave.clone();
        App::new()
            .wrap_fn(move |req, srv| {
                let render_clave_clone = render_clave_clone.clone();
                let auth_header = req.headers().get("Authorization");

                match auth_header.and_then(|h| h.to_str().ok()) {
                    Some(header) if header == render_clave_clone => Either::Left(srv.call(req)),
                    _ => Either::Right(ok(req.into_response(
                        HttpResponse::Unauthorized().finish().map_into_boxed_body(),
                    ))),
                }
            })
            .route("/", web::get().to(index))
            .route("/generate", web::post().to(generate_prompt))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

async fn index() -> impl Responder {
    HttpResponse::Ok().body("Hello! Send a POST request to /generate with your prompt.")
}

async fn generate_prompt(prompt: web::Json<Prompt>) -> impl Responder {
    match llamar_llama(&prompt.text).await {
        Ok(response) => HttpResponse::Ok().body(response),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    }
}

#[derive(serde::Deserialize)]
struct Prompt {
    text: String,
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
