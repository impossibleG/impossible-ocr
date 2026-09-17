//! Explicit command-line entry point for offline curated graph qualification.

use std::{env, path::Path};

use impossible_ocr_onnx::{ModelRole, qualify_curated_graph, render_qualified_contract};

const USAGE: &str =
    "usage: qualify-onnx --role <detector|english-recognizer> --artifact <explicit-path>";

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), &'static str> {
    let mut arguments = env::args_os().skip(1);
    if arguments.next().as_deref() != Some("--role".as_ref()) {
        return Err(USAGE);
    }
    let role = match arguments.next().and_then(|value| value.into_string().ok()) {
        Some(value) if value == "detector" => ModelRole::Detector,
        Some(value) if value == "english-recognizer" => ModelRole::EnglishRecognizer,
        _ => return Err(USAGE),
    };
    if arguments.next().as_deref() != Some("--artifact".as_ref()) {
        return Err(USAGE);
    }
    let artifact = arguments.next().ok_or(USAGE)?;
    if arguments.next().is_some() {
        return Err(USAGE);
    }
    let contract = qualify_curated_graph(Path::new(&artifact), role)
        .map_err(|_| "qualification failed; artifact paths and graph details are redacted")?;
    let rendered = render_qualified_contract(&contract)
        .map_err(|_| "qualification failed; artifact paths and graph details are redacted")?;
    print!("{rendered}");
    Ok(())
}
