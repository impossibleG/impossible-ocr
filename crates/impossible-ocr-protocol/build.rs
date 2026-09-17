//! Generates the checked protocol bindings with a vendored `protoc`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc);
    tonic_prost_build::configure()
        .build_transport(false)
        .compile_with_config(config, &["proto/impossible_ocr_v1.proto"], &["proto"])?;
    Ok(())
}
