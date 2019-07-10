//! `cinder` — the runner and explain tool.

use std::process::ExitCode;

use cindervm::{Code, VERSION};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("--help") | Some("-h") => {
