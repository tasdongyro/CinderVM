//! `cinderc` — the assembler CLI: assemble `.cdx`, disassemble `.cdxb`, run
//! images, and emit the ISA reference.

use std::process::ExitCode;

use cindervm::image::Image;
use cindervm::{asm, build, disas, interp, isa, trap, verify, Diag, Value, VERSION};

fn usage() -> String {
    format!(
        "cinderc {VERSION} — CinderVM toolchain\n\n\
         usage:\n  cinderc build <in.cdx> [-o out.cdxb]\n  \
         cinderc run <in.cdxb> [--trace]\n  \
         cinderc dis <in.cdxb>\n  \
         cinderc --emit-isa-md\n  \
         cinderc --version\n"
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }
    match args[0].as_str() {
        "--version" => {
            println!("cinderc {VERSION}");
            ExitCode::SUCCESS
        }
        "--emit-isa-md" => {
