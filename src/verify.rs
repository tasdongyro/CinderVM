//! Verification: the precondition for an [`Image`]. Nothing runs before
//! `admit` proves the invariants in the crate root's module docs.

use std::collections::HashMap;

use crate::asm::Object;
use crate::cfg;
use crate::diag::{Code, Diag, Result};
use crate::image::{self, FuncMeta, Image};
use crate::isa::{Op, Operand, Ty};

/// Prove the invariants and seal the image.
pub fn admit(object: Object) -> Result<Image> {
    image::seal_decode_check(&object.code)?;
    let mut funcs = object.funcs;
    if funcs.is_empty() {
        return Err(Diag::new(Code::NoEntryPoint, "image has no functions"));
    }
    let func_index: HashMap<String, u32> = funcs
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.clone(), i as u32))
        .collect();

    for f in &funcs {
        check_function(
            &object.code,
            f,
            &func_index,
            &object.jump_tables,
            object.constants.len(),
        )?;
    }

    let mut image = image::build(
        object.name,
        object.constants,
        funcs,
