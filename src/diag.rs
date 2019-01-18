//! Diagnostics: source spans, stable error codes, and rendering.
//!
//! Every failure in this crate is a [`Diag`] carrying a machine-readable
//! [`Code`] and, where a source is available, a span into it. The codes are
//! part of the public contract — `corpus/MANIFEST.tsv` names the exact code
//! each rejected image must produce, so renaming one breaks the conformance
//! suite on purpose.

use core::fmt;

/// Byte range into a source file, plus a precomputed line/column for rendering.
///
/// Spans survive assembly: `asm` attaches one to every emitted instruction and
/// `image` stores them in an optional debug section, which is what lets a
/// *verifier* failure point at `.cdx` source even though the verifier only ever
/// sees bytecode.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
    pub line: u32,
    pub col: u32,
}

impl Span {
    #[must_use]
    pub const fn new(lo: u32, hi: u32, line: u32, col: u32) -> Self {
        Self { lo, hi, line, col }
    }

    /// A span with no extent, used for synthesized instructions (branch padding,
    /// the implicit `halt` on a fallthrough at end of function).
    #[must_use]
    pub const fn synthetic() -> Self {
        Self {
            lo: 0,
            hi: 0,
            line: 0,
            col: 0,
        }
    }

    #[must_use]
    pub const fn is_synthetic(self) -> bool {
        self.line == 0
    }

    /// Smallest span covering both. Used when a diagnostic blames a region
    /// spanning several instructions, e.g. an unbalanced `fork`/`commit` pair.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        if self.is_synthetic() {
            return other;
        }
        if other.is_synthetic() {
            return self;
        }
        let (lo, line, col) = if self.lo <= other.lo {
            (self.lo, self.line, self.col)
        } else {
            (other.lo, other.line, other.col)
        };
        Self {
            lo,
            hi: self.hi.max(other.hi),
            line,
            col,
        }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_synthetic() {
            f.write_str("<synthetic>")
        } else {
            write!(f, "{}:{}", self.line, self.col)
        }
    }
}

macro_rules! codes {
    ($( $variant:ident = $text:literal, $phase:ident ; $blurb:literal );* $(;)?) => {
        /// Stable error code. The string form is what appears in diagnostics and
        /// in `corpus/MANIFEST.tsv`.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub enum Code { $( $variant ),* }

        impl Code {
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self { $( Self::$variant => $text ),* }
            }

            /// Which stage can emit this code. Used by the corpus runner to
            /// check that a rejection happened at the stage it claims.
            #[must_use]
            pub const fn phase(self) -> Phase {
                match self { $( Self::$variant => Phase::$phase ),* }
            }

            /// One-line explanation, surfaced by `cinder explain <code>`.
            #[must_use]
            pub const fn blurb(self) -> &'static str {
                match self { $( Self::$variant => $blurb ),* }
            }

            /// Parse a code from its string form. The corpus manifest is text,
            /// so this is the inverse the runner needs.
            #[must_use]
            pub fn parse(s: &str) -> Option<Self> {
                match s { $( $text => Some(Self::$variant), )* _ => None }
            }

            /// Every code, for `cinder explain --list` and for the test that
            /// asserts the manifest only references codes that exist.
            #[must_use]
            pub const fn all() -> &'static [Self] {
                &[ $( Self::$variant ),* ]
            }
        }
    };
}

/// Pipeline stage a diagnostic originates from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    /// Tokenizing `.cdx`.
    Lex,
    /// Parsing, symbol resolution, encoding.
    Assemble,
    /// Container decode and sealing.
    Load,
    /// Abstract interpretation.
    Verify,
    /// Execution.
    Run,
    /// Snapshot restore.
    Restore,
    /// Journal integrity and replay divergence.
    Journal,
}

codes! {
    // ── lexer ────────────────────────────────────────────────────────────
