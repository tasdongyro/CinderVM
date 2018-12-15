//! Small hashes for the journal chain, snapshot integrity, and the image
//! checksum. Deliberately not a cryptographic hash: these protect against
//! corruption and replay drift, not adversaries.

/// FNV-1a 64 over bytes.
#[must_use]
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
