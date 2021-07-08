//! Integration of external crates and foreign types.
//!
//! This module provides implementations of Eudoxus traits for foreign types.
//! Integrated crates are re-exported within a sub-module, which can be used to
//! avoid versioning errors.
//!
//! Re-exported types are hidden in Eudoxus' documentation. Refer to the
//! documentation for integrated crates at the corresponding version.

// TODO: Re-enable these modules.
// Feature modules. These are empty unless Cargo features are enabled.
pub mod cgmath;
//pub mod glam;
//pub mod mint;
//pub mod nalgebra;
//pub mod ultraviolet;
