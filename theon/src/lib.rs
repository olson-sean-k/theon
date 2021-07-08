//! **Theon** abstracts Euclidean spaces and geometric queries with support for
//! popular linear algebra and spatial crates in the Rust ecosystem.

// TODO: Require the `geometry-nalgebra` feature for doc tests.
//       See https://github.com/rust-lang/rust/issues/43781

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/olson-sean-k/theon/master/doc/theon-favicon.ico"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/olson-sean-k/theon/master/doc/theon.svg?sanitize=true"
)]

pub mod lapack;
pub mod ops;
pub mod query;

pub use eudoxus::adjunct;
pub use eudoxus::space;
pub use eudoxus::{AsPosition, AsPositionMut, Position};

pub mod prelude {
    //! Re-exports commonly used types and traits.

    pub use crate::query::Intersection as _;
}
