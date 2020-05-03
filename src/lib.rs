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

pub mod adjunct;
pub mod integration;
pub mod lapack;
pub mod ops;
pub mod query;
pub mod space;

use decorum::R64;
use num::{self, Num, NumCast, One, Zero};

use crate::space::EuclideanSpace;

pub mod prelude {
    //! Re-exports commonly used types and traits.

    pub use crate::query::Intersection as _;
}

pub type Position<T> = <T as AsPosition>::Position;

/// Positional data.
///
/// This trait exposes positional data for geometric types.
///
/// # Examples
///
/// Exposing positional data for a vertex:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate theon;
/// #
/// use nalgebra::{Point3, Vector3};
/// use theon::AsPosition;
///
/// pub struct Vertex {
///     position: Point3<f64>,
///     normal: Vector3<f64>,
/// }
///
/// impl AsPosition for Vertex {
///     type Position = Point3<f64>;
///
///     fn as_position(&self) -> &Self::Position {
///         &self.position
///     }
///
///     fn as_position_mut(&mut self) -> &mut Self::Position {
///         &mut self.position
///     }
/// }
/// ```
pub trait AsPosition {
    type Position: EuclideanSpace;

    fn as_position(&self) -> &Self::Position;

    fn as_position_mut(&mut self) -> &mut Self::Position;

    fn transform<F>(&mut self, mut f: F)
    where
        F: FnMut(&Self::Position) -> Self::Position,
    {
        *self.as_position_mut() = f(self.as_position());
    }

    fn map_position<F>(mut self, f: F) -> Self
    where
        Self: Sized,
        F: FnMut(&Self::Position) -> Self::Position,
    {
        self.transform(f);
        self
    }
}

impl<S> AsPosition for S
where
    S: EuclideanSpace,
{
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }

    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

/// Linearly interpolates between two values.
pub fn lerp<T>(a: T, b: T, f: R64) -> T
where
    T: Num + NumCast,
{
    let f = num::clamp(f, Zero::zero(), One::one());
    let af = <R64 as NumCast>::from(a).unwrap() * (R64::one() - f);
    let bf = <R64 as NumCast>::from(b).unwrap() * f;
    <T as NumCast>::from(af + bf).unwrap()
}
