pub mod adjunct;
pub mod integration;
mod primitive;
pub mod space;

use std::ops::{Add, Sub};
use typenum::{NonZero, Unsigned, U1};

use crate::space::EuclideanSpace;

pub type Position<T> = <T as AsPosition>::Position;

pub trait Natural: Add<U1> + NonZero + Sub<U1> + Unsigned {}

impl<T> Natural for T where T: Add<U1> + NonZero + Sub<U1> + Unsigned {}

pub trait Increment: Natural {
    type Output: Natural;
}

impl<N> Increment for N
where
    N: Natural,
    <N as Add<U1>>::Output: Natural,
{
    type Output = <N as Add<U1>>::Output;
}

/// Immutable positional data.
///
/// This trait exposes positional data for geometric types along with its
/// mutable variant `AsPositionMut`.
///
/// # Examples
///
/// Exposing immutable positional data for a vertex:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate eudoxus;
/// #
/// use eudoxus::AsPosition;
/// use nalgebra::{Point3, Vector3};
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
/// }
/// ```
pub trait AsPosition {
    type Position: EuclideanSpace;

    fn as_position(&self) -> &Self::Position;
}

/// Mutable positional data.
///
/// This trait exposes positional data for geometric types along with its
/// immutable variant `AsPosition`.
///
/// # Examples
///
/// Exposing mutable positional data for a vertex:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate eudoxus;
/// #
/// use eudoxus::{AsPosition, AsPositionMut};
/// use nalgebra::{Point3, Vector3};
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
/// }
///
/// impl AsPositionMut for Vertex {
///     fn as_position_mut(&mut self) -> &mut Self::Position {
///         &mut self.position
///     }
/// }
/// ```
pub trait AsPositionMut: AsPosition {
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

impl<'a, T> AsPosition for &'a T
where
    T: AsPosition,
    T::Position: EuclideanSpace,
{
    type Position = <T as AsPosition>::Position;

    fn as_position(&self) -> &Self::Position {
        T::as_position(self)
    }
}

impl<'a, T> AsPosition for &'a mut T
where
    T: AsPosition,
    T::Position: EuclideanSpace,
{
    type Position = <T as AsPosition>::Position;

    fn as_position(&self) -> &Self::Position {
        T::as_position(self)
    }
}

impl<'a, T> AsPositionMut for &'a mut T
where
    T: AsPositionMut,
{
    fn as_position_mut(&mut self) -> &mut Self::Position {
        T::as_position_mut(self)
    }
}
