pub mod convert;
#[macro_use]
pub mod ops;
pub mod query;
pub mod space;

// Feature modules. These are empty unless `geometry-*` features are enabled.
mod cgmath;
mod mint;
mod nalgebra;

use std::cmp::Ordering;

use decorum::R64;
use num::{self, Num, NumCast, One, Zero};

pub mod prelude {
    pub use crate::query::{Intersection as _, ReciprocalIntersection as _};
    pub use crate::Lattice as _;
}

pub trait Category {
    type Object;
}

impl<T> Category for (T, T) {
    type Object = T;
}

impl<T> Category for (T, T, T) {
    type Object = T;
}

pub trait Converged: Category {
    fn converged(value: Self::Object) -> Self;
}

impl<T> Converged for (T, T)
where
    T: Clone,
{
    fn converged(value: Self::Object) -> Self {
        (value.clone(), value)
    }
}

impl<T> Converged for (T, T, T)
where
    T: Clone,
{
    fn converged(value: Self::Object) -> Self {
        (value.clone(), value.clone(), value)
    }
}

pub trait Lattice: PartialOrd + Sized {
    fn meet(&self, other: &Self) -> Self;

    fn join(&self, other: &Self) -> Self;

    fn meet_join(&self, other: &Self) -> (Self, Self) {
        (self.meet(other), self.join(other))
    }

    fn partial_min<'a>(&'a self, other: &'a Self) -> Option<&'a Self> {
        match self.partial_cmp(other) {
            Some(Ordering::Greater) => Some(other),
            Some(_) => Some(self),
            None => None,
        }
    }

    fn partial_max<'a>(&'a self, other: &'a Self) -> Option<&'a Self> {
        match self.partial_cmp(other) {
            Some(Ordering::Less) => Some(other),
            Some(_) => Some(self),
            None => None,
        }
    }

    fn partial_ordered_pair<'a>(&'a self, other: &'a Self) -> Option<(&'a Self, &'a Self)> {
        match self.partial_cmp(other) {
            Some(Ordering::Less) => Some((self, other)),
            Some(_) => Some((other, self)),
            None => None,
        }
    }

    fn partial_clamp<'a>(&'a self, min: &'a Self, max: &'a Self) -> Option<&'a Self> {
        let _ = (min, max);
        unimplemented!() // TODO:
    }
}

impl<T> Lattice for T
where
    T: Copy + PartialOrd + Sized,
{
    fn meet(&self, other: &Self) -> Self {
        if *self <= *other {
            *self
        }
        else {
            *other
        }
    }

    fn join(&self, other: &Self) -> Self {
        if *self >= *other {
            *self
        }
        else {
            *other
        }
    }
}

pub fn lerp<T>(a: T, b: T, f: R64) -> T
where
    T: Num + NumCast,
{
    let f = num::clamp(f, Zero::zero(), One::one());
    let af = <R64 as NumCast>::from(a).unwrap() * (R64::one() - f);
    let bf = <R64 as NumCast>::from(b).unwrap() * f;
    <T as NumCast>::from(af + bf).unwrap()
}

fn partial_min<T>(a: T, b: T) -> T
where
    T: Copy + Lattice,
{
    *a.partial_min(&b).unwrap()
}

fn partial_max<T>(a: T, b: T) -> T
where
    T: Copy + Lattice,
{
    *a.partial_max(&b).unwrap()
}
