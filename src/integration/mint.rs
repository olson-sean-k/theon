#![cfg(feature = "geometry-mint")]

// TODO: It is not possible to implement vector space traits for `mint` types,
//       because they require foreign traits on foreign types.
// TODO: Implement as many traits as possible.

use decorum::R64;
use mint::{Point2, Point3, Vector2, Vector3};
use num::traits::{Num, NumCast, One, Zero};
use std::ops::Neg;
use typenum::{U2, U3};

use crate::adjunct::{Adjunct, Converged, Fold, FromItems, IntoItems, Map, ZipMap};
use crate::ops::{Cross, Dot, Interpolate, Project};
use crate::space::{Basis, FiniteDimensional};

impl<T> Adjunct for Vector2<T> {
    type Item = T;
}

impl<T> Adjunct for Vector3<T> {
    type Item = T;
}

impl<T> Basis for Vector2<T>
where
    T: One + Zero,
{
    type Bases = [Self; 2];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector2 {
                x: T::one(),
                y: T::zero(),
            }),
            1 => Some(Vector2 {
                x: T::zero(),
                y: T::one(),
            }),
            _ => None,
        }
    }
}

impl<T> Basis for Vector3<T>
where
    T: One + Zero,
{
    type Bases = [Self; 3];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
            Self::canonical_basis_component(2).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector3 {
                x: T::one(),
                y: T::zero(),
                z: T::zero(),
            }),
            1 => Some(Vector3 {
                x: T::zero(),
                y: T::one(),
                z: T::zero(),
            }),
            2 => Some(Vector3 {
                x: T::zero(),
                y: T::zero(),
                z: T::one(),
            }),
            _ => None,
        }
    }
}

impl<T> Cross for Vector3<T>
where
    T: Copy + Neg<Output = T> + Num,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Vector3 {
            x: (self.y * other.z) - (self.z * other.y),
            y: (self.z * other.x) - (self.x * other.z),
            z: (self.x * other.y) - (self.y * other.x),
        }
    }
}

impl<T> Converged for Vector2<T>
where
    T: Copy,
{
    fn converged(value: T) -> Self {
        Vector2 { x: value, y: value }
    }
}

impl<T> Converged for Vector3<T>
where
    T: Copy,
{
    fn converged(value: T) -> Self {
        Vector3 {
            x: value,
            y: value,
            z: value,
        }
    }
}

impl<T> Dot for Vector2<T>
where
    T: Num,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        (self.x * other.x) + (self.y * other.y)
    }
}

impl<T> Dot for Vector3<T>
where
    T: Num,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    }
}

impl<T> FiniteDimensional for Vector2<T> {
    type N = U2;
}

impl<T> FiniteDimensional for Vector3<T> {
    type N = U3;
}

impl<T> FromItems for Vector2<T> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut items = items.into_iter().take(2);
        match (items.next(), items.next()) {
            (Some(x), Some(y)) => Some(Vector2 { x, y }),
            _ => None,
        }
    }
}

impl<T> FromItems for Vector3<T> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut items = items.into_iter().take(3);
        match (items.next(), items.next(), items.next()) {
            (Some(x), Some(y), Some(z)) => Some(Vector3 { x, y, z }),
            _ => None,
        }
    }
}

impl<T> Interpolate for Vector2<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Vector2 {
            x: crate::lerp(self.x, other.x, f),
            y: crate::lerp(self.y, other.y, f),
        }
    }
}

impl<T> Interpolate for Vector3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Vector3 {
            x: crate::lerp(self.x, other.x, f),
            y: crate::lerp(self.y, other.y, f),
            z: crate::lerp(self.z, other.z, f),
        }
    }
}

impl<T> IntoItems for Vector2<T> {
    type Output = [T; 2];

    fn into_items(self) -> Self::Output {
        [self.x, self.y]
    }
}

impl<T> IntoItems for Vector3<T> {
    type Output = [T; 3];

    fn into_items(self) -> Self::Output {
        [self.x, self.y, self.z]
    }
}

impl<T> Fold for Vector2<T>
where
    T: Copy,
{
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in &[self.x, self.y] {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T> Fold for Vector3<T>
where
    T: Copy,
{
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in &[self.x, self.y, self.z] {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T, U> Map<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector2 {
            x: f(self.x),
            y: f(self.y),
        }
    }
}

impl<T, U> Map<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector3 {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }
}

impl<T> Project<Vector2<T>> for Vector2<T>
where
    T: Copy + Num,
{
    type Output = Vector2<T>;

    fn project(self, other: Vector2<T>) -> Self::Output {
        let n = other.dot(self);
        let d = self.dot(self);
        self.map(|a| a * (n / d))
    }
}

impl<T> Project<Vector3<T>> for Vector3<T>
where
    T: Copy + Num,
{
    type Output = Vector3<T>;

    fn project(self, other: Vector3<T>) -> Self::Output {
        let n = other.dot(self);
        let d = self.dot(self);
        self.map(|a| a * (n / d))
    }
}

impl<T, U> ZipMap<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector2 {
            x: f(self.x, other.x),
            y: f(self.y, other.y),
        }
    }
}

impl<T, U> ZipMap<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector3 {
            x: f(self.x, other.x),
            y: f(self.y, other.y),
            z: f(self.z, other.z),
        }
    }
}

impl<T> Adjunct for Point2<T>
where
    T: Num,
{
    type Item = T;
}

impl<T> Adjunct for Point3<T>
where
    T: Num,
{
    type Item = T;
}

impl<T> Interpolate for Point2<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Point2 {
            x: crate::lerp(self.x, other.x, f),
            y: crate::lerp(self.y, other.y, f),
        }
    }
}

impl<T> Interpolate for Point3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Point3 {
            x: crate::lerp(self.x, other.x, f),
            y: crate::lerp(self.y, other.y, f),
            z: crate::lerp(self.z, other.z, f),
        }
    }
}
