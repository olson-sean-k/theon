#![cfg(feature = "geometry-mint")]

// TODO: It is not possible to implement vector space traits for `mint` types,
//       because they require foreign traits on foreign types.
// TODO: Implement as many traits as possible.

use decorum::R64;
use mint::{Point2, Point3, Vector2, Vector3};
use num::{Num, NumCast};
use std::ops::Neg;

use crate::ops::{Cross, Dot, Interpolate, Map, Project, Reduce, ZipMap};
use crate::space::{Basis, FiniteDimensional};
use crate::{Composite, Converged, FromItems, IntoItems};

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

impl<T> Composite for Vector2<T>
where
    T: Num,
{
    type Item = T;
}

impl<T> Composite for Vector3<T>
where
    T: Num,
{
    type Item = T;
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

impl<T> Composite for Point2<T>
where
    T: Num,
{
    type Item = T;
}

impl<T> Composite for Point3<T>
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
