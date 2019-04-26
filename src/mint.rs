#![cfg(feature = "geometry-mint")]

// TODO: It is not possible to implement vector space traits for `mint` types,
//       because they require foreign traits on foreign types.
// TODO: Implement as many traits as possible.

use decorum::R64;
use mint::{Point2, Point3, Vector2, Vector3};
use num::{Num, NumCast};
use std::ops::Neg;

use crate::convert::{FromObjects, IntoObjects};
use crate::ops::{Cross, Dot, Interpolate, Map, Project, Reduce, ZipMap};
use crate::space::{Basis, FiniteDimensional};
use crate::{Category, Converged};

impl<T> Cross for Vector3<T>
where
    T: Clone + Neg<Output = T> + Num,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Vector3 {
            x: (self.y.clone() * other.z.clone()) - (self.z.clone() * other.y.clone()),
            y: -((self.x.clone() * other.z.clone()) - (self.z * other.x.clone())),
            z: (self.x.clone() * other.y.clone()) - (self.y * other.x),
        }
    }
}

impl<T> Category for Vector2<T>
where
    T: Num,
{
    type Object = T;
}

impl<T> Category for Vector3<T>
where
    T: Num,
{
    type Object = T;
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

impl<T> Category for Point2<T>
where
    T: Num,
{
    type Object = T;
}

impl<T> Category for Point3<T>
where
    T: Num,
{
    type Object = T;
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
