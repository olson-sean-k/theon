#![cfg(feature = "geometry-cgmath")]

#[doc(hidden)]
pub use cgmath::*;

use approx::AbsDiffEq;
use arrayvec::ArrayVec;
use decorum::{Real, R64};
use num::{Num, NumCast, One, Zero};
use typenum::consts::{U2, U3};

use crate::adjunct::{Adjunct, Converged, Fold, FromItems, IntoItems, Map, Pop, Push, ZipMap};
use crate::ops::{Cross, Dot, Interpolate};
use crate::space::{
    AffineSpace, Basis, DualSpace, EuclideanSpace, FiniteDimensional, InnerSpace, VectorSpace,
};
use crate::{AsPosition, AsPositionMut};

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
    type Bases = ArrayVec<[Self; 2]>;

    fn canonical_basis() -> Self::Bases {
        ArrayVec::from([
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
        ])
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector2::new(T::one(), T::zero())),
            1 => Some(Vector2::new(T::zero(), T::one())),
            _ => None,
        }
    }
}

impl<T> Basis for Vector3<T>
where
    T: One + Zero,
{
    type Bases = ArrayVec<[Self; 3]>;

    fn canonical_basis() -> Self::Bases {
        ArrayVec::from([
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
            Self::canonical_basis_component(2).unwrap(),
        ])
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector3::new(T::one(), T::zero(), T::zero())),
            1 => Some(Vector3::new(T::zero(), T::one(), T::zero())),
            2 => Some(Vector3::new(T::zero(), T::zero(), T::one())),
            _ => None,
        }
    }
}

impl<T> Converged for Vector2<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Vector2::new(value, value)
    }
}

impl<T> Converged for Vector3<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Vector3::new(value, value, value)
    }
}

impl<T> Cross for Vector3<T>
where
    T: BaseFloat,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Self::cross(self, other)
    }
}

impl<T> Dot for Vector2<T>
where
    T: BaseFloat,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        <Self as cgmath::InnerSpace>::dot(self, other)
    }
}

impl<T> Dot for Vector3<T>
where
    T: BaseFloat,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        <Self as cgmath::InnerSpace>::dot(self, other)
    }
}

impl<T> DualSpace for Vector2<T>
where
    T: AbsDiffEq + BaseNum + Real,
{
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl<T> DualSpace for Vector3<T>
where
    T: AbsDiffEq + BaseNum + Real,
{
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl<T> FiniteDimensional for Vector2<T> {
    type N = U2;
}

impl<T> FiniteDimensional for Vector3<T> {
    type N = U3;
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

impl<T> FromItems for Vector2<T> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut items = items.into_iter().take(2);
        match (items.next(), items.next()) {
            (Some(a), Some(b)) => Some(Vector2::new(a, b)),
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
            (Some(a), Some(b), Some(c)) => Some(Vector3::new(a, b, c)),
            _ => None,
        }
    }
}

impl<T> InnerSpace for Vector2<T> where T: BaseFloat + Real {}

impl<T> InnerSpace for Vector3<T> where T: BaseFloat + Real {}

impl<T> Interpolate for Vector2<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        self.zip_map(other, |a, b| crate::lerp(a, b, f))
    }
}

impl<T> Interpolate for Vector3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        self.zip_map(other, |a, b| crate::lerp(a, b, f))
    }
}

impl<T> IntoItems for Vector2<T> {
    type Output = ArrayVec<[T; 2]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoItems for Vector3<T> {
    type Output = ArrayVec<[T; 3]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y, self.z])
    }
}

impl<T, U> Map<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector2::new(f(self.x), f(self.y))
    }
}

impl<T, U> Map<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector3::new(f(self.x), f(self.y), f(self.z))
    }
}

impl<T> Pop for Vector3<T>
where
    T: BaseNum,
{
    type Output = Vector2<T>;

    fn pop(self) -> (Self::Output, T) {
        (self.truncate(), self.z)
    }
}

impl<T> Push for Vector2<T>
where
    T: BaseNum,
{
    type Output = Vector3<T>;

    fn push(self, z: T) -> Self::Output {
        self.extend(z)
    }
}

impl<T> VectorSpace for Vector2<T>
where
    T: AbsDiffEq + BaseNum + Real,
{
    type Scalar = T;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar> {
        match index {
            0 => Some(self.x),
            1 => Some(self.y),
            _ => None,
        }
    }

    fn into_xy(self) -> (Self::Scalar, Self::Scalar) {
        (self.x, self.y)
    }
}

impl<T> VectorSpace for Vector3<T>
where
    T: AbsDiffEq + BaseNum + Real,
{
    type Scalar = T;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar> {
        match index {
            0 => Some(self.x),
            1 => Some(self.y),
            2 => Some(self.z),
            _ => None,
        }
    }

    fn into_xyz(self) -> (Self::Scalar, Self::Scalar, Self::Scalar) {
        (self.x, self.y, self.z)
    }
}

impl<T, U> ZipMap<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector2::new(f(self.x, other.x), f(self.y, other.y))
    }
}

impl<T, U> ZipMap<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
    }
}

impl<T> Adjunct for Point2<T> {
    type Item = T;
}

impl<T> Adjunct for Point3<T> {
    type Item = T;
}

impl<T> AffineSpace for Point2<T>
where
    T: AbsDiffEq + BaseNum + Real,
{
    type Translation = Vector2<T>;
}

impl<T> AffineSpace for Point3<T>
where
    T: AbsDiffEq + BaseNum + Real,
{
    type Translation = Vector3<T>;
}

impl<T> AsPosition for Point2<T>
where
    Self: EuclideanSpace,
    T: BaseNum,
{
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl<T> AsPosition for Point3<T>
where
    Self: EuclideanSpace,
    T: BaseNum,
{
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl<T> AsPositionMut for Point2<T>
where
    Self: EuclideanSpace,
    T: BaseNum,
{
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl<T> AsPositionMut for Point3<T>
where
    Self: EuclideanSpace,
    T: BaseNum,
{
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl<T> Converged for Point2<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Point2::new(value, value)
    }
}

impl<T> Converged for Point3<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Point3::new(value, value, value)
    }
}

impl<T> EuclideanSpace for Point2<T>
where
    T: BaseFloat + Real,
{
    type CoordinateSpace = Vector2<T>;

    fn origin() -> Self {
        <Self as cgmath::EuclideanSpace>::origin()
    }
}

impl<T> EuclideanSpace for Point3<T>
where
    T: BaseFloat + Real,
{
    type CoordinateSpace = Vector3<T>;

    fn origin() -> Self {
        <Self as cgmath::EuclideanSpace>::origin()
    }
}

impl<T> FiniteDimensional for Point2<T> {
    type N = U2;
}

impl<T> FiniteDimensional for Point3<T> {
    type N = U3;
}

impl<T> Fold for Point2<T>
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

impl<T> Fold for Point3<T>
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

impl<T> FromItems for Point2<T> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut items = items.into_iter().take(2);
        match (items.next(), items.next()) {
            (Some(a), Some(b)) => Some(Point2::new(a, b)),
            _ => None,
        }
    }
}

impl<T> FromItems for Point3<T> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut items = items.into_iter().take(3);
        match (items.next(), items.next(), items.next()) {
            (Some(a), Some(b), Some(c)) => Some(Point3::new(a, b, c)),
            _ => None,
        }
    }
}

impl<T> Interpolate for Point2<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        self.zip_map(other, |a, b| crate::lerp(a, b, f))
    }
}

impl<T> IntoItems for Point2<T> {
    type Output = ArrayVec<[T; 2]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoItems for Point3<T> {
    type Output = ArrayVec<[T; 3]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y, self.z])
    }
}

impl<T> Interpolate for Point3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        self.zip_map(other, |a, b| crate::lerp(a, b, f))
    }
}

impl<T, U> Map<U> for Point2<T> {
    type Output = Point2<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Point2::new(f(self.x), f(self.y))
    }
}

impl<T, U> Map<U> for Point3<T> {
    type Output = Point3<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Point3::new(f(self.x), f(self.y), f(self.z))
    }
}

impl<T> Pop for Point3<T> {
    type Output = Point2<T>;

    fn pop(self) -> (Self::Output, T) {
        let Point3 { x, y, z } = self;
        (Point2::new(x, y), z)
    }
}

impl<T> Push for Point2<T> {
    type Output = Point3<T>;

    fn push(self, z: T) -> Self::Output {
        let Point2 { x, y } = self;
        Point3::new(x, y, z)
    }
}

impl<T, U> ZipMap<U> for Point2<T> {
    type Output = Point2<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Point2::new(f(self.x, other.x), f(self.y, other.y))
    }
}

impl<T, U> ZipMap<U> for Point3<T> {
    type Output = Point3<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Point3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
    }
}
