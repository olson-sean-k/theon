#![cfg(feature = "geometry-cgmath")]

use arrayvec::ArrayVec;
use cgmath::{self, BaseFloat, BaseNum, Point2, Point3, Vector2, Vector3};
use decorum::{Real, R64};
use num::{Num, NumCast};
use typenum::consts::{U2, U3};

use crate::ops::{Cross, Dot, Interpolate, Map, Reduce, ZipMap};
use crate::space::{
    AffineSpace, Basis, EuclideanSpace, FiniteDimensional, InnerSpace, VectorSpace,
};
use crate::{Category, Converged, FromObjects, IntoObjects};

impl<T> Basis for Vector2<T>
where
    T: Num,
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
    T: Num,
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

impl<T> Category for Vector2<T> {
    type Object = T;
}

impl<T> Category for Vector3<T> {
    type Object = T;
}

impl<T> Converged for Vector2<T>
where
    T: Copy,
{
    fn converged(value: Self::Object) -> Self {
        Vector2::new(value, value)
    }
}

impl<T> Converged for Vector3<T>
where
    T: Copy,
{
    fn converged(value: Self::Object) -> Self {
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

impl<T> FiniteDimensional for Vector2<T> {
    type N = U2;
}

impl<T> FiniteDimensional for Vector3<T> {
    type N = U3;
}

impl<T> FromObjects for Vector2<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(2);
        match (objects.next(), objects.next()) {
            (Some(a), Some(b)) => Some(Vector2::new(a, b)),
            _ => None,
        }
    }
}

impl<T> FromObjects for Vector3<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(3);
        match (objects.next(), objects.next(), objects.next()) {
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

impl<T> IntoObjects for Vector2<T> {
    type Output = ArrayVec<[T; 2]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoObjects for Vector3<T> {
    type Output = ArrayVec<[T; 3]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.x, self.y, self.z])
    }
}

impl<T, U> Map<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        Vector2::new(f(self.x), f(self.y))
    }
}

impl<T, U> Map<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        Vector3::new(f(self.x), f(self.y), f(self.z))
    }
}

impl<T> Reduce<T, T> for Vector2<T>
where
    T: Copy,
{
    fn reduce<F>(self, mut seed: T, mut f: F) -> T
    where
        F: FnMut(T, T) -> T,
    {
        for a in &[self.x, self.y] {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T> Reduce<T, T> for Vector3<T>
where
    T: Copy,
{
    fn reduce<F>(self, mut seed: T, mut f: F) -> T
    where
        F: FnMut(T, T) -> T,
    {
        for a in &[self.x, self.y, self.z] {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T> VectorSpace for Vector2<T>
where
    T: BaseNum + Real,
{
    type Scalar = T;

    fn scalar_component(&self, index: usize) -> Option<&Self::Scalar> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.y),
            _ => None,
        }
    }
}

impl<T> VectorSpace for Vector3<T>
where
    T: BaseNum + Real,
{
    type Scalar = T;

    fn scalar_component(&self, index: usize) -> Option<&Self::Scalar> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.y),
            2 => Some(&self.z),
            _ => None,
        }
    }
}

impl<T, U> ZipMap<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Vector2::new(f(self.x, other.x), f(self.y, other.y))
    }
}

impl<T, U> ZipMap<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Vector3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
    }
}

impl<T> AffineSpace for Point2<T>
where
    T: BaseFloat + BaseNum + Real,
{
    type Translation = Vector2<T>;
}

impl<T> AffineSpace for Point3<T>
where
    T: BaseFloat + BaseNum + Real,
{
    type Translation = Vector3<T>;
}

impl<T> Category for Point2<T> {
    type Object = T;
}

impl<T> Category for Point3<T> {
    type Object = T;
}

impl<T> Converged for Point2<T>
where
    T: Copy,
{
    fn converged(value: Self::Object) -> Self {
        Point2::new(value, value)
    }
}

impl<T> Converged for Point3<T>
where
    T: Copy,
{
    fn converged(value: Self::Object) -> Self {
        Point3::new(value, value, value)
    }
}

impl<T> EuclideanSpace for Point2<T>
where
    T: BaseFloat + BaseNum + Real,
{
    type CoordinateSpace = Vector2<T>;

    fn origin() -> Self {
        <Self as cgmath::EuclideanSpace>::origin()
    }
}

impl<T> EuclideanSpace for Point3<T>
where
    T: BaseFloat + BaseNum + Real,
{
    type CoordinateSpace = Vector3<T>;

    fn origin() -> Self {
        <Self as cgmath::EuclideanSpace>::origin()
    }
}

impl<T> FromObjects for Point2<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(2);
        match (objects.next(), objects.next()) {
            (Some(a), Some(b)) => Some(Point2::new(a, b)),
            _ => None,
        }
    }
}

impl<T> FromObjects for Point3<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(3);
        match (objects.next(), objects.next(), objects.next()) {
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

impl<T> IntoObjects for Point2<T> {
    type Output = ArrayVec<[T; 2]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoObjects for Point3<T> {
    type Output = ArrayVec<[T; 3]>;

    fn into_objects(self) -> Self::Output {
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
        F: FnMut(Self::Object) -> U,
    {
        Point2::new(f(self.x), f(self.y))
    }
}

impl<T, U> Map<U> for Point3<T> {
    type Output = Point3<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        Point3::new(f(self.x), f(self.y), f(self.z))
    }
}

impl<T> Reduce<T, T> for Point2<T>
where
    T: Copy,
{
    fn reduce<F>(self, mut seed: T, mut f: F) -> T
    where
        F: FnMut(T, T) -> T,
    {
        for a in &[self.x, self.y] {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T> Reduce<T, T> for Point3<T>
where
    T: Copy,
{
    fn reduce<F>(self, mut seed: T, mut f: F) -> T
    where
        F: FnMut(T, T) -> T,
    {
        for a in &[self.x, self.y, self.z] {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T, U> ZipMap<U> for Point2<T> {
    type Output = Point2<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Point2::new(f(self.x, other.x), f(self.y, other.y))
    }
}

impl<T, U> ZipMap<U> for Point3<T> {
    type Output = Point3<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Point3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
    }
}
