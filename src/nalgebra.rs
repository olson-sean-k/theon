#![cfg(feature = "geometry-nalgebra")]

use alga::general::Ring;
use arrayvec::ArrayVec;
use decorum::{Real, R64};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::DimName;
use nalgebra::core::Matrix;
use nalgebra::{Point, Point2, Point3, Scalar, Vector2, Vector3, VectorN};
use num::{Num, NumCast};
use std::ops::{AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use typenum::{NonZero, Unsigned};

use crate::convert::{FromObjects, IntoObjects};
use crate::ops::{Cross, Dot, Interpolate, Map, Reduce, ZipMap};
use crate::space::{
    AffineSpace, Basis, EuclideanSpace, FiniteDimensional, InnerSpace, VectorSpace,
};
use crate::{Category, Converged};

impl<T, D> Basis for VectorN<T, D>
where
    T: Num + Scalar,
    D: DimName,
    D::Value: NonZero + Unsigned,
    DefaultAllocator: Allocator<T, D>,
{
    type Bases = Vec<Self>;

    fn canonical_basis() -> Self::Bases {
        (0..D::dim())
            .into_iter()
            .map(|dimension| {
                let mut basis = Self::zeros();
                *basis.get_mut(dimension).unwrap() = T::one();
                basis
            })
            .collect()
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        let mut basis = Self::zeros();
        if let Some(component) = basis.get_mut(index) {
            *component = T::one();
            Some(basis)
        }
        else {
            None
        }
    }
}

impl<T, D> Category for VectorN<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Object = T;
}

impl<T, D> Converged for VectorN<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn converged(value: Self::Object) -> Self {
        VectorN::repeat(value)
    }
}

impl<T> Cross for Vector3<T>
where
    T: Num + Ring + Scalar,
    <<T as Mul>::Output as Sub>::Output: Neg<Output = T>,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Matrix::cross(&self, &other)
    }
}

impl<T, D> Dot for VectorN<T, D>
where
    T: AddAssign + MulAssign + Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        Matrix::dot(&self, &other)
    }
}

impl<T, D> FiniteDimensional for VectorN<T, D>
where
    T: Scalar,
    D: DimName,
    D::Value: NonZero + Unsigned,
    DefaultAllocator: Allocator<T, D>,
{
    type N = D::Value;
}

impl<T, D> FromObjects for VectorN<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        Some(VectorN::from_iterator(objects))
    }
}

impl<T, D> InnerSpace for VectorN<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar,
    D: DimName,
    D::Value: NonZero + Unsigned,
    DefaultAllocator: Allocator<T, D>,
    Self: Copy,
{
}

impl<T, D> Interpolate for VectorN<T, D>
where
    T: Num + NumCast + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        VectorN::<T, D>::zip_map(&self, &other, |a, b| crate::lerp(a, b, f))
    }
}

impl<T> IntoObjects for Vector2<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 2]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoObjects for Vector3<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 3]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.x, self.y, self.z])
    }
}

impl<T, U, D> Map<U> for VectorN<T, D>
where
    T: Num + Scalar,
    U: Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<U, D>,
{
    type Output = VectorN<U, D>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        VectorN::<T, D>::map(&self, f)
    }
}

impl<T, U, D> Reduce<T, U> for VectorN<T, D>
where
    T: Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        for a in self.iter() {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T, D> VectorSpace for VectorN<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar,
    D: DimName,
    D::Value: NonZero + Unsigned,
    DefaultAllocator: Allocator<T, D>,
    Self: Copy,
{
    type Scalar = T;

    fn scalar_component(&self, index: usize) -> Option<&Self::Scalar> {
        self.get(index)
    }
}

impl<T, U, D> ZipMap<U> for VectorN<T, D>
where
    T: Num + Scalar,
    U: Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<U, D>,
{
    type Output = VectorN<U, D>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        VectorN::<T, D>::zip_map(&self, &other, f)
    }
}

impl<T, D> AffineSpace for Point<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar + SubAssign,
    D: DimName,
    D::Value: NonZero + Unsigned,
    DefaultAllocator: Allocator<T, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Copy,
{
    type Translation = VectorN<T, D>;
}

impl<T, D> Category for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Object = T;
}

impl<T, D> Converged for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn converged(value: Self::Object) -> Self {
        Point::from(VectorN::<T, D>::converged(value))
    }
}

impl<T, D> EuclideanSpace for Point<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar + SubAssign,
    D: DimName,
    D::Value: NonZero + Unsigned,
    DefaultAllocator: Allocator<T, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Copy,
{
    type CoordinateSpace = VectorN<T, D>;

    fn origin() -> Self {
        Point::<T, D>::origin()
    }

    fn coordinates(&self) -> Self::CoordinateSpace {
        self.coords.clone()
    }
}

impl<T, D> FromObjects for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        Some(Point::from(VectorN::from_iterator(objects)))
    }
}

impl<T, D> Interpolate for Point<T, D>
where
    T: Num + NumCast + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Point::from(self.coords.lerp(other.coords, f))
    }
}

impl<T> IntoObjects for Point2<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 2]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoObjects for Point3<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 3]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.x, self.y, self.z])
    }
}

impl<T, U, D> Map<U> for Point<T, D>
where
    T: Num + Scalar,
    U: Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<U, D>,
{
    type Output = Point<U, D>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        Point::from(self.coords.map(f))
    }
}

impl<T, U, D> Reduce<T, U> for Point<T, D>
where
    T: Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn reduce<F>(self, seed: U, f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        self.coords.reduce(seed, f)
    }
}

impl<T, U, D> ZipMap<U> for Point<T, D>
where
    T: Num + Scalar,
    U: Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<U, D>,
{
    type Output = Point<U, D>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Point::from(self.coords.zip_map(other.coords, f))
    }
}
