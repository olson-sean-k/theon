#![cfg(feature = "geometry-nalgebra")]

use arrayvec::ArrayVec;
use decorum::{Real, R64};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::{DimName, DimNameMax, DimNameMaximum, DimNameMin, U1};
use nalgebra::{
    Matrix2, Matrix3, MatrixMN, Point, Point2, Point3, RowVector2, RowVector3, RowVectorN, Scalar,
    Vector2, Vector3, VectorN,
};
use num::{Num, NumCast, One, Zero};
use std::ops::{AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use typenum::NonZero;

use crate::ops::{Cross, Dot, Interpolate, Map, MulMN, Reduce, ZipMap};
use crate::space::{
    AffineSpace, Basis, DualSpace, EuclideanSpace, FiniteDimensional, InnerSpace, Matrix,
    VectorSpace,
};
use crate::{Composite, Converged, FromItems, IntoItems};

impl<T, D> Basis for VectorN<T, D>
where
    T: One + Scalar + Zero,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    Self: FiniteDimensional,
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

impl<T, R, C> Composite for MatrixMN<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    type Item = T;
}

impl<T, R, C> Converged for MatrixMN<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn converged(value: Self::Item) -> Self {
        Self::from_element(value)
    }
}

impl<T> Cross for Vector3<T>
where
    T: Num + Scalar,
    <<T as Mul>::Output as Sub>::Output: Neg<Output = T>,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Vector3::new(
            (self.y * other.z) - (self.z * other.y),
            (self.z * other.x) - (self.x * other.z),
            (self.x * other.y) - (self.y * other.x),
        )
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
        nalgebra::Matrix::dot(&self, &other)
    }
}

impl<T, D> DualSpace for RowVectorN<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D, U1>,
    DefaultAllocator: Allocator<T, U1, D>,
    VectorN<T, D>: Copy + FiniteDimensional<N = Self::N>,
    Self: Copy + FiniteDimensional,
{
    type Dual = VectorN<T, D>;

    fn transpose(self) -> Self::Dual {
        nalgebra::Matrix::transpose(&self)
    }
}

// Implemented for row and column vectors only.
impl<T, R, C> FiniteDimensional for MatrixMN<T, R, C>
where
    T: Scalar,
    R: DimName + DimNameMax<C> + DimNameMin<C, Output = U1>,
    <DimNameMaximum<R, C> as DimName>::Value: NonZero,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    type N = <DimNameMaximum<R, C> as DimName>::Value;
}

impl<T, R, C> FromItems for MatrixMN<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        Some(Self::from_iterator(items))
    }
}

impl<T, D> InnerSpace for VectorN<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    Self: Copy,
{
}

impl<T, R, C> Interpolate for MatrixMN<T, R, C>
where
    T: Num + NumCast + Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        MatrixMN::<T, R, C>::zip_map(&self, &other, |a, b| crate::lerp(a, b, f))
    }
}

impl<T> IntoItems for Vector2<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 2]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoItems for Vector3<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 3]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y, self.z])
    }
}

impl<T, U, R, C> Map<U> for MatrixMN<T, R, C>
where
    T: Scalar,
    U: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
    DefaultAllocator: Allocator<U, R, C>,
{
    type Output = MatrixMN<U, R, C>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        MatrixMN::<T, R, C>::map(&self, f)
    }
}

// TODO: Use a (more) generic implementation.
impl<T> Matrix for Matrix2<T>
where
    T: AddAssign + MulAssign + Real + Scalar,
{
    type Row = RowVector2<T>;
    type Column = Vector2<T>;
    type Transpose = Self;

    fn row_count() -> usize {
        Self::Column::dimensions()
    }

    fn column_count() -> usize {
        Self::Row::dimensions()
    }

    fn row_component(&self, index: usize) -> Option<Self::Row> {
        if index < <Self as Matrix>::row_count() {
            Some(nalgebra::Matrix::row(self, index).into_owned())
        }
        else {
            None
        }
    }

    fn column_component(&self, index: usize) -> Option<Self::Column> {
        if index < <Self as Matrix>::column_count() {
            Some(nalgebra::Matrix::column(self, index).into_owned())
        }
        else {
            None
        }
    }

    fn transpose(self) -> Self::Transpose {
        nalgebra::Matrix::transpose(&self)
    }
}

impl<T> Matrix for Matrix3<T>
where
    T: AddAssign + MulAssign + Real + Scalar,
{
    type Row = RowVector3<T>;
    type Column = Vector3<T>;
    type Transpose = Self;

    fn row_count() -> usize {
        Self::Column::dimensions()
    }

    fn column_count() -> usize {
        Self::Row::dimensions()
    }

    fn row_component(&self, index: usize) -> Option<Self::Row> {
        if index < <Self as Matrix>::row_count() {
            Some(nalgebra::Matrix::row(self, index).into_owned())
        }
        else {
            None
        }
    }

    fn column_component(&self, index: usize) -> Option<Self::Column> {
        if index < <Self as Matrix>::column_count() {
            Some(nalgebra::Matrix::column(self, index).into_owned())
        }
        else {
            None
        }
    }

    fn transpose(self) -> Self::Transpose {
        nalgebra::Matrix::transpose(&self)
    }
}

// TODO: Use a (more) generic implementation.
impl<T> MulMN<Matrix2<T>> for Matrix2<T>
where
    T: AddAssign + MulAssign + Real + Scalar,
{
    type Output = Matrix2<T>;

    // TODO: Proxy to the `Mul` implementation, which should be much faster.
}

impl<T, U, R, C> Reduce<U> for MatrixMN<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in self.iter() {
            seed = f(seed, *a);
        }
        seed
    }
}

impl<T, R, C> VectorSpace for MatrixMN<T, R, C>
where
    T: AddAssign + MulAssign + Real + Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
    Self: Copy,
{
    type Scalar = T;

    fn scalar_component(&self, index: usize) -> Option<&Self::Scalar> {
        self.get(index)
    }

    fn multiplicative_identity() -> Self {
        Self::identity()
    }
}

impl<T, U, R, C> ZipMap<U> for MatrixMN<T, R, C>
where
    T: Scalar,
    U: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
    DefaultAllocator: Allocator<U, R, C>,
{
    type Output = MatrixMN<U, R, C>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        MatrixMN::<T, R, C>::zip_map(&self, &other, f)
    }
}

impl<T, D> AffineSpace for Point<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar + SubAssign,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Copy,
{
    type Translation = VectorN<T, D>;
}

impl<T, D> Composite for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Item = T;
}

impl<T, D> Converged for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn converged(value: Self::Item) -> Self {
        Point::from(VectorN::<T, D>::converged(value))
    }
}

impl<T, D> EuclideanSpace for Point<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar + SubAssign,
    D: DimName,
    D::Value: NonZero,
    DefaultAllocator: Allocator<T, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Copy,
    VectorN<T, D>: FiniteDimensional<N = Self::N>,
{
    type CoordinateSpace = VectorN<T, D>;

    fn origin() -> Self {
        Point::<T, D>::origin()
    }

    fn coordinates(&self) -> Self::CoordinateSpace {
        self.coords.clone()
    }
}

impl<T, D> FiniteDimensional for Point<T, D>
where
    T: Scalar,
    D: DimName,
    D::Value: NonZero,
    DefaultAllocator: Allocator<T, D>,
{
    type N = D::Value;
}

impl<T, D> FromItems for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        Some(Point::from(VectorN::from_iterator(items)))
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

impl<T> IntoItems for Point2<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 2]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y])
    }
}

impl<T> IntoItems for Point3<T>
where
    T: Scalar,
{
    type Output = ArrayVec<[T; 3]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.x, self.y, self.z])
    }
}

impl<T, U, D> Map<U> for Point<T, D>
where
    T: Scalar,
    U: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<U, D>,
{
    type Output = Point<U, D>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Point::from(self.coords.map(f))
    }
}

impl<T, U, D> Reduce<U> for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn reduce<F>(self, seed: U, f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        self.coords.reduce(seed, f)
    }
}

impl<T, U, D> ZipMap<U> for Point<T, D>
where
    T: Scalar,
    U: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<U, D>,
{
    type Output = Point<U, D>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Point::from(self.coords.zip_map(other.coords, f))
    }
}
