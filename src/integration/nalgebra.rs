#![cfg(feature = "nalgebra")]

use approx::AbsDiffEq;
use decorum::R64;
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::{
    DimName, DimNameAdd, DimNameDiff, DimNameMax, DimNameMaximum, DimNameMin, DimNameSub,
    DimNameSum, ToTypenum, U1,
};
use nalgebra::base::{
    Matrix2, Matrix3, OMatrix, OVector, RowVector2, RowVector3, Scalar, Vector2, Vector3, Vector4,
};
use nalgebra::geometry::{OPoint, Point2, Point3};
use num::traits::real::Real;
use num::traits::{Num, NumCast, One, Zero};
use std::ops::{AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use typenum::NonZero;

use crate::adjunct::{
    Adjunct, Converged, Extend, Fold, FromItems, IntoItems, Map, Truncate, ZipMap,
};
use crate::ops::{Cross, Dot, Interpolate, MulMN};
use crate::space::{
    AffineSpace, Basis, DualSpace, EuclideanSpace, FiniteDimensional, Homogeneous, InnerSpace,
    Matrix, SquareMatrix, VectorSpace,
};
use crate::{AsPosition, AsPositionMut};

impl<T, R, C> Adjunct for OMatrix<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
{
    type Item = T;
}

impl<T, D> Basis for OVector<T, D>
where
    T: One + Scalar + Zero,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    Self: FiniteDimensional,
{
    type Bases = Vec<Self>;

    fn canonical_basis() -> Self::Bases {
        (0..D::dim())
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

impl<T, R, C> Converged for OMatrix<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
{
    fn converged(value: Self::Item) -> Self {
        Self::from_element(value)
    }
}

impl<T> Cross for Vector3<T>
where
    // TODO: Is the `Copy` requirement too strict? See `Fold` implementation.
    T: Copy + Num + Scalar,
    <<T as Mul>::Output as Sub>::Output: Neg<Output = T>,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        let [ax, ay, az]: [T; 3] = self.into();
        let [bx, by, bz]: [T; 3] = other.into();
        Vector3::new(
            (ay * bz) - (az * by),
            (az * bx) - (ax * bz),
            (ax * by) - (ay * bx),
        )
    }
}

impl<T, D> Dot for OVector<T, D>
where
    T: AddAssign + MulAssign + Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        nalgebra::Matrix::dot(&self, &other)
    }
}

impl<T, R, C> DualSpace for OMatrix<T, R, C>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
    R: DimName + DimNameMin<C, Output = U1>,
    C: DimName + DimNameMin<R, Output = U1>,
    DefaultAllocator: Allocator<R, C> + Allocator<C, R>,
    OMatrix<T, C, R>: Copy + FiniteDimensional<N = <Self as FiniteDimensional>::N>,
    Self: Copy + FiniteDimensional,
{
    type Dual = OMatrix<T, C, R>;

    fn transpose(self) -> Self::Dual {
        nalgebra::Matrix::transpose(&self)
    }
}

impl<T, D> Extend<OVector<T, DimNameSum<D, U1>>> for OVector<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar,
    D: DimName + DimNameAdd<U1>,
    DefaultAllocator: Allocator<D> + Allocator<DimNameSum<D, U1>>,
{
    fn extend(self, x: T) -> OVector<T, DimNameSum<D, U1>> {
        OVector::<_, DimNameSum<D, _>>::from_iterator(self.into_iter().cloned().chain(Some(x)))
    }
}

impl<T, R, C> FiniteDimensional for OMatrix<T, R, C>
where
    T: Scalar,
    R: DimName + DimNameMax<C> + DimNameMin<C, Output = U1>,
    C: DimName,
    DimNameMaximum<R, C>: ToTypenum,
    <DimNameMaximum<R, C> as ToTypenum>::Typenum: NonZero,
    DefaultAllocator: Allocator<R, C>,
{
    type N = <DimNameMaximum<R, C> as ToTypenum>::Typenum;
}

impl<T, R, C> Fold for OMatrix<T, R, C>
where
    // TODO: Re-examine adjunct traits that take items by value.
    T: Clone + Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
{
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in self.iter() {
            seed = f(seed, a.clone());
        }
        seed
    }
}

impl<T, R, C> FromItems for OMatrix<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
{
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        Some(Self::from_iterator(items))
    }
}

impl<T> Homogeneous for Vector2<T>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    type ProjectiveSpace = Vector3<T>;
}

impl<T> Homogeneous for Vector3<T>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    type ProjectiveSpace = Vector4<T>;
}

impl<T, D> InnerSpace for OVector<T, D>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    Self: Copy,
{
}

impl<T, R, C> Interpolate for OMatrix<T, R, C>
where
    T: Num + NumCast + Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        OMatrix::<T, R, C>::zip_map(&self, &other, |a, b| crate::lerp(a, b, f))
    }
}

impl<T> IntoItems for Vector2<T>
where
    T: Scalar,
{
    type Output = [T; 2];

    fn into_items(self) -> Self::Output {
        self.into()
    }
}

impl<T> IntoItems for Vector3<T>
where
    T: Scalar,
{
    type Output = [T; 3];

    fn into_items(self) -> Self::Output {
        self.into()
    }
}

impl<T, U, R, C> Map<U> for OMatrix<T, R, C>
where
    T: Scalar,
    U: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OMatrix<U, R, C>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        OMatrix::<T, R, C>::map(&self, f)
    }
}

// TODO: Use a (more) generic implementation.
impl<T> Matrix for Matrix2<T>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    type Row = RowVector2<T>;
    type Column = Vector2<T>;
    type Transpose = Self;

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
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    type Row = RowVector3<T>;
    type Column = Vector3<T>;
    type Transpose = Self;

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
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    type Output = Matrix2<T>;

    fn mul_mn(self, other: Matrix2<T>) -> <Self as MulMN<Matrix2<T>>>::Output {
        self * other
    }
}

impl<T> MulMN<Matrix3<T>> for Matrix3<T>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    type Output = Matrix3<T>;

    fn mul_mn(self, other: Matrix3<T>) -> <Self as MulMN<Matrix3<T>>>::Output {
        self * other
    }
}

impl<T> SquareMatrix for Matrix2<T>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    fn multiplicative_identity() -> Self {
        nalgebra::Matrix2::<T>::identity()
    }
}

impl<T> SquareMatrix for Matrix3<T>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
{
    fn multiplicative_identity() -> Self {
        nalgebra::Matrix3::<T>::identity()
    }
}

impl<T, D> Truncate<OVector<T, DimNameDiff<D, U1>>> for OVector<T, D>
where
    T: Real + Scalar,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D> + Allocator<DimNameDiff<D, U1>>,
{
    fn truncate(self) -> (OVector<T, DimNameDiff<D, U1>>, T) {
        let n = self.len();
        let x = *self.get(n - 1).unwrap();
        (
            OVector::<_, DimNameDiff<D, _>>::from_iterator(self.into_iter().take(n - 1).cloned()),
            x,
        )
    }
}

// TODO: This is too general. Only "linear" types should implement this.
impl<T, R, C> VectorSpace for OMatrix<T, R, C>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
    Self: Copy,
{
    type Scalar = T;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar> {
        self.get(index).cloned()
    }
}

impl<T, U, R, C> ZipMap<U> for OMatrix<T, R, C>
where
    T: Scalar,
    U: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OMatrix<U, R, C>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        OMatrix::<T, R, C>::zip_map(&self, &other, f)
    }
}

impl<T, D> Adjunct for OPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    type Item = T;
}

impl<T, D> AffineSpace for OPoint<T, D>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar + SubAssign,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Copy,
{
    type Translation = OVector<T, D>;
}

impl<T, D> AsPosition for OPoint<T, D>
where
    Self: EuclideanSpace,
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl<T, D> AsPositionMut for OPoint<T, D>
where
    Self: EuclideanSpace,
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl<T, D> Converged for OPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    fn converged(value: Self::Item) -> Self {
        OPoint::from(OVector::<T, D>::converged(value))
    }
}

impl<T, D> Extend<OPoint<T, DimNameSum<D, U1>>> for OPoint<T, D>
where
    T: Scalar,
    D: DimName + DimNameAdd<U1>,
    DefaultAllocator: Allocator<D> + Allocator<DimNameSum<D, U1>>,
    OVector<T, D>: Adjunct<Item = T> + Extend<OVector<T, DimNameSum<D, U1>>>,
{
    fn extend(self, x: T) -> OPoint<T, DimNameSum<D, U1>> {
        self.coords.extend(x).into()
    }
}

impl<T, D> EuclideanSpace for OPoint<T, D>
where
    T: AbsDiffEq + AddAssign + MulAssign + NumCast + Real + Scalar + SubAssign,
    D: DimName + ToTypenum,
    D::Typenum: NonZero,
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Copy,
    OVector<T, D>: FiniteDimensional<N = Self::N>,
{
    type CoordinateSpace = OVector<T, D>;

    fn origin() -> Self {
        OPoint::<T, D>::origin()
    }

    fn into_coordinates(self) -> Self::CoordinateSpace {
        self.coords
    }
}

impl<T, D> FiniteDimensional for OPoint<T, D>
where
    T: Scalar,
    D: DimName + ToTypenum,
    D::Typenum: NonZero,
    DefaultAllocator: Allocator<D>,
{
    type N = D::Typenum;
}

impl<T, D> Fold for OPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    fn fold<U, F>(self, seed: U, f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        self.coords.fold(seed, f)
    }
}

impl<T, D> FromItems for OPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        Some(OPoint::from(OVector::from_iterator(items)))
    }
}

impl<T, D> Interpolate for OPoint<T, D>
where
    T: Num + NumCast + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        OPoint::from(self.coords.lerp(other.coords, f))
    }
}

impl<T> IntoItems for Point2<T>
where
    T: Scalar,
{
    type Output = [T; 2];

    fn into_items(self) -> Self::Output {
        self.coords.into()
    }
}

impl<T> IntoItems for Point3<T>
where
    T: Scalar,
{
    type Output = [T; 3];

    fn into_items(self) -> Self::Output {
        self.coords.into()
    }
}

impl<T, U, D> Map<U> for OPoint<T, D>
where
    T: Scalar,
    U: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    type Output = OPoint<U, D>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        OPoint::from(self.coords.map(f))
    }
}

impl<T, D> Truncate<OPoint<T, DimNameDiff<D, U1>>> for OPoint<T, D>
where
    T: Scalar,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D> + Allocator<DimNameDiff<D, U1>>,
    OVector<T, D>: Adjunct<Item = T> + Truncate<OVector<T, DimNameDiff<D, U1>>>,
{
    fn truncate(self) -> (OPoint<T, DimNameDiff<D, U1>>, T) {
        let (vector, x) = self.coords.truncate();
        (vector.into(), x)
    }
}

impl<T, U, D> ZipMap<U> for OPoint<T, D>
where
    T: Scalar,
    U: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    type Output = OPoint<U, D>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        OPoint::from(self.coords.zip_map(other.coords, f))
    }
}
