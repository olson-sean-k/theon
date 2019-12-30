#![cfg(feature = "geometry-nalgebra")]

use arrayvec::ArrayVec;
use decorum::{Real, R64};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::{
    DimName, DimNameAdd, DimNameDiff, DimNameMax, DimNameMaximum, DimNameMin, DimNameSub,
    DimNameSum, U1,
};
use nalgebra::{
    Matrix2, Matrix3, MatrixMN, Point, Point2, Point3, RowVector2, RowVector3, Scalar, Vector2,
    Vector3, VectorN,
};
use num::{Num, NumCast, One, Zero};
use std::ops::{AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use typenum::NonZero;

use crate::ops::{Cross, Dot, Fold, Interpolate, Map, MulMN, Pop, Push, ZipMap};
use crate::space::{
    AffineSpace, Basis, DualSpace, EuclideanSpace, FiniteDimensional, InnerSpace, Matrix,
    SquareMatrix, VectorSpace,
};
use crate::{Converged, FromItems, IntoItems, Series};

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

impl<T, R, C> DualSpace for MatrixMN<T, R, C>
where
    T: AddAssign + MulAssign + Real + Scalar,
    R: DimName + DimNameMin<C, Output = U1>,
    C: DimName + DimNameMin<R, Output = U1>,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, C, R>,
    MatrixMN<T, C, R>: Copy + FiniteDimensional<N = <Self as FiniteDimensional>::N>,
    Self: Copy + FiniteDimensional,
{
    type Dual = MatrixMN<T, C, R>;

    fn transpose(self) -> Self::Dual {
        nalgebra::Matrix::transpose(&self)
    }
}

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

impl<T, U, R, C> Fold<U> for MatrixMN<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn fold<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in self.iter() {
            seed = f(seed, *a);
        }
        seed
    }
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
    DefaultAllocator: Allocator<T, R, C> + Allocator<U, R, C>,
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

    fn mul_mn(self, other: Matrix2<T>) -> <Self as MulMN<Matrix2<T>>>::Output {
        self * other
    }
}

impl<T> MulMN<Matrix3<T>> for Matrix3<T>
where
    T: AddAssign + MulAssign + Real + Scalar,
{
    type Output = Matrix3<T>;

    fn mul_mn(self, other: Matrix3<T>) -> <Self as MulMN<Matrix3<T>>>::Output {
        self * other
    }
}

// TODO: It is possible to implement `Pop` for both column and row vectors.
//       However, this is not possible for `Push`, because it may be ambiguous
//       if a push should proceed down a row or column.  Moreover, the type
//       bounds to constrain such an implementation would be very complex.
//
//  impl<T, R, C> Pop for MatrixMN<T, R, C>
//  where
//      T: AddAssign + MulAssign + Real + Scalar,
//      R: DimName + DimNameMin<C, Output = U1> + DimNameSub<U1>,
//      C: DimName + DimNameMin<R, Output = U1> + DimNameSub<U1>,
//      DimNameDiff<R, U1>: DimNameMax<U1>,
//      DimNameDiff<C, U1>: DimNameMax<U1>,
//      DefaultAllocator: Allocator<T, R, C>,
//      DefaultAllocator: Allocator<
//          T,
//          DimNameMaximum<DimNameDiff<R, U1>, U1>,
//          DimNameMaximum<DimNameDiff<C, U1>, U1>,
//      >,
//  {
//      type Output =
//          MatrixMN<T, DimNameMaximum<DimNameDiff<R, U1>, U1>, DimNameMaximum<DimNameDiff<C, U1>, U1>>;
//
//      fn pop(self) -> (Self::Output, T) {
//          let n = self.len();
//          let x = *self.get(n - 1).unwrap();
//          (
//              MatrixMN::<
//                  T,
//                  DimNameMaximum<DimNameDiff<R, U1>, U1>,
//                  DimNameMaximum<DimNameDiff<C, U1>, U1>,
//              >::from_iterator(self.into_iter().take(n - 1).cloned()),
//              x,
//          )
//      }
//  }

impl<T, D> Pop for VectorN<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, DimNameDiff<D, U1>>,
{
    type Output = VectorN<T, DimNameDiff<D, U1>>;

    fn pop(self) -> (Self::Output, T) {
        let n = self.len();
        let x = *self.get(n - 1).unwrap();
        (
            VectorN::<_, DimNameDiff<D, _>>::from_iterator(self.into_iter().take(n - 1).cloned()),
            x,
        )
    }
}

impl<T, D> Push for VectorN<T, D>
where
    T: AddAssign + MulAssign + Real + Scalar,
    D: DimName + DimNameAdd<U1>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, DimNameSum<D, U1>>,
{
    type Output = VectorN<T, DimNameSum<D, U1>>;

    fn push(self, x: T) -> Self::Output {
        VectorN::<_, DimNameSum<D, _>>::from_iterator(self.into_iter().cloned().chain(Some(x)))
    }
}

impl<T, R, C> Series for MatrixMN<T, R, C>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    type Item = T;
}

impl<T> SquareMatrix for Matrix2<T>
where
    T: AddAssign + MulAssign + Real + Scalar,
{
    fn multiplicative_identity() -> Self {
        nalgebra::Matrix2::<T>::identity()
    }
}

impl<T> SquareMatrix for Matrix3<T>
where
    T: AddAssign + MulAssign + Real + Scalar,
{
    fn multiplicative_identity() -> Self {
        nalgebra::Matrix3::<T>::identity()
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
}

impl<T, U, R, C> ZipMap<U> for MatrixMN<T, R, C>
where
    T: Scalar,
    U: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C> + Allocator<U, R, C>,
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

impl<T, D> Series for Point<T, D>
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

    fn into_coordinates(self) -> Self::CoordinateSpace {
        self.coords
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

impl<T, U, D> Fold<U> for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn fold<F>(self, seed: U, f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        self.coords.fold(seed, f)
    }
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
    DefaultAllocator: Allocator<T, D> + Allocator<U, D>,
{
    type Output = Point<U, D>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Point::from(self.coords.map(f))
    }
}

impl<T, D> Pop for Point<T, D>
where
    T: Scalar,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, DimNameDiff<D, U1>>,
    VectorN<T, D>: Series<Item = T> + Pop<Output = VectorN<T, DimNameDiff<D, U1>>>,
{
    type Output = Point<T, DimNameDiff<D, U1>>;

    fn pop(self) -> (Self::Output, T) {
        let (vector, x) = self.coords.pop();
        (vector.into(), x)
    }
}

impl<T, D> Push for Point<T, D>
where
    T: Scalar,
    D: DimName + DimNameAdd<U1>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, DimNameSum<D, U1>>,
    VectorN<T, D>: Series<Item = T> + Push<Output = VectorN<T, DimNameSum<D, U1>>>,
{
    type Output = Point<T, DimNameSum<D, U1>>;

    fn push(self, x: T) -> Self::Output {
        self.coords.push(x).into()
    }
}

impl<T, U, D> ZipMap<U> for Point<T, D>
where
    T: Scalar,
    U: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<U, D>,
{
    type Output = Point<U, D>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Point::from(self.coords.zip_map(other.coords, f))
    }
}
