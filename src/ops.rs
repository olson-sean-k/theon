use decorum::R64;
use itertools::iproduct;
use num::{Bounded, One, Zero};
use std::ops::{Add, Mul};

use crate::space::{DualSpace, FiniteDimensional, Matrix, VectorSpace};
use crate::{Composite, FromItems, Lattice};

pub trait Project<T = Self> {
    type Output;

    fn project(self, other: T) -> Self::Output;
}

pub trait Interpolate<T = Self>: Sized {
    type Output;

    fn lerp(self, other: T, f: R64) -> Self::Output;

    fn midpoint(self, other: T) -> Self::Output {
        self.lerp(other, 0.5.into())
    }
}

pub trait Dot<T = Self> {
    type Output;

    fn dot(self, other: T) -> Self::Output;
}

pub trait Cross<T = Self> {
    type Output;

    fn cross(self, other: T) -> Self::Output;
}

pub trait MulMN<T = Self>: Matrix
where
    T: Matrix<Scalar = Self::Scalar>,
    // The `VectorSpace<Scalar = Self::Scalar>` and `FiniteDimensional` bounds
    // are redundant, but are needed by the compiler.
    <T as Matrix>::Column: VectorSpace<Scalar = Self::Scalar>,
    Self::Row: DualSpace<Dual = <T as Matrix>::Column>
        + FiniteDimensional<N = <T::Column as FiniteDimensional>::N>,
{
    // TODO: This implementation requires `FromItems`, which could be
    //       cumbersome to implement.
    type Output: FromItems + Matrix<Scalar = Self::Scalar>;

    fn mul_mn(self, other: T) -> <Self as MulMN<T>>::Output {
        FromItems::from_items(
            iproduct!(
                (0..Self::row_count()).map(|index| self.row_component(index).unwrap().transpose()),
                (0..T::column_count()).map(|index| other.column_component(index).unwrap())
            )
            .map(|(row, column)| row.per_item_product(column).sum()),
        )
        .unwrap()
    }
}

pub trait Map<T = <Self as Composite>::Item>: Composite {
    type Output: Composite<Item = T>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> T;
}

impl<T, U> Map<U> for (T, T) {
    type Output = (U, U);

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        (f(self.0), f(self.1))
    }
}

impl<T, U> Map<U> for (T, T, T) {
    type Output = (U, U, U);

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        (f(self.0), f(self.1), f(self.2))
    }
}

pub trait Zip {
    type Output: Composite + FromItems;

    fn zip(self) -> Self::Output;
}

pub trait ZipMap<T = <Self as Composite>::Item>: Composite {
    type Output: Composite<Item = T>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> T;

    fn per_item_sum(self, other: Self) -> Self::Output
    where
        Self: Composite<Item = T> + Sized,
        T: Add<Output = T>,
    {
        self.zip_map(other, |a, b| a + b)
    }

    fn per_item_product(self, other: Self) -> Self::Output
    where
        Self: Composite<Item = T> + Sized,
        T: Mul<Output = T>,
    {
        self.zip_map(other, |a, b| a * b)
    }

    fn per_item_partial_min(self, other: Self) -> Self::Output
    where
        Self: Composite<Item = T> + Sized,
        T: Copy + Lattice,
    {
        self.zip_map(other, |a, b| crate::partial_min(a, b))
    }

    fn per_item_partial_max(self, other: Self) -> Self::Output
    where
        Self: Composite<Item = T> + Sized,
        T: Copy + Lattice,
    {
        self.zip_map(other, |a, b| crate::partial_max(a, b))
    }
}

impl<T, U> ZipMap<U> for (T, T) {
    type Output = (U, U);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        (f(self.0, other.0), f(self.1, other.1))
    }
}

impl<T, U> ZipMap<U> for (T, T, T) {
    type Output = (U, U, U);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        (f(self.0, other.0), f(self.1, other.1), f(self.2, other.2))
    }
}

pub trait Fold<T = <Self as Composite>::Item>: Composite {
    fn fold<F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T;

    fn sum(self) -> T
    where
        Self: Composite<Item = T> + Sized,
        T: Add<Output = T> + Zero,
    {
        self.fold(Zero::zero(), |sum, n| sum + n)
    }

    fn product(self) -> T
    where
        Self: Composite<Item = T> + Sized,
        T: Mul<Output = T> + One,
    {
        self.fold(One::one(), |product, n| product * n)
    }

    fn partial_min(self) -> T
    where
        Self: Composite<Item = T> + Sized,
        T: Bounded + Copy + Lattice,
    {
        self.fold(Bounded::max_value(), |min, n| crate::partial_min(min, n))
    }

    fn partial_max(self) -> T
    where
        Self: Composite<Item = T> + Sized,
        T: Bounded + Copy + Lattice,
    {
        self.fold(Bounded::min_value(), |max, n| crate::partial_max(max, n))
    }
}

impl<T, U> Fold<U> for (T, T) {
    fn fold<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        seed = f(seed, self.0);
        seed = f(seed, self.1);
        seed
    }
}

impl<T, U> Fold<U> for (T, T, T) {
    fn fold<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        seed = f(seed, self.0);
        seed = f(seed, self.1);
        seed = f(seed, self.2);
        seed
    }
}
