//! Static groups of homogeneous data.
//!
//! This module provides APIs for types that expose a rectangular and ordered
//! set of homogeneous data. These types are typically array-like, but these
//! APIs are not limited to arrays. Any type that provides ordered elements of
//! the same type may be capable of supporting these APIs.
//!
//! A type that implements these traits and operations is known as an _adjunct_.
//! See the `Adjunct` trait for more.
//!
//! Implementations for adjunct traits are provided for integrated foreign types
//! when enabling geometry features. For example, implementations of `Adjunct`
//! and other traits are provided for `nalgebra` types when the
//! `geometry-nalgebra` feature is enabled.

use arrayvec::ArrayVec;
use num::{Bounded, One, Zero};
use std::ops::{Add, Mul};

use crate::Lattice;

pub trait Adjunct {
    type Item;
}

pub trait IntoItems: Adjunct {
    type Output: IntoIterator<Item = Self::Item>;

    fn into_items(self) -> Self::Output;
}

pub trait FromItems: Adjunct + Sized {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>;
}

pub trait Converged: Adjunct {
    fn converged(value: Self::Item) -> Self;
}

pub trait Map<T = <Self as Adjunct>::Item>: Adjunct {
    type Output: Adjunct<Item = T>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> T;
}

pub trait Pop: Adjunct {
    type Output: Adjunct<Item = Self::Item>;

    fn pop(self) -> (Self::Output, Self::Item);
}

pub trait Push: Adjunct {
    type Output: Adjunct<Item = Self::Item>;

    fn push(self, item: Self::Item) -> Self::Output;
}

pub trait ZipMap<T = <Self as Adjunct>::Item>: Adjunct {
    type Output: Adjunct<Item = T>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> T;

    fn per_item_sum(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T> + Sized,
        T: Add<Output = T>,
    {
        self.zip_map(other, |a, b| a + b)
    }

    fn per_item_product(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T> + Sized,
        T: Mul<Output = T>,
    {
        self.zip_map(other, |a, b| a * b)
    }

    fn per_item_partial_min(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T> + Sized,
        T: Copy + Lattice,
    {
        self.zip_map(other, crate::partial_min)
    }

    fn per_item_partial_max(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T> + Sized,
        T: Copy + Lattice,
    {
        self.zip_map(other, crate::partial_max)
    }
}

pub trait Fold<T = <Self as Adjunct>::Item>: Adjunct {
    fn fold<F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T;

    fn sum(self) -> T
    where
        Self: Adjunct<Item = T> + Sized,
        T: Add<Output = T> + Zero,
    {
        self.fold(Zero::zero(), |sum, n| sum + n)
    }

    fn product(self) -> T
    where
        Self: Adjunct<Item = T> + Sized,
        T: Mul<Output = T> + One,
    {
        self.fold(One::one(), |product, n| product * n)
    }

    fn partial_min(self) -> T
    where
        Self: Adjunct<Item = T> + Sized,
        T: Bounded + Copy + Lattice,
    {
        self.fold(Bounded::max_value(), crate::partial_min)
    }

    fn partial_max(self) -> T
    where
        Self: Adjunct<Item = T> + Sized,
        T: Bounded + Copy + Lattice,
    {
        self.fold(Bounded::min_value(), crate::partial_max)
    }
}

// TODO: Use macros to implement these traits for arrays and tuples.

impl<T> Converged for (T, T)
where
    T: Clone,
{
    fn converged(value: Self::Item) -> Self {
        (value.clone(), value)
    }
}

impl<T> Converged for (T, T, T)
where
    T: Clone,
{
    fn converged(value: Self::Item) -> Self {
        (value.clone(), value.clone(), value)
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

impl<T> FromItems for (T, T) {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut items = items.into_iter().take(2);
        match (items.next(), items.next()) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }
}

impl<T> FromItems for (T, T, T) {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut items = items.into_iter().take(3);
        match (items.next(), items.next(), items.next()) {
            (Some(a), Some(b), Some(c)) => Some((a, b, c)),
            _ => None,
        }
    }
}

impl<T> IntoItems for (T, T) {
    type Output = ArrayVec<[T; 2]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.0, self.1])
    }
}

impl<T> IntoItems for (T, T, T) {
    type Output = ArrayVec<[T; 3]>;

    fn into_items(self) -> Self::Output {
        ArrayVec::from([self.0, self.1, self.2])
    }
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

impl<T> Pop for (T, T, T) {
    type Output = (T, T);

    fn pop(self) -> (Self::Output, Self::Item) {
        let (a, b, c) = self;
        ((a, b), c)
    }
}

impl<T> Push for (T, T) {
    type Output = (T, T, T);

    fn push(self, item: Self::Item) -> Self::Output {
        let (a, b) = self;
        (a, b, item)
    }
}

impl<T> Adjunct for (T, T) {
    type Item = T;
}

impl<T> Adjunct for (T, T, T) {
    type Item = T;
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
