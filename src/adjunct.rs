//! Static groups of homogeneous data.
//!
//! This module provides APIs for types that expose a rectangular and ordered set of homogeneous
//! data. These types are typically array-like, but these APIs are not limited to arrays. Any type
//! that provides ordered elements of the same type may be capable of supporting these APIs.
//!
//! A type that implements these traits and operations is known as an _adjunct_. See the `Adjunct`
//! trait for more.
//!
//! Implementations for adjunct traits are provided for integrated foreign types when enabling
//! geometry features. For example, implementations of `Adjunct` and other traits are provided for
//! `nalgebra` types when the `nalgebra` feature is enabled.

use decorum::cmp::{self, EmptyOrd};
use num_traits::bounds::Bounded;
use num_traits::identities::{One, Zero};
use std::ops::{Add, Mul};

pub trait Adjunct: Sized {
    type Item;
}

pub trait IntoItems: Adjunct {
    type Output: IntoIterator<Item = Self::Item>;

    fn into_items(self) -> Self::Output;
}

pub trait FromItems: Adjunct {
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

// TODO: Consider renaming the `Truncate` and `Extend` traits to `TruncateMap`, `TruncateInto`,
//       etc., because these traits must support multiple output types.
pub trait Truncate<S>: Adjunct
where
    S: Adjunct<Item = Self::Item>,
{
    fn truncate(self) -> (S, Self::Item);
}

pub trait Extend<S>: Adjunct
where
    S: Adjunct<Item = Self::Item>,
{
    fn extend(self, item: Self::Item) -> S;
}

pub trait ZipMap<T = <Self as Adjunct>::Item>: Adjunct {
    type Output: Adjunct<Item = T>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> T;

    fn per_item_sum(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T>,
        T: Add<Output = T>,
    {
        self.zip_map(other, |a, b| a + b)
    }

    fn per_item_product(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T>,
        T: Mul<Output = T>,
    {
        self.zip_map(other, |a, b| a * b)
    }

    fn per_item_min_or_empty(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T>,
        T: EmptyOrd,
    {
        self.zip_map(other, cmp::min_or_empty)
    }

    fn per_item_max_or_empty(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = T>,
        T: EmptyOrd,
    {
        self.zip_map(other, cmp::max_or_empty)
    }
}

pub trait Fold: Adjunct {
    fn fold<T, F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T;

    fn sum(self) -> Self::Item
    where
        Self::Item: Add<Output = Self::Item> + Zero,
    {
        self.fold(Zero::zero(), |sum, n| sum + n)
    }

    fn product(self) -> Self::Item
    where
        Self::Item: Mul<Output = Self::Item> + One,
    {
        self.fold(One::one(), |product, n| product * n)
    }

    fn min_or_empty(self) -> Self::Item
    where
        Self::Item: Bounded + EmptyOrd,
    {
        self.fold(Bounded::max_value(), cmp::min_or_empty)
    }

    fn max_or_empty(self) -> Self::Item
    where
        Self::Item: Bounded + EmptyOrd,
    {
        self.fold(Bounded::min_value(), cmp::max_or_empty)
    }

    fn any<F>(self, mut f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.fold(false, |sum, item| {
            if sum {
                sum
            }
            else {
                f(item)
            }
        })
    }

    fn all<F>(self, mut f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.fold(true, |sum, item| {
            if sum {
                f(item)
            }
            else {
                sum
            }
        })
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

impl<T> Fold for (T, T) {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        seed = f(seed, self.0);
        seed = f(seed, self.1);
        seed
    }
}

impl<T> Fold for (T, T, T) {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
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
    type Output = [T; 2];

    fn into_items(self) -> Self::Output {
        [self.0, self.1]
    }
}

impl<T> IntoItems for (T, T, T) {
    type Output = [T; 3];

    fn into_items(self) -> Self::Output {
        [self.0, self.1, self.2]
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

impl<T> Truncate<(T, T)> for (T, T, T) {
    fn truncate(self) -> ((T, T), Self::Item) {
        let (a, b, c) = self;
        ((a, b), c)
    }
}

impl<T> Extend<(T, T, T)> for (T, T) {
    fn extend(self, item: Self::Item) -> (T, T, T) {
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
