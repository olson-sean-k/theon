//! Homogeneous structured data.
//!
//! This module provides traits that abstract over structured homogeneous data.
//! In particular, these traits are implemented by array-like types with some
//! number of structured homogeneous elements. Here, _structure_ refers to
//! layout and ordering. A type that implements these traits is known as an
//! _adjunct_ and must implement the most basic `Adjunct` trait.
//!
//! Adjunct traits mirror iterator operations but act on both bounded and
//! unbounded data. Eudoxus uses these traits to manipulate data structures that
//! describe geometric constructs, typically as a matrix of scalar values.
//! Adjuncts can be both simple `struct`s with well-defined fields and more
//! dynamic types like `Vec`s.
//!
//! Implementations for adjunct traits are provided for integrated foreign types
//! when enabling geometry features. For example, implementations of `Adjunct`
//! and other traits are provided for `nalgebra` types when the
//! `geometry-nalgebra` feature is enabled. See the `integration` module.

use num_traits::{One, Zero};
use std::ops::{Add, Mul};
use typenum::{Max, Min, U1, U2};

use crate::space::{FiniteDimensional, ProjectiveDimensions};
use crate::{Increment, Natural};

pub trait Layout {
    type Bounds: Adjunct<Item = usize> + AsRef<[usize]> + Copy + Map<usize, Output = Self::Bounds>;

    const N: usize;
    const BOUNDS: Self::Bounds;

    fn is_zero() -> bool {
        Self::BOUNDS.as_ref().iter().any(|&bound| bound == 0)
    }
}

pub trait ShapeEq<T>: Layout
where
    T: Layout,
{
}

impl<N, R, C> ShapeEq<(N,)> for (R, C)
where
    N: Natural,
    R: Max<C, Output = N> + Min<C, Output = U1> + Natural,
    C: Natural,
{
}

impl<N, R, C> ShapeEq<(R, C)> for (N,)
where
    N: Natural,
    R: Max<C, Output = N> + Min<C, Output = U1> + Natural,
    C: Natural,
{
}

impl<R1, C1, R2, C2> ShapeEq<(R1, C1)> for (R2, C2)
where
    R1: Max<C1> + Min<C1> + Natural,
    C1: Natural,
    R2: Max<C2, Output = <R1 as Max<C1>>::Output>
        + Min<C2, Output = <R1 as Min<C1>>::Output>
        + Natural,
    C2: Natural,
{
}

pub trait ShapeSpan: Layout {
    type Output: Natural;
}

impl<N> ShapeSpan for (N,)
where
    N: Natural,
{
    type Output = U1;
}

impl<R, C> ShapeSpan for (R, C)
where
    R: Min<C> + Natural,
    C: Natural,
    U2: Min<<R as Min<C>>::Output>,
    <U2 as Min<<R as Min<C>>::Output>>::Output: Natural,
{
    type Output = <U2 as Min<<R as Min<C>>::Output>>::Output;
}

impl<N> Layout for (N,)
where
    N: Natural,
{
    type Bounds = [usize; 1];

    const N: usize = 1;
    const BOUNDS: Self::Bounds = [N::USIZE];
}

impl<R, C> Layout for (R, C)
where
    R: Natural,
    C: Natural,
{
    type Bounds = [usize; 2];

    const N: usize = 2;
    const BOUNDS: Self::Bounds = [R::USIZE, C::USIZE];
}

pub trait Shaped: Adjunct {
    type Layout: Layout;

    const ORDERING: <Self::Layout as Layout>::Bounds;

    fn get(&self, index: <Self::Layout as Layout>::Bounds) -> Option<&Self::Item>;
}

// TODO: This work was done a long time ago, so this comment _could_ be wrong, but it really
//       appears that this trait is meant to replace the `Linear` trait below. The "2" is probably
//       an artifact of experimentation with shape and layout ahead of replacing `Linear`.
pub trait Linear2: Shaped
where
    Self::Layout: ShapeSpan<Output = U1>,
{
    fn get(&self, index: usize) -> Option<&Self::Item> {
        <Self as Shaped>::get(
            self,
            <Self::Layout as Layout>::BOUNDS.clone().map(|bound| {
                if bound == 1 {
                    0
                }
                else {
                    index
                }
            }),
        )
    }
}

pub trait Adjunct: Sized {
    type Item;
}

pub trait Linear: Adjunct {
    fn get(&self, index: usize) -> Option<&Self::Item>;
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

// NOTE: This trait uses an input parameter `S` instead of an associated type
//       `Output`, because some types can be reasonably truncated into various
//       types. For example, the `glam` crate provides several vector
//       representations, and its `Vec3` and `Vec3A` types both extend into and
//       truncate from the `Vec4` type.
pub trait TruncateInto<S>: FiniteDimensional<N = ProjectiveDimensions<S>>
where
    S: Adjunct<Item = Self::Item> + FiniteDimensional,
    S::N: Increment,
{
    fn truncate(self) -> (S, Self::Item);
}

// NOTE: This trait uses an input parameter `S` instead of an associated type
//       `Output`, because some types can be reasonably extended into various
//       types. For example, the `glam` crate provides several vector
//       representations, and its `Vec3` and `Vec3A` types both extend into and
//       truncate from the `Vec4` type.
pub trait ExtendInto<S>: FiniteDimensional
where
    S: Adjunct<Item = Self::Item> + FiniteDimensional<N = ProjectiveDimensions<Self>>,
    Self::N: Increment,
{
    fn extend(self, item: Self::Item) -> S;
}

pub trait TryFromIterator<I>: Linear
where
    I: Iterator<Item = Self::Item>,
{
    type Error;
    type Remainder: Iterator<Item = I::Item>;

    fn try_from_iter(items: I) -> Result<(Self, Option<Self::Remainder>), Self::Error>;
}

pub trait IntoIterator: Linear {
    type Output: Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::Output;
}

pub trait IteratorExt: Iterator + Sized {
    fn try_collect<T>(self) -> Result<(T, Option<T::Remainder>), T::Error>
    where
        T: TryFromIterator<Self, Item = Self::Item>,
    {
        T::try_from_iter(self)
    }

    // TODO: Move this into Theon.
    fn try_collect_all<T>(self) -> Result<T, ()>
    where
        T: TryFromIterator<Self, Item = Self::Item>,
    {
        let (collection, remainder) = self.try_collect::<T>().map_err(|_| ())?;
        if remainder.is_some() {
            Err(())
        }
        else {
            Ok(collection)
        }
    }
}

impl<I> IteratorExt for I where I: Iterator + Sized {}
