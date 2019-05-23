use decorum::R64;

use crate::Composite;

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

pub trait ZipMap<T = <Self as Composite>::Item>: Composite {
    type Output: Composite<Item = T>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> T;
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

pub trait Reduce<T = <Self as Composite>::Item>: Composite {
    fn reduce<F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T;
}

impl<T, U> Reduce<U> for (T, T) {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        seed = f(seed, self.0);
        seed = f(seed, self.1);
        seed
    }
}

impl<T, U> Reduce<U> for (T, T, T) {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        seed = f(seed, self.0);
        seed = f(seed, self.1);
        seed = f(seed, self.2);
        seed
    }
}
