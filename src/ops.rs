use decorum::R64;

use crate::Category;

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

pub trait Map<T>: Category {
    type Output: Category<Object = T>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> T;
}

impl<T, U> Map<U> for (T, T) {
    type Output = (U, U);

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        (f(self.0), f(self.1))
    }
}

impl<T, U> Map<U> for (T, T, T) {
    type Output = (U, U, U);

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        (f(self.0), f(self.1), f(self.2))
    }
}

pub trait ZipMap<T>: Category {
    type Output: Category<Object = T>;

    fn zip_map<F>(self, other: Self, f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> T;
}

impl<T, U> ZipMap<U> for (T, T) {
    type Output = (U, U);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        (f(self.0, other.0), f(self.1, other.1))
    }
}

impl<T, U> ZipMap<U> for (T, T, T) {
    type Output = (U, U, U);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        (f(self.0, other.0), f(self.1, other.1), f(self.2, other.2))
    }
}

pub trait Reduce<T, U = T> {
    fn reduce<F>(self, seed: U, f: F) -> U
    where
        F: FnMut(U, T) -> U;
}

impl<T, U> Reduce<T, U> for (T, T) {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        seed = f(seed, self.0);
        seed = f(seed, self.1);
        seed
    }
}

impl<T, U> Reduce<T, U> for (T, T, T) {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        seed = f(seed, self.0);
        seed = f(seed, self.1);
        seed = f(seed, self.2);
        seed
    }
}
