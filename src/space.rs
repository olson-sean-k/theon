use decorum::Real;
use num::{NumCast, One, Zero};
use std::ops::{Add, Mul, Neg, Sub};
use typenum::consts::{U0, U1, U2, U3};
use typenum::type_operators::Cmp;
use typenum::{Greater, NonZero, Unsigned};

use crate::ops::{Dot, Project};
use crate::Category;

pub type Dimensions<S> = <<S as EuclideanSpace>::CoordinateSpace as FiniteDimensional>::N;
pub type Scalar<S> = <Vector<S> as VectorSpace>::Scalar;
pub type Vector<S> = <S as EuclideanSpace>::CoordinateSpace;

pub trait FiniteDimensional {
    type N: NonZero + Unsigned;

    fn dimensions() -> usize {
        Self::N::USIZE
    }
}

pub trait Basis: FiniteDimensional + Sized {
    type Bases: IntoIterator<Item = Self>;

    fn canonical_basis() -> Self::Bases;

    fn canonical_basis_component(index: usize) -> Option<Self> {
        Self::canonical_basis().into_iter().nth(index)
    }

    fn x() -> Self
    where
        Self::N: Cmp<U0, Output = Greater>,
    {
        Self::canonical_basis_component(0).unwrap()
    }

    fn y() -> Self
    where
        Self::N: Cmp<U1, Output = Greater>,
    {
        Self::canonical_basis_component(1).unwrap()
    }

    fn z() -> Self
    where
        Self::N: Cmp<U2, Output = Greater>,
    {
        Self::canonical_basis_component(2).unwrap()
    }
}

pub trait VectorSpace:
    Add<Output = Self>
    + Category<Object = <Self as VectorSpace>::Scalar>
    + Copy
    + FiniteDimensional
    + Mul<<Self as VectorSpace>::Scalar, Output = Self>
    + Neg<Output = Self>
    + Zero
{
    type Scalar: Copy + Real;

    fn scalar_component(&self, index: usize) -> Option<&Self::Scalar>;

    fn from_x(x: Self::Scalar) -> Self
    where
        Self: Basis + FiniteDimensional<N = U1>,
    {
        Self::x() * x
    }

    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self
    where
        Self: Basis + FiniteDimensional<N = U2>,
    {
        (Self::x() * x) + (Self::y() * y)
    }

    fn from_xyz(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self
    where
        Self: Basis + FiniteDimensional<N = U3>,
    {
        (Self::x() * x) + (Self::y() * y) + (Self::z() * z)
    }

    fn mean<I>(vectors: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
    {
        let mut vectors = vectors.into_iter();
        if let Some(mut sum) = vectors.next() {
            let mut n = 1usize;
            for vector in vectors {
                n += 1;
                sum = sum + vector;
            }
            NumCast::from(n).map(move |n| sum * (Self::Scalar::one() / n))
        }
        else {
            None
        }
    }
}

pub trait InnerSpace: Dot<Output = <Self as VectorSpace>::Scalar> + VectorSpace {
    fn normalize(self) -> Option<Self> {
        let magnitude = self.magnitude();
        if magnitude != Zero::zero() {
            Some(self * (Self::Scalar::one() / magnitude))
        }
        else {
            None
        }
    }

    fn square_magnitude(self) -> Self::Scalar {
        Dot::dot(self, self)
    }

    fn magnitude(self) -> Self::Scalar {
        Real::sqrt(self.square_magnitude())
    }
}

impl<T> Project<T> for T
where
    T: InnerSpace,
{
    type Output = T;

    fn project(self, other: T) -> Self::Output {
        let n = other.dot(self);
        let d = self.dot(self);
        self * (n / d)
    }
}

pub trait AffineSpace:
    Add<<Self as AffineSpace>::Translation, Output = Self>
    + Category<Object = <<Self as AffineSpace>::Translation as VectorSpace>::Scalar>
    + Copy
    + Sub<Output = <Self as AffineSpace>::Translation>
{
    type Translation: VectorSpace;

    fn translate(self, translation: Self::Translation) -> Self {
        self + translation
    }

    fn difference(self, other: Self) -> Self::Translation {
        self - other
    }
}

pub trait EuclideanSpace:
    AffineSpace<Translation = <Self as EuclideanSpace>::CoordinateSpace>
{
    type CoordinateSpace: InnerSpace;

    fn origin() -> Self;

    fn coordinates(&self) -> Self::CoordinateSpace {
        self.clone() - Self::origin()
    }

    fn centroid<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
    {
        VectorSpace::mean(points.into_iter().map(|point| point.coordinates()))
            .map(|mean| Self::origin() + mean)
    }

    fn from_x(x: Scalar<Self>) -> Self
    where
        Self::CoordinateSpace: Basis + FiniteDimensional<N = U1>,
    {
        Self::origin() + Vector::<Self>::from_x(x)
    }

    fn from_xy(x: Scalar<Self>, y: Scalar<Self>) -> Self
    where
        Self::CoordinateSpace: Basis + FiniteDimensional<N = U2>,
    {
        Self::origin() + Vector::<Self>::from_xy(x, y)
    }

    fn from_xyz(x: Scalar<Self>, y: Scalar<Self>, z: Scalar<Self>) -> Self
    where
        Self::CoordinateSpace: Basis + FiniteDimensional<N = U3>,
    {
        Self::origin() + Vector::<Self>::from_xyz(x, y, z)
    }
}
