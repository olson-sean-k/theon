//! Vector and affine spaces.

use decorum::Real;
use num::{NumCast, One, Zero};
use std::ops::{Add, Mul, Neg, Sub};
use typenum::consts::{U0, U1, U2, U3};
use typenum::type_operators::Cmp;
use typenum::{Greater, NonZero, Unsigned};

use crate::ops::{Dot, Fold, Pop, Project, Push, ZipMap};
use crate::Composite;

/// The scalar of a `EuclideanSpace`.
pub type Scalar<S> = <Vector<S> as VectorSpace>::Scalar;

/// The vector (translation, coordinate space, etc.) of a `EuclideanSpace`.
pub type Vector<S> = <S as EuclideanSpace>::CoordinateSpace;

pub trait FiniteDimensional {
    type N: NonZero + Unsigned;

    fn dimensions() -> usize {
        Self::N::USIZE
    }
}

/// Describes the basis of a vector space.
pub trait Basis: FiniteDimensional + Sized {
    type Bases: IntoIterator<Item = Self>;

    /// Gets a type that can be converted into an iterator over the _canonical_
    /// or _standard_ basis vectors of the space.
    ///
    /// Such basis vectors must only have one component set to the
    /// multiplicative identity one and all other components set to zero.
    /// Moreover, the set of basis vectors must contain ordered and unique
    /// elements and be of size equal to the dimensionality of the space.
    ///
    /// For example, the set of basis vectors for the real coordinate space
    /// $\Reals^2$ is:
    ///
    /// $\\{\begin{bmatrix}1\\\0\end{bmatrix},\begin{bmatrix}0\\\1\end{bmatrix}\\}$
    fn canonical_basis() -> Self::Bases;

    fn canonical_basis_component(index: usize) -> Option<Self> {
        Self::canonical_basis().into_iter().nth(index)
    }

    /// Gets the basis vector $\hat{i}$ that describes the $x$ axis.
    fn x() -> Self
    where
        Self::N: Cmp<U0, Output = Greater>,
    {
        Self::canonical_basis_component(0).unwrap()
    }

    /// Gets the basis vector $\hat{j}$ that describes the $y$ axis.
    fn y() -> Self
    where
        Self::N: Cmp<U1, Output = Greater>,
    {
        Self::canonical_basis_component(1).unwrap()
    }

    /// Gets the basis vector $\hat{k}$ that describes the $z$ axis.
    fn z() -> Self
    where
        Self::N: Cmp<U2, Output = Greater>,
    {
        Self::canonical_basis_component(2).unwrap()
    }
}

pub trait VectorSpace:
    Add<Output = Self>
    + Composite<Item = <Self as VectorSpace>::Scalar>
    + Copy
    + Fold<<Self as VectorSpace>::Scalar>
    + Mul<<Self as VectorSpace>::Scalar, Output = Self>
    + Neg<Output = Self>
    + Zero
    + ZipMap<<Self as VectorSpace>::Scalar, Output = Self>
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

    fn from_homogeneous(vector: Self::ProjectiveSpace) -> Option<Self>
    where
        Self: Homogeneous,
        Self::ProjectiveSpace: Pop<Output = Self> + VectorSpace<Scalar = Self::Scalar>,
    {
        let (vector, factor) = vector.pop();
        if factor.is_zero() {
            Some(vector)
        }
        else {
            None
        }
    }

    fn into_homogeneous(self) -> Self::ProjectiveSpace
    where
        Self: Homogeneous + Push<Output = <Self as Homogeneous>::ProjectiveSpace>,
        Self::ProjectiveSpace: VectorSpace<Scalar = Self::Scalar>,
    {
        self.push(Zero::zero())
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

pub trait DualSpace: FiniteDimensional + VectorSpace {
    type Dual: DualSpace + FiniteDimensional<N = Self::N> + VectorSpace<Scalar = Self::Scalar>;

    fn transpose(self) -> Self::Dual;
}

pub trait Matrix: VectorSpace {
    type Row: DualSpace + FiniteDimensional + VectorSpace<Scalar = Self::Scalar>;
    type Column: DualSpace + FiniteDimensional + VectorSpace<Scalar = Self::Scalar>;
    type Transpose: Matrix<
        Scalar = Self::Scalar,
        Row = <Self::Column as DualSpace>::Dual,
        Column = <Self::Row as DualSpace>::Dual,
    >;

    fn row_count() -> usize {
        Self::Column::dimensions()
    }

    fn column_count() -> usize {
        Self::Row::dimensions()
    }

    fn scalar_component(&self, row: usize, column: usize) -> Option<&Self::Scalar> {
        <Self as VectorSpace>::scalar_component(self, row + (column * Self::row_count()))
    }

    fn row_component(&self, index: usize) -> Option<Self::Row>;

    fn column_component(&self, index: usize) -> Option<Self::Column>;

    fn transpose(self) -> Self::Transpose;
}

pub trait SquareMatrix:
    Matrix<Row = <<Self as Matrix>::Column as DualSpace>::Dual>
    + Mul<Output = Self>
    + Mul<<Self as Matrix>::Column, Output = <Self as Matrix>::Column>
where
    Self::Row: FiniteDimensional<N = <Self::Column as FiniteDimensional>::N>,
{
    fn multiplicative_identity() -> Self;
}

pub trait AffineSpace:
    Add<<Self as AffineSpace>::Translation, Output = Self>
    + Composite<Item = <<Self as AffineSpace>::Translation as VectorSpace>::Scalar>
    + Copy
    + Fold<<<Self as AffineSpace>::Translation as VectorSpace>::Scalar>
    + Sub<Output = <Self as AffineSpace>::Translation>
    + ZipMap<<<Self as AffineSpace>::Translation as VectorSpace>::Scalar, Output = Self>
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
    AffineSpace<Translation = <Self as EuclideanSpace>::CoordinateSpace> + FiniteDimensional
{
    type CoordinateSpace: Basis + InnerSpace + FiniteDimensional<N = <Self as FiniteDimensional>::N>;

    fn origin() -> Self;

    fn from_coordinates(coordinates: Self::CoordinateSpace) -> Self {
        Self::origin() + coordinates
    }

    fn into_coordinates(self) -> Self::CoordinateSpace {
        self - Self::origin()
    }

    fn from_x(x: Scalar<Self>) -> Self
    where
        Self: FiniteDimensional<N = U1>,
    {
        Self::from_coordinates(Vector::<Self>::from_x(x))
    }

    fn from_xy(x: Scalar<Self>, y: Scalar<Self>) -> Self
    where
        Self: FiniteDimensional<N = U2>,
    {
        Self::from_coordinates(Vector::<Self>::from_xy(x, y))
    }

    fn from_xyz(x: Scalar<Self>, y: Scalar<Self>, z: Scalar<Self>) -> Self
    where
        Self: FiniteDimensional<N = U3>,
    {
        Self::from_coordinates(Vector::<Self>::from_xyz(x, y, z))
    }

    fn from_homogeneous(vector: Self::ProjectiveSpace) -> Option<Self>
    where
        Self: Homogeneous,
        Self::ProjectiveSpace:
            Pop<Output = Self::CoordinateSpace> + VectorSpace<Scalar = Scalar<Self>>,
    {
        let (vector, factor) = vector.pop();
        if factor.is_zero() {
            None
        }
        else {
            Some(Self::from_coordinates(vector * factor.recip()))
        }
    }

    fn into_homogeneous(self) -> Self::ProjectiveSpace
    where
        Self: Homogeneous,
        Self::CoordinateSpace: Push<Output = Self::ProjectiveSpace>,
        Self::ProjectiveSpace: VectorSpace<Scalar = Scalar<Self>>,
    {
        self.into_coordinates().push(One::one())
    }

    fn centroid<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
    {
        VectorSpace::mean(points.into_iter().map(|point| point.into_coordinates()))
            .map(|mean| Self::origin() + mean)
    }
}

// TODO: Constrain the dimensionality of the projective space. This introduces
//       noisy type bounds, but ensures that the projective space has exactly
//       one additional dimension (the line at infinity).
//
//   pub trait Homogeneous: FiniteDimensional
//   where
//       Self::N: Add<U1>,
//       <Self::N as Add<U1>>::Output: NonZero + Unsigned,
//   {
//       type ProjectiveSpace: FiniteDimensional<N = <Self::N as Add<U1>>::Output>;
//   }
pub trait Homogeneous: FiniteDimensional {
    type ProjectiveSpace: FiniteDimensional;
}
