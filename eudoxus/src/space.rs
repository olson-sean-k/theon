//! Vector and affine spaces.

use num_traits::{One, Zero};
use std::ops::{Add, Div, Mul, Neg, Sub};
use typenum::{Cmp, Greater, Unsigned, U0, U1, U2, U3, U4};

use crate::adjunct::{Adjunct, Converged, ExtendInto, Fold, Linear, Map, TruncateInto, ZipMap};
use crate::{AsPosition, Increment, Natural};

/// The scalar of a `EuclideanSpace`.
pub type Scalar<S> = <Vector<S> as VectorSpace>::Scalar;

/// The vector (translation, coordinate space, etc.) of a `EuclideanSpace`.
pub type Vector<S> = <S as EuclideanSpace>::CoordinateSpace;

/// The projective space of a `EuclideanSpace`.
pub type Projective<S> = <Vector<S> as Homogeneous>::ProjectiveSpace;

/// The finite dimensions of the projective space of a `EuclideanSpace`.
///
/// See `FiniteDimensional`.
pub type ProjectiveDimensions<S> = <<S as FiniteDimensional>::N as Increment>::Output;

trait OptionExt<T> {
    fn expect_dimensional_component(self) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    fn expect_dimensional_component(self) -> T {
        self.expect("dimensional component not found")
    }
}

pub trait FiniteDimensional: Linear {
    type N: Natural;

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
    /// multiplicative identity and all other components set to the additive
    /// identity (one and zero in $\Reals$, respectively). Moreover, the set of
    /// basis vectors must contain ordered and unique elements and be of size
    /// equal to the dimensionality of the space.
    ///
    /// For example, the set of canonical basis vectors for the real coordinate
    /// space $\Reals^3$ is:
    ///
    /// $$
    /// \\{\hat{i},\hat{j},\hat{k}\\}=
    /// \\left\\{
    /// \begin{bmatrix}1\\\0\\\0\end{bmatrix},
    /// \begin{bmatrix}0\\\1\\\0\end{bmatrix},
    /// \begin{bmatrix}0\\\0\\\1\end{bmatrix}
    /// \\right\\}
    /// $$
    fn canonical_basis() -> Self::Bases;

    fn canonical_basis_component(index: usize) -> Option<Self> {
        Self::canonical_basis().into_iter().nth(index)
    }

    /// Gets the canonical basis vector $\hat{i}$ that describes the $x$ axis.
    fn i() -> Self
    where
        Self::N: Cmp<U0, Output = Greater>,
    {
        Self::canonical_basis_component(0).expect_dimensional_component()
    }

    /// Gets the canonical basis vector $\hat{j}$ that describes the $y$ axis.
    fn j() -> Self
    where
        Self::N: Cmp<U1, Output = Greater>,
    {
        Self::canonical_basis_component(1).expect_dimensional_component()
    }

    /// Gets the canonical basis vector $\hat{k}$ that describes the $z$ axis.
    fn k() -> Self
    where
        Self::N: Cmp<U2, Output = Greater>,
    {
        Self::canonical_basis_component(2).expect_dimensional_component()
    }
}

pub trait VectorSpace:
    Add<Output = Self>
    + Adjunct<Item = <Self as VectorSpace>::Scalar>
    + Converged
    + Copy
    + Fold
    + PartialEq
    + Map<<Self as VectorSpace>::Scalar, Output = Self>
    + Mul<<Self as VectorSpace>::Scalar, Output = Self>
    + Neg<Output = Self>
    + ZipMap<<Self as VectorSpace>::Scalar, Output = Self>
{
    type Scalar: Add<Output = Self::Scalar>
        + Copy
        + Div<Output = Self::Scalar>
        + Mul<Output = Self::Scalar>
        + Neg<Output = Self::Scalar>
        + One
        + PartialEq
        + PartialOrd
        + Zero;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar>
    where
        Self: Linear,
    {
        self.get(index).copied()
    }

    fn from_x(x: Self::Scalar) -> Self
    where
        Self: Basis + FiniteDimensional<N = U1>,
    {
        Self::i() * x
    }

    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self
    where
        Self: Basis + FiniteDimensional<N = U2>,
    {
        (Self::i() * x) + (Self::j() * y)
    }

    fn from_xyz(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self
    where
        Self: Basis + FiniteDimensional<N = U3>,
    {
        (Self::i() * x) + (Self::j() * y) + (Self::k() * z)
    }

    fn from_xyzw(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar, w: Self::Scalar) -> Self
    where
        Self: Basis + FiniteDimensional<N = U4>,
    {
        (Self::i() * x)
            + (Self::j() * y)
            + (Self::k() * z)
            + (Self::canonical_basis_component(3).expect_dimensional_component() * w)
    }

    fn into_x(self) -> Self::Scalar
    where
        Self: FiniteDimensional<N = U1>,
    {
        self.x()
    }

    fn into_xy(self) -> (Self::Scalar, Self::Scalar)
    where
        Self: FiniteDimensional<N = U2>,
    {
        (self.x(), self.y())
    }

    fn into_xyz(self) -> (Self::Scalar, Self::Scalar, Self::Scalar)
    where
        Self: FiniteDimensional<N = U3>,
    {
        (self.x(), self.y(), self.z())
    }

    fn into_xyzw(self) -> (Self::Scalar, Self::Scalar, Self::Scalar, Self::Scalar)
    where
        Self: FiniteDimensional<N = U4>,
    {
        (self.x(), self.y(), self.z(), self.w())
    }

    fn x(&self) -> Self::Scalar
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U0, Output = Greater>,
    {
        self.scalar_component(0).expect_dimensional_component()
    }

    fn y(&self) -> Self::Scalar
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U1, Output = Greater>,
    {
        self.scalar_component(1).expect_dimensional_component()
    }

    fn z(&self) -> Self::Scalar
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U2, Output = Greater>,
    {
        self.scalar_component(2).expect_dimensional_component()
    }

    fn w(&self) -> Self::Scalar
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U3, Output = Greater>,
    {
        self.scalar_component(3).expect_dimensional_component()
    }

    fn zero() -> Self {
        Converged::converged(Zero::zero())
    }

    fn is_zero(&self) -> bool {
        self.all(|x| x.is_zero())
    }

    fn from_homogeneous(projective: Self::ProjectiveSpace) -> Option<Self>
    where
        Self: Homogeneous,
        Self::ProjectiveSpace: FiniteDimensional<N = ProjectiveDimensions<Self>>
            + TruncateInto<Self>
            + VectorSpace<Scalar = Self::Scalar>,
        Self::N: Increment,
    {
        let (v, a) = projective.truncate();
        if a.is_zero() {
            Some(v)
        }
        else {
            None
        }
    }

    fn into_homogeneous(self) -> Self::ProjectiveSpace
    where
        Self: Homogeneous + ExtendInto<<Self as Homogeneous>::ProjectiveSpace>,
        Self::ProjectiveSpace:
            FiniteDimensional<N = ProjectiveDimensions<Self>> + VectorSpace<Scalar = Self::Scalar>,
        Self::N: Increment,
    {
        self.extend(Zero::zero())
    }
}

pub trait InnerSpace: VectorSpace {
    fn dot(self, other: Self) -> Self::Scalar {
        self.zip_map(other, |a, b| a * b).sum()
    }

    fn project(self, other: Self) -> Self {
        let n = other.dot(self);
        let d = self.dot(self);
        self * (n / d)
    }

    // TODO: Consider removing this. Instead, document the fact that `dot`
    //       produces the square magnitude in a metric space.
    fn square_magnitude(self) -> Self::Scalar {
        self.dot(self)
    }
}

pub trait DualSpace: FiniteDimensional + VectorSpace {
    type Dual: DualSpace<Dual = Self>
        + FiniteDimensional<N = Self::N>
        + VectorSpace<Scalar = Self::Scalar>;

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

    fn scalar_component(&self, row: usize, column: usize) -> Option<Self::Scalar>;

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
    + Adjunct<Item = <<Self as AffineSpace>::Translation as VectorSpace>::Scalar>
    + Converged
    + Copy
    + Fold
    + Map<<<Self as AffineSpace>::Translation as VectorSpace>::Scalar, Output = Self>
    + PartialEq
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
    AffineSpace<Translation = <Self as EuclideanSpace>::CoordinateSpace>
    + AsPosition<Position = Self>
    + FiniteDimensional
{
    type CoordinateSpace: Basis + InnerSpace + FiniteDimensional<N = <Self as FiniteDimensional>::N>;

    fn origin() -> Self {
        Self::converged(Scalar::<Self>::zero())
    }

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

    fn from_xyzw(x: Scalar<Self>, y: Scalar<Self>, z: Scalar<Self>, w: Scalar<Self>) -> Self
    where
        Self: FiniteDimensional<N = U4>,
    {
        Self::from_coordinates(Vector::<Self>::from_xyzw(x, y, z, w))
    }

    fn into_x(self) -> Scalar<Self>
    where
        Self: FiniteDimensional<N = U1>,
    {
        self.into_coordinates().into_x()
    }

    fn into_xy(self) -> (Scalar<Self>, Scalar<Self>)
    where
        Self: FiniteDimensional<N = U2>,
    {
        self.into_coordinates().into_xy()
    }

    fn into_xyz(self) -> (Scalar<Self>, Scalar<Self>, Scalar<Self>)
    where
        Self: FiniteDimensional<N = U3>,
    {
        self.into_coordinates().into_xyz()
    }

    fn into_xyzw(self) -> (Scalar<Self>, Scalar<Self>, Scalar<Self>, Scalar<Self>)
    where
        Self: FiniteDimensional<N = U4>,
    {
        self.into_coordinates().into_xyzw()
    }

    fn x(&self) -> Scalar<Self>
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U0, Output = Greater>,
    {
        self.get(0).copied().expect_dimensional_component()
    }

    fn y(&self) -> Scalar<Self>
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U1, Output = Greater>,
    {
        self.get(1).copied().expect_dimensional_component()
    }

    fn z(&self) -> Scalar<Self>
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U2, Output = Greater>,
    {
        self.get(2).copied().expect_dimensional_component()
    }

    fn w(&self) -> Scalar<Self>
    where
        Self: FiniteDimensional,
        Self::N: Cmp<U3, Output = Greater>,
    {
        self.get(3).copied().expect_dimensional_component()
    }

    fn from_homogeneous(projective: Projective<Self>) -> Option<Self>
    where
        Self::CoordinateSpace: Homogeneous,
        Projective<Self>: FiniteDimensional<N = ProjectiveDimensions<Self>>
            + TruncateInto<Self::CoordinateSpace>
            + VectorSpace<Scalar = Scalar<Self>>,
        Self::N: Increment,
    {
        let (v, a) = projective.truncate();
        if a.is_zero() {
            None
        }
        else {
            Some(Self::from_coordinates(v * (Scalar::<Self>::one() / a)))
        }
    }

    fn into_homogeneous(self) -> Projective<Self>
    where
        Self::CoordinateSpace: Homogeneous + ExtendInto<Projective<Self>>,
        Projective<Self>:
            FiniteDimensional<N = ProjectiveDimensions<Self>> + VectorSpace<Scalar = Scalar<Self>>,
        Self::N: Increment,
    {
        self.into_coordinates().extend(One::one())
    }
}

pub trait Homogeneous: FiniteDimensional + VectorSpace
where
    Self::N: Increment,
{
    type ProjectiveSpace: FiniteDimensional<N = ProjectiveDimensions<Self>>
        + VectorSpace<Scalar = Self::Scalar>;
}
