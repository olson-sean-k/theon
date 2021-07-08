use decorum::R64;
use itertools::iproduct;

use crate::adjunct::{Fold, FromItems, ZipMap};
use crate::space::{DualSpace, FiniteDimensional, Matrix, VectorSpace};

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
