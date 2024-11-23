//! LAPACK and non-trivial linear algebra.

#![cfg(all(feature = "lapack", target_os = "linux"))]

use ndarray::{Array, Ix2};
use ndarray_linalg::convert;
use ndarray_linalg::layout::MatrixLayout;
use ndarray_linalg::svd::SVDInto;
use typenum::type_operators::Cmp;
use typenum::{Greater, Unsigned, U2};

use crate::adjunct::{FromItems, IntoItems};
use crate::query::{Plane, Unit};
use crate::space::{EuclideanSpace, FiniteDimensional, Scalar, Vector};

/// Scalar types that can be used with LAPACK.
pub trait Lapack: ndarray_linalg::types::Lapack + ndarray_linalg::types::Scalar {}

impl<T> Lapack for T where T: ndarray_linalg::types::Lapack + ndarray_linalg::types::Scalar {}

impl<S> Plane<S>
where
    S: EuclideanSpace + FiniteDimensional,
    <S as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
{
    pub fn from_points<I>(points: I) -> Option<Self>
    where
        Scalar<S>: Lapack,
        Vector<S>: FromItems + IntoItems,
        I: AsRef<[S]>,
    {
        svd_ev_plane(points)
    }
}

/// Maps columnar data into a two-dimensional array.
///
/// Produces a two-dimensional array that forms a matrix from each input column.
fn map_into_array<I, T, U, F>(columns: I, f: F) -> Option<Array<U::Item, Ix2>>
where
    I: AsRef<[T]>,
    U: FiniteDimensional + IntoItems,
    F: Fn(&T) -> U,
{
    let columns = columns.as_ref();
    let n = columns.len();
    convert::into_matrix(
        MatrixLayout::F {
            col: n as i32,
            lda: <U as FiniteDimensional>::N::USIZE as i32,
        },
        columns
            .iter()
            .map(f)
            .flat_map(|column| column.into_items())
            .collect(),
    )
    .ok()
}

// TODO: Handle edge cases and improve error handling.
/// Computes a best-fit plane from a set of points.
///
/// The plane is fit using least squares via a singular value decomposition.
fn svd_ev_plane<S, I>(points: I) -> Option<Plane<S>>
where
    S: EuclideanSpace + FiniteDimensional,
    <S as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
    Scalar<S>: Lapack,
    Vector<S>: FromItems + IntoItems,
    I: AsRef<[S]>,
{
    let points = points.as_ref();
    let centroid = EuclideanSpace::centroid(points.iter().cloned())?;
    let m = map_into_array(points, |point| *point - centroid)?;
    // TODO: Fails at runtime if `V^T` is not requested.
    if let Ok((Some(u), sigma, _)) = m.svd_into(true, true) {
        let i = sigma
            .iter()
            .enumerate()
            .min_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())?
            .0;
        if i < u.ncols() {
            let normal = Vector::<S>::from_items(u.column(i).into_iter().cloned())?;
            Some(Plane {
                origin: centroid,
                normal: Unit::try_from_inner(normal)?,
            })
        }
        else {
            None
        }
    }
    else {
        None
    }
}

#[cfg(all(test, feature = "geometry-nalgebra"))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Point3;

    use crate::query::Plane;
    use crate::space::{EuclideanSpace, Vector};

    type E3 = Point3<f64>;

    #[test]
    fn determined_svd_ev_plane_e3() {
        let plane = Plane::<E3>::from_points(vec![
            EuclideanSpace::from_xyz(1.0, 0.0, 0.0),
            EuclideanSpace::from_xyz(0.0, 1.0, 0.0),
            EuclideanSpace::from_xyz(0.0, 0.0, 0.0),
        ])
        .unwrap();
        assert_abs_diff_eq!(Vector::<E3>::z(), plane.normal.get().clone());
    }

    #[test]
    fn overdetermined_svd_ev_plane_e3() {
        let plane = Plane::<E3>::from_points(vec![
            EuclideanSpace::from_xyz(1.0, 1.0, 0.0),
            EuclideanSpace::from_xyz(2.0, 1.0, 0.0),
            EuclideanSpace::from_xyz(3.0, 1.0, 0.0),
            EuclideanSpace::from_xyz(2.0, 1.0, 0.0),
            EuclideanSpace::from_xyz(2.0, 2.0, 0.0),
            EuclideanSpace::from_xyz(2.0, 3.0, 0.0),
        ])
        .unwrap();
        assert_abs_diff_eq!(Vector::<E3>::z(), plane.normal.get().clone());
    }
}
