#![cfg(all(feature = "array", target_os = "linux"))]

use ndarray::{Array, Ix2};
use ndarray_linalg::convert;
use ndarray_linalg::layout::MatrixLayout;
use ndarray_linalg::svd::SVDInto;
use typenum::type_operators::Cmp;
use typenum::{Greater, Unsigned, U2};

use crate::query::{Plane, Unit};
use crate::space::{EuclideanSpace, FiniteDimensional, Scalar, Vector};
use crate::{FromItems, IntoItems};

pub use ndarray_linalg::types::Scalar as ArrayScalar;

impl<S> Plane<S>
where
    S: EuclideanSpace + FiniteDimensional,
    <S as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
{
    pub fn from_points<I>(points: I) -> Option<Self>
    where
        Scalar<S>: ArrayScalar,
        Vector<S>: FromItems + IntoItems,
        I: AsRef<[S]> + Clone + IntoIterator<Item = S>,
    {
        svd_ev_plane(points)
    }
}

/// Maps columnar data into a two-dimensional array.
///
/// Produces a two-dimensional array that forms a matrix from each input
/// column.
fn map_into_array<I, T, F>(columns: I, f: F) -> Option<Array<T::Item, Ix2>>
where
    I: AsRef<[<I as IntoIterator>::Item]> + IntoIterator,
    T: FiniteDimensional + IntoItems,
    T::Item: ArrayScalar,
    F: Fn(I::Item) -> T,
{
    let n = columns.as_ref().len();
    convert::into_matrix(
        MatrixLayout::F((n as i32, <T as FiniteDimensional>::N::USIZE as i32)),
        columns
            .into_iter()
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
    Scalar<S>: ArrayScalar,
    Vector<S>: FromItems + IntoItems,
    I: AsRef<[S]> + Clone + IntoIterator<Item = S>,
{
    let centroid = EuclideanSpace::centroid(points.clone())?;
    let m = map_into_array(points, |point| point - centroid)?;
    // TODO: Fails at runtime if `V^T` is not requested.
    if let Ok((Some(u), sigma, _)) = m.svd_into(true, true) {
        let i = sigma
            .iter()
            .enumerate()
            .min_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())?
            .0;
        if i < u.cols() {
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

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use crate::query::Plane;
    use crate::space::{EuclideanSpace, Vector};

    type E3 = Point3<f64>;

    #[test]
    fn determined_svd_ev_plane_e3() {
        let plane = Plane::<E3>::from_points(vec![
            EuclideanSpace::from_xyz(1.0, 0.0, 0.0),
            EuclideanSpace::from_xyz(0.5, 0.5, 0.0),
            EuclideanSpace::from_xyz(0.0, 1.0, 0.0),
        ])
        .unwrap();
        assert_eq!(Vector::<E3>::z(), plane.normal.get().clone());
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
        assert_eq!(Vector::<E3>::z(), plane.normal.get().clone());
    }
}
