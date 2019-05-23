//! Spatial queries.
//!
//! This module provides types and traits for performing spatial queries.

use decorum::{Infinite, Real};
use num::{Bounded, One, Zero};
use std::ops::Neg;

use crate::ops::{Reduce, ZipMap};
use crate::space::{Basis, EuclideanSpace, InnerSpace, Scalar, Vector};
use crate::Lattice;

/// Pair-wise intersection.
///
/// Determines if two entities intersect and produces data describing the
/// intersection. Each pairing of entities produces its own intersection
/// data.
///
/// # Examples
///
/// Testing for intersecions of an axis-aligned bounding box and a ray:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate theon;
/// #
/// use nalgebra::Point2;
/// use theon::space::{Basis, EuclideanSpace, VectorSpace};
/// use theon::query::{Aabb, Intersection, Ray, Unit};
///
/// type E2 = Point2<f64>;
///
/// # fn main() {
/// let aabb = Aabb::<E2> {
///     origin: EuclideanSpace::from_xy(1.0, -1.0),
///     extent: VectorSpace::from_xy(2.0, 2.0),
/// };
/// let ray = Ray::<E2> {
///     origin: EuclideanSpace::origin(),
///     direction: Unit::try_from_inner(Basis::x()).unwrap(),
/// };
/// if let Some((min, max)) = ray.intersection(&aabb) {
///     // ...
/// }
/// # }
/// ```
pub trait Intersection<T> {
    type Output;

    fn intersection(&self, _: &T) -> Option<Self::Output>;
}
macro_rules! impl_reciprocal_intersection {
    (provider => $t:ident, target => $u:ident) => {
        impl<S> Intersection<$t<S>> for $u<S>
        where
            S: EuclideanSpace,
            $t<S>: Intersection<$u<S>>,
        {
            type Output = <$t<S> as Intersection<$u<S>>>::Output;

            fn intersection(&self, other: &$t<S>) -> Option<Self::Output> {
                other.intersection(self)
            }
        }
    };
}

/// Unit vector.
///
/// Primarily represents a direction within an `InnerSpace`.
#[derive(Clone, Copy)]
pub struct Unit<S>
where
    S: InnerSpace,
{
    inner: S,
}

impl<S> Unit<S>
where
    S: InnerSpace,
{
    fn from_inner_unchecked(inner: S) -> Self {
        Unit { inner }
    }

    /// Creates a `Unit` from a non-zero magnitude vector.
    ///
    /// The given vector is normalized. If the vector's magnitude is zero, then
    /// `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate theon;
    /// #
    /// use nalgebra::Vector3;
    /// use theon::space::Basis;
    /// use theon::query::Unit;
    ///
    /// type R3 = Vector3<f64>;
    ///
    /// # fn main() {
    /// let unit = Unit::<R3>::try_from_inner(Basis::x()).unwrap();
    /// # }
    /// ```
    pub fn try_from_inner(inner: S) -> Option<Self> {
        inner.normalize().map(|inner| Unit { inner })
    }

    pub fn into_inner(self) -> S {
        self.inner
    }

    pub fn get(&self) -> &S {
        self.as_ref()
    }

    #[must_use]
    pub fn try_set(&mut self, inner: S) -> Option<&S> {
        if let Some(inner) = inner.normalize() {
            self.inner = inner;
            Some(&self.inner)
        }
        else {
            None
        }
    }
}

impl<S> AsRef<S> for Unit<S>
where
    S: InnerSpace,
{
    fn as_ref(&self) -> &S {
        &self.inner
    }
}

impl<S> Default for Unit<S>
where
    S: Basis + InnerSpace,
{
    fn default() -> Self {
        Unit {
            inner: S::canonical_basis_component(0).unwrap(),
        }
    }
}

/// Ray or half-line.
///
/// Describes a decomposed line with an _origin_ or _initial point_ and a
/// _direction_. Rays extend infinitely from their origin. The origin $P_0$ and
/// the point $P_0 + \hat{u}$ (where $\hat{u}$ is the direction of the ray)
/// form a half-line originating from $P_0$.
#[derive(Clone)]
pub struct Ray<S>
where
    S: EuclideanSpace,
{
    /// The origin or initial point of the ray.
    pub origin: S,
    /// The unit direction in which the ray extends from its origin.
    pub direction: Unit<Vector<S>>,
}

impl<S> Ray<S>
where
    S: EuclideanSpace,
{
    /// Reverses the direction of the ray.
    ///
    /// Reversing a ray yields its _opposite_, with the same origin and the
    /// opposing half-line.
    pub fn reverse(self) -> Self {
        let Ray { origin, direction } = self;
        Ray {
            origin,
            direction: Unit::from_inner_unchecked(-direction.into_inner()),
        }
    }
}

impl<S> Copy for Ray<S>
where
    S: EuclideanSpace,
    Vector<S>: Copy,
{
}

impl<S> Default for Ray<S>
where
    S: EuclideanSpace,
    Vector<S>: Basis,
{
    fn default() -> Self {
        Ray {
            origin: S::origin(),
            direction: Unit::default(),
        }
    }
}

// TODO: This implementation requires `INF` and `-INF` representations.
impl<S> Intersection<Aabb<S>> for Ray<S>
where
    S: EuclideanSpace,
    Scalar<S>: Bounded + Infinite + Lattice,
{
    /// The minimum and maximum _times of impact_ of the intersection.
    ///
    /// The times of impact $t_{min}$ and $t_{max}$ describe the distance along
    /// the half-line from the ray's origin at which the intersection occurs.
    type Output = (Scalar<S>, Scalar<S>);

    /// Determines the minimum and maximum _times of impact_ of a `Ray`
    /// intersection with an `Aabb`.
    ///
    /// Given a ray formed by an origin $P_0$ and a unit direction $\hat{u}$,
    /// the nearest point of intersection is $P_0 + (t_{min}\hat{u})$.
    ///
    /// # Examples
    ///
    /// Determine the point of impact between a ray and axis-aligned bounding box:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate theon;
    /// #
    /// use nalgebra::Point2;
    /// use theon::space::{Basis, EuclideanSpace, VectorSpace};
    /// use theon::query::{Aabb, Intersection, Ray, Unit};
    ///
    /// type E2 = Point2<f64>;
    ///
    /// # fn main() {
    /// let aabb = Aabb::<E2> {
    ///     origin: EuclideanSpace::from_xy(1.0, -1.0),
    ///     extent: VectorSpace::from_xy(2.0, 2.0),
    /// };
    /// let ray = Ray::<E2> {
    ///     origin: EuclideanSpace::origin(),
    ///     direction: Unit::try_from_inner(Basis::x()).unwrap(),
    /// };
    /// let (min, _) = ray.intersection(&aabb).unwrap();
    /// let point = ray.origin + (ray.direction.get() * min);
    /// # }
    fn intersection(&self, aabb: &Aabb<S>) -> Option<Self::Output> {
        let direction = self.direction.get().clone();
        let origin = (aabb.origin - self.origin).zip_map(direction, |a, b| a / b);
        let endpoint = ((aabb.endpoint()) - self.origin).zip_map(direction, |a, b| a / b);
        let min = origin
            .zip_map(endpoint, |a, b| crate::partial_min(a, b))
            .reduce(Bounded::min_value(), |max, a| crate::partial_max(max, a));
        let max = origin
            .zip_map(endpoint, |a, b| crate::partial_max(a, b))
            .reduce(Bounded::max_value(), |min, a| crate::partial_min(min, a));
        if max < Zero::zero() || min > max {
            None
        }
        else {
            Some((min, max))
        }
    }
}
impl_reciprocal_intersection!(provider => Ray, target => Aabb);

impl<S> Neg for Ray<S>
where
    S: EuclideanSpace,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.reverse()
    }
}

/// Axis-aligned bounding box.
///
/// Represents an $n$-dimensional volume along each basis vector of a Euclidean
/// space. The bounding box is defined by the region between its _origin_ and
/// _endpoint_.
#[derive(Clone)]
pub struct Aabb<S>
where
    S: EuclideanSpace,
{
    /// The _origin_ of the bounding box.
    ///
    /// The origin does **not** necessarily represent the lower or upper bound
    /// of the `Aabb`. See `lower_bound` and `upper_bound`.
    pub origin: S,
    /// The _extent_ of the bounding box.
    ///
    /// The extent describes the endpoint as a translation from the origin. The
    /// endpoint $P_E$ is formed by $P_0 + \vec{v}$, where $P_0$ is the origin
    /// and $\vec{v}$ is the extent.
    pub extent: Vector<S>,
}

impl<S> Aabb<S>
where
    S: EuclideanSpace,
{
    /// Creates an `Aabb` from a set of points.
    ///
    /// The bounding box is formed from the lower and upper bounds of the
    /// points. If the set of points is empty, then the `Aabb` will sit at the
    /// origin with zero volume.
    pub fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator<Item = S>,
    {
        let mut min = S::origin();
        let mut max = S::origin();
        for point in points {
            min = min.zip_map(point, |a, b| crate::partial_min(a, b));
            max = max.zip_map(point, |a, b| crate::partial_max(a, b));
        }
        Aabb {
            origin: min,
            extent: max - min,
        }
    }

    pub fn endpoint(&self) -> S {
        self.origin + self.extent
    }

    pub fn upper_bound(&self) -> S {
        self.origin
            .zip_map(self.endpoint(), |a, b| crate::partial_max(a, b))
    }

    pub fn lower_bound(&self) -> S {
        self.origin
            .zip_map(self.endpoint(), |a, b| crate::partial_min(a, b))
    }

    /// Gets the Lebesgue measure ($n$-dimensional volume) of the bounding box.
    ///
    /// This value is analogous to _length_, _area_, and _volume_ in one, two,
    /// and three dimensions, respectively.
    pub fn volume(&self) -> Scalar<S> {
        self.origin
            .zip_map(self.endpoint(), |a, b| (a - b).abs())
            .reduce(One::one(), |product, a| product * a)
    }

    pub fn union(&self, aabb: &Self) -> Self {
        let origin = self
            .lower_bound()
            .zip_map(aabb.lower_bound(), |a, b| crate::partial_min(a, b));
        let extent = self
            .upper_bound()
            .zip_map(aabb.upper_bound(), |a, b| crate::partial_max(a, b))
            - origin;
        Aabb { origin, extent }
    }
}

impl<S> Copy for Aabb<S>
where
    S: EuclideanSpace,
    Vector<S>: Copy,
{
}

impl<S> Default for Aabb<S>
where
    S: EuclideanSpace,
{
    fn default() -> Self {
        Aabb {
            origin: S::origin(),
            extent: Vector::<S>::zero(),
        }
    }
}

impl<S> Intersection<Aabb<S>> for Aabb<S>
where
    S: EuclideanSpace,
{
    type Output = Self;

    fn intersection(&self, _: &Self) -> Option<Self::Output> {
        None // TODO:
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3};

    use crate::query::{Aabb, Intersection, Ray, Unit};
    use crate::space::{Basis, EuclideanSpace};
    use crate::Converged;

    type E2 = Point2<f64>;
    type E3 = Point3<f64>;

    #[test]
    fn aabb_ray_intersection_e2() {
        let aabb = Aabb::<E2> {
            origin: EuclideanSpace::origin(),
            extent: Converged::converged(1.0),
        };
        let ray = Ray::<E2> {
            origin: EuclideanSpace::from_xy(-1.0, 0.5),
            direction: Unit::try_from_inner(Basis::x()).unwrap(),
        };
        assert_eq!(Some((1.0, 2.0)), ray.intersection(&aabb));
        assert_eq!(None, ray.reverse().intersection(&aabb));
    }

    #[test]
    fn aabb_ray_intersection_e3() {
        let aabb = Aabb::<E3> {
            origin: EuclideanSpace::origin(),
            extent: Converged::converged(1.0),
        };
        let ray = Ray::<E3> {
            origin: EuclideanSpace::from_xyz(-1.0, 0.5, 0.5),
            direction: Unit::try_from_inner(Basis::x()).unwrap(),
        };
        assert_eq!(Some((1.0, 2.0)), ray.intersection(&aabb));
        assert_eq!(None, ray.reverse().intersection(&aabb));
    }
}
