//! Spatial queries.
//!
//! This module provides types and traits for performing spatial queries.

use decorum::cmp::IntrinsicOrd;
use decorum::{Infinite, Real};
use num::{Bounded, Signed, Zero};
use std::ops::Neg;
use typenum::type_operators::Cmp;
use typenum::{Greater, U0, U1, U2};

use crate::adjunct::{Fold, ZipMap};
use crate::ops::Dot;
use crate::space::{Basis, EuclideanSpace, FiniteDimensional, InnerSpace, Scalar, Vector};

// Intersections are implemented for types with a lesser lexographical order.
// For example, `Intersection` is implemented for `Aabb` before `Plane`, with
// `Plane` having a trivial symmetric implementation.
/// Intersection of geometric objects.
///
/// Determines if a pair of objects intersects and produces data describing the
/// intersection. Each set of objects produces its own intersection data as the
/// `Output` type.
///
/// A symmetrical implementation is provided for heterogeneous pairs:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate theon;
/// #
/// # use nalgebra::Point3;
/// # use theon::query::{Intersection, Line, Plane, Unit};
/// # use theon::space::EuclideanSpace;
/// #
/// # type E3 = Point3<f64>;
/// #
/// # let line = Line::<E3> {
/// #     origin: EuclideanSpace::origin(),
/// #     direction: Unit::x(),
/// # };
/// # let plane = Plane::<E3> {
/// #     origin: EuclideanSpace::from_xyz(1.0, 0.0, 0.0),
/// #     normal: Unit::x(),
/// # };
/// // These queries are equivalent.
/// if let Some(t) = line.intersection(&plane) { /* ... */ }
/// if let Some(t) = plane.intersection(&line) { /* ... */ }
/// ```
///
/// # Examples
///
/// Testing for intersection of an axis-aligned bounding box and a ray:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate theon;
/// #
/// use nalgebra::Point2;
/// use theon::query::{Aabb, Intersection, Ray, Unit};
/// use theon::space::{Basis, EuclideanSpace, VectorSpace};
///
/// type E2 = Point2<f64>;
///
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
/// ```
pub trait Intersection<T> {
    type Output;

    fn intersection(&self, other: &T) -> Option<Self::Output>;
}
macro_rules! impl_symmetrical_intersection {
    ($a:ident, $b:ident $(,)?) => {
        /// Symmetrical intersection.
        impl<S> Intersection<$a<S>> for $b<S>
        where
            S: EuclideanSpace,
            $a<S>: Intersection<$b<S>>,
        {
            type Output = <$a<S> as Intersection<$b<S>>>::Output;

            fn intersection(&self, other: &$a<S>) -> Option<Self::Output> {
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
    /// use theon::query::Unit;
    /// use theon::space::Basis;
    ///
    /// type R3 = Vector3<f64>;
    ///
    /// let unit = Unit::<R3>::try_from_inner(Basis::x()).unwrap();
    /// ```
    pub fn try_from_inner(inner: S) -> Option<Self> {
        inner.normalize().map(|inner| Unit { inner })
    }

    pub fn into_inner(self) -> S {
        self.inner
    }

    pub fn x() -> Self
    where
        S: Basis + FiniteDimensional,
        S::N: Cmp<U0, Output = Greater>,
    {
        Self::try_from_inner(Basis::x()).expect("zero-vector")
    }

    pub fn y() -> Self
    where
        S: Basis + FiniteDimensional,
        S::N: Cmp<U1, Output = Greater>,
    {
        Self::try_from_inner(Basis::y()).expect("zero-vector")
    }

    pub fn z() -> Self
    where
        S: Basis + FiniteDimensional,
        S::N: Cmp<U2, Output = Greater>,
    {
        Self::try_from_inner(Basis::z()).expect("zero-vector")
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

    pub fn reverse(self) -> Self {
        // TODO: This assumes that the `Neg` implementation does not affect
        //       magnitude.
        let Unit { inner, .. } = self;
        Self::from_inner_unchecked(-inner)
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

impl<S> Neg for Unit<S>
where
    S: InnerSpace,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.reverse()
    }
}

/// Line.
///
/// Describes a line containing an _origin_ point and a _direction_. Lines
/// extend infinitely from their origin along their direction $\hat{u}$. Unlike
/// `Ray`, the direction component of `Line` extends in both the positive and
/// negative.
///
/// This representation is typically known as the _vector form_ $P_0 +
/// t\hat{u}$ where $t$ is some non-zero _time of impact_.
#[derive(Clone)]
pub struct Line<S>
where
    S: EuclideanSpace,
{
    /// The origin or contained point of the line.
    pub origin: S,
    /// The unit direction(s) in which the line extends from its origin.
    pub direction: Unit<Vector<S>>,
}

impl<S> Line<S>
where
    S: EuclideanSpace,
{
    pub fn into_ray(self) -> Ray<S> {
        let Line { origin, direction } = self;
        Ray { origin, direction }
    }
}

impl<S> Copy for Line<S>
where
    S: EuclideanSpace,
    Vector<S>: Copy,
{
}

impl<S> Default for Line<S>
where
    S: EuclideanSpace,
{
    fn default() -> Self {
        Line {
            origin: S::origin(),
            direction: Unit::default(),
        }
    }
}

/// Intersection of a line and plane.
impl<S> Intersection<Plane<S>> for Line<S>
where
    S: EuclideanSpace + FiniteDimensional,
    <S as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
{
    /// The _time of impact_ of the intersection.
    ///
    /// The time of impact $t$ describes the distance from the line's origin
    /// point at which the intersection occurs.
    type Output = Scalar<S>;

    // TODO: Detect lines that lie within the plane.
    /// Determines the _time of impact_ of a `Plane` intersection with a
    /// `Line`.
    ///
    /// Given a line formed from an origin $P_0$ and a unit direction
    /// $\hat{u}$, the point of intersection with the plane is $P_0 +
    /// (t\hat{u})$.
    fn intersection(&self, plane: &Plane<S>) -> Option<Self::Output> {
        let line = self;
        let direction = *line.direction.get();
        let normal = *plane.normal.get();
        let product = direction.dot(normal);
        if product != Zero::zero() {
            Some((plane.origin - line.origin).dot(normal) / product)
        }
        else {
            None
        }
    }
}
impl_symmetrical_intersection!(Line, Plane);

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
    pub fn into_line(self) -> Line<S> {
        let Ray { origin, direction } = self;
        Line { origin, direction }
    }

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
{
    fn default() -> Self {
        Ray {
            origin: S::origin(),
            direction: Unit::default(),
        }
    }
}

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
        Scalar<S>: IntrinsicOrd,
    {
        let mut min = S::origin();
        let mut max = S::origin();
        for point in points {
            min = min.per_item_min_or_undefined(point);
            max = max.per_item_max_or_undefined(point);
        }
        Aabb {
            origin: min,
            extent: max - min,
        }
    }

    pub fn endpoint(&self) -> S {
        self.origin + self.extent
    }

    pub fn upper_bound(&self) -> S
    where
        Scalar<S>: IntrinsicOrd,
    {
        self.origin.per_item_max_or_undefined(self.endpoint())
    }

    pub fn lower_bound(&self) -> S
    where
        Scalar<S>: IntrinsicOrd,
    {
        self.origin.per_item_min_or_undefined(self.endpoint())
    }

    /// Gets the Lebesgue measure ($n$-dimensional volume) of the bounding box.
    ///
    /// This value is analogous to _length_, _area_, and _volume_ in one, two,
    /// and three dimensions, respectively.
    pub fn volume(&self) -> Scalar<S> {
        self.origin
            .zip_map(self.endpoint(), |a, b| (a - b).abs())
            .product()
    }

    pub fn union(&self, aabb: &Self) -> Self
    where
        Scalar<S>: IntrinsicOrd,
    {
        let origin = self
            .lower_bound()
            .per_item_min_or_undefined(aabb.lower_bound());
        let extent = self
            .upper_bound()
            .per_item_max_or_undefined(aabb.upper_bound())
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

/// Intersection of axis-aligned bounding boxes.
impl<S> Intersection<Aabb<S>> for Aabb<S>
where
    S: EuclideanSpace,
{
    type Output = Self;

    fn intersection(&self, _: &Aabb<S>) -> Option<Self::Output> {
        None // TODO:
    }
}

/// Intersection of an axis-aligned bounding box and ray.
impl<S> Intersection<Ray<S>> for Aabb<S>
where
    S: EuclideanSpace,
    Scalar<S>: Bounded + Infinite + IntrinsicOrd + Signed,
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
    fn intersection(&self, ray: &Ray<S>) -> Option<Self::Output> {
        // Avoid computing `NaN`s. Note that multiplying by the inverse (instead
        // of dividing) avoids dividing zero by zero, but does not avoid
        // multiplying zero by infinity.
        let pdiv = |a: Scalar<S>, b: Scalar<S>| {
            if a.is_zero() {
                a
            }
            else {
                a / b
            }
        };
        let aabb = self;
        let direction = *ray.direction.get();
        let origin = (aabb.origin - ray.origin).zip_map(direction, pdiv);
        let endpoint = ((aabb.endpoint()) - ray.origin).zip_map(direction, pdiv);
        let min = origin
            .per_item_min_or_undefined(endpoint)
            .max_or_undefined();
        let max = origin
            .per_item_max_or_undefined(endpoint)
            .min_or_undefined();
        if max.is_negative() || min > max || min.is_undefined() || max.is_undefined() {
            None
        }
        else {
            Some((min, max))
        }
    }
}
impl_symmetrical_intersection!(Aabb, Ray);

#[derive(Clone)]
pub struct Plane<S>
where
    S: EuclideanSpace,
{
    pub origin: S,
    pub normal: Unit<Vector<S>>,
}

impl<S> Copy for Plane<S>
where
    S: EuclideanSpace,
    Vector<S>: Copy,
{
}

/// Intersection of a plane and a ray.
impl<S> Intersection<Ray<S>> for Plane<S>
where
    S: EuclideanSpace + FiniteDimensional,
    <S as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
    Scalar<S>: Signed,
{
    /// The _time of impact_ of the intersection.
    ///
    /// The time of impact $t$ describes the distance along the half-line from
    /// the ray's origin at which the intersection occurs.
    type Output = Scalar<S>;

    // TODO: Detect rays that lie within the plane.
    /// Determines the _time of impact_ of a `Ray` intersection with a `Plane`.
    ///
    /// Given a ray formed by an origin $P_0$ and a unit direction $\hat{u}$,
    /// the point of intersection with the plane is $P_0 + (t\hat{u})$.
    fn intersection(&self, ray: &Ray<S>) -> Option<Self::Output> {
        let plane = self;
        ray.into_line().intersection(plane).and_then(|t| {
            if t.is_positive() {
                Some(t)
            }
            else {
                None
            }
        })
    }
}
impl_symmetrical_intersection!(Plane, Ray);

#[cfg(all(test, feature = "geometry-nalgebra"))]
mod tests {
    use decorum::N64;
    use nalgebra::{Point2, Point3};

    use crate::adjunct::Converged;
    use crate::query::{Aabb, Intersection, Plane, Ray, Unit};
    use crate::space::{Basis, EuclideanSpace};

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

    // Ensure that certain values do not produce `NaN`s when querying the
    // intersection of `Aabb` and `Ray`.
    #[test]
    fn aabb_ray_intersection_nan() {
        let aabb = Aabb::<Point2<N64>> {
            origin: EuclideanSpace::origin(),
            extent: Converged::converged(1.0.into()),
        };
        let ray = Ray::<Point2<N64>> {
            origin: EuclideanSpace::origin(),
            direction: Unit::x(),
        };
        assert_eq!(Some((0.0.into(), 1.0.into())), ray.intersection(&aabb));
    }

    #[test]
    fn plane_ray_intersection_e3() {
        let plane = Plane::<E3> {
            origin: EuclideanSpace::from_xyz(0.0, 0.0, 1.0),
            normal: Unit::try_from_inner(Basis::z()).unwrap(),
        };
        let ray = Ray::<E3> {
            origin: EuclideanSpace::origin(),
            direction: Unit::try_from_inner(Basis::z()).unwrap(),
        };
        assert_eq!(Some(1.0), ray.intersection(&plane));
        assert_eq!(None, ray.reverse().intersection(&plane));
    }
}
