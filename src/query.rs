//! Spatial queries.
//!
//! This module provides types and traits for performing spatial queries.

use approx::abs_diff_eq;
use decorum::cmp::EmptyOrd;
use decorum::InfinityEncoding;
use num_traits::bounds::Bounded;
use num_traits::identities::Zero;
use num_traits::real::Real;
use num_traits::sign::Signed;
use std::fmt::{self, Debug, Formatter};
use std::ops::Neg;
use typenum::type_operators::Cmp;
use typenum::{Greater, U0, U1, U2};

use crate::adjunct::{Fold, ZipMap};
use crate::ops::Dot;
use crate::space::{
    Basis, EuclideanSpace, FiniteDimensional, InnerSpace, Scalar, Vector, VectorSpace,
};

// Intersections are implemented for types with a lesser lexographical order. For example,
// `Intersection` is implemented for `Aabb` before `Plane`, with `Plane` having a trivial symmetric
// implementation.
/// Intersection of geometric objects.
///
/// Determines if a pair of objects intersects and produces data describing the intersection. Each
/// set of objects produces its own intersection data as the `Output` type.
///
/// A symmetrical implementation is provided for heterogeneous pairs:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate theon;
/// #
/// # use nalgebra::Point3;
/// # use theon::query::{Intersection, Line, LinePlane, Plane, Unit};
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
/// if let Some(LinePlane::TimeOfImpact(t)) = line.intersection(&plane) { /* ... */ }
/// if let Some(LinePlane::TimeOfImpact(t)) = plane.intersection(&line) { /* ... */ }
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
/// use theon::space::{EuclideanSpace, VectorSpace};
///
/// type E2 = Point2<f64>;
///
/// let aabb = Aabb::<E2> {
///     origin: EuclideanSpace::from_xy(1.0, -1.0),
///     extent: VectorSpace::from_xy(2.0, 2.0),
/// };
/// let ray = Ray::<E2> {
///     origin: EuclideanSpace::origin(),
///     direction: Unit::x(),
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
    ($a:ident $(,)?) => {
        /// Symmetrical intersection.
        impl<S> Intersection<$a<S>> for S
        where
            S: EuclideanSpace,
            $a<S>: Intersection<S>,
        {
            type Output = <$a<S> as Intersection<S>>::Output;

            fn intersection(&self, other: &$a<S>) -> Option<Self::Output> {
                other.intersection(self)
            }
        }
    };
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// The given vector is normalized. If the vector's magnitude is zero, then `None` is returned.
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
    /// let unit = Unit::<R3>::try_from_inner(Basis::i()).unwrap();
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
        Self::from_inner_unchecked(Basis::i())
    }

    pub fn y() -> Self
    where
        S: Basis + FiniteDimensional,
        S::N: Cmp<U1, Output = Greater>,
    {
        Self::from_inner_unchecked(Basis::j())
    }

    pub fn z() -> Self
    where
        S: Basis + FiniteDimensional,
        S::N: Cmp<U2, Output = Greater>,
    {
        Self::from_inner_unchecked(Basis::k())
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
        // TODO: This assumes that the `Neg` implementation does not affect magnitude.
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
/// Describes a line containing an _origin_ point and a _direction_. Lines extend infinitely from
/// their origin along their direction $\hat{u}$. Unlike `Ray`, the direction component of `Line`
/// extends in both the positive and negative.
///
/// This representation is typically known as the _vector form_ $P_0 + t\hat{u}$ where $t$ is some
/// non-zero _time of impact_.
#[derive(Clone, Copy, PartialEq)]
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
    pub fn x() -> Self
    where
        S: FiniteDimensional,
        S::N: Cmp<U0, Output = Greater>,
    {
        Line {
            origin: S::origin(),
            direction: Unit::x(),
        }
    }

    pub fn y() -> Self
    where
        S: FiniteDimensional,
        S::N: Cmp<U1, Output = Greater>,
    {
        Line {
            origin: S::origin(),
            direction: Unit::y(),
        }
    }

    pub fn z() -> Self
    where
        S: FiniteDimensional,
        S::N: Cmp<U2, Output = Greater>,
    {
        Line {
            origin: S::origin(),
            direction: Unit::z(),
        }
    }

    pub fn into_ray(self) -> Ray<S> {
        let Line { origin, direction } = self;
        Ray { origin, direction }
    }
}

// TODO: Provide higher dimensional intercepts, such as the xy-intercept in three dimensions.
impl<S> Line<S>
where
    S: EuclideanSpace + FiniteDimensional<N = U2>,
{
    pub fn slope(&self) -> Option<Scalar<S>> {
        let (x, y) = self.direction.get().into_xy();
        if x.is_zero() {
            None
        }
        else {
            Some(y / x)
        }
    }

    pub fn x_intercept(&self) -> Option<Scalar<S>> {
        self.intersection(&Line::x())
            .and_then(|embedding| match embedding {
                LineLine::Point(point) => Some(point.into_xy().0),
                _ => None,
            })
    }

    pub fn y_intercept(&self) -> Option<Scalar<S>> {
        self.intersection(&Line::y())
            .and_then(|embedding| match embedding {
                LineLine::Point(point) => Some(point.into_xy().1),
                _ => None,
            })
    }
}

impl<S> Debug for Line<S>
where
    S: Debug + EuclideanSpace,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("Line")
            .field("origin", &self.origin)
            .field("direction", &self.direction)
            .finish()
    }
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

/// Intersection of lines.
#[derive(Clone, Copy, PartialEq)]
pub enum LineLine<S>
where
    S: EuclideanSpace,
{
    // Lines and rays typically produce times of impact for point intersections, but this
    // implementation computes the point. While this is a bit inconsistent, it avoids needing to
    // know from which line the time of impact applies.
    Point(S),
    Line(Line<S>),
}

impl<S> LineLine<S>
where
    S: EuclideanSpace,
{
    pub fn into_point(self) -> Option<S> {
        match self {
            LineLine::Point(point) => Some(point),
            _ => None,
        }
    }

    pub fn into_line(self) -> Option<Line<S>> {
        match self {
            LineLine::Line(line) => Some(line),
            _ => None,
        }
    }
}

impl<S> Debug for LineLine<S>
where
    S: Debug + EuclideanSpace,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        match *self {
            LineLine::Point(point) => write!(formatter, "Point({:?})", point),
            LineLine::Line(line) => write!(formatter, "Line({:?})", line),
        }
    }
}

// TODO: Though higher dimensional intersections are probably less useful, consider a more general
//       implementation. This could use projection into two dimensions followed by confirmation in
//       the higher dimension.
/// Intersection of lines in two dimensions.
impl<S> Intersection<Line<S>> for Line<S>
where
    S: EuclideanSpace + FiniteDimensional<N = U2>,
{
    type Output = LineLine<S>;

    fn intersection(&self, other: &Line<S>) -> Option<Self::Output> {
        let (x1, y1) = if (self.origin - other.origin).is_zero() {
            // Detect like origins and avoid zeroes in the numerator by translating the origin.
            (self.origin + *self.direction.get()).into_xy()
        }
        else {
            self.origin.into_xy()
        };
        let (u1, v1) = self.direction.get().into_xy();
        let (x2, y2) = other.origin.into_xy();
        let (u2, v2) = other.direction.get().into_xy();
        let numerator = (u2 * (y1 - y2)) - (v2 * (x1 - x2));
        let denominator = (v2 * u1) - (u2 * v1);
        match (numerator.is_zero(), denominator.is_zero()) {
            (true, true) => Some(LineLine::Line(*self)),
            (false, true) => None,
            _ => {
                let quotient = numerator / denominator;
                Some(LineLine::Point(S::from_xy(
                    x1 + (quotient * u1),
                    y1 + (quotient * v1),
                )))
            }
        }
    }
}

/// Intersection of a line and a plane.
#[derive(Clone, Copy, PartialEq)]
pub enum LinePlane<S>
where
    S: EuclideanSpace,
{
    TimeOfImpact(Scalar<S>),
    Line(Line<S>),
}

impl<S> LinePlane<S>
where
    S: EuclideanSpace,
{
    pub fn into_time_of_impact(self) -> Option<Scalar<S>> {
        match self {
            LinePlane::TimeOfImpact(time) => Some(time),
            _ => None,
        }
    }

    pub fn into_line(self) -> Option<Line<S>> {
        match self {
            LinePlane::Line(line) => Some(line),
            _ => None,
        }
    }
}

impl<S> Debug for LinePlane<S>
where
    S: Debug + EuclideanSpace,
    Scalar<S>: Debug,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        match *self {
            LinePlane::TimeOfImpact(x) => write!(formatter, "TimeOfImpact({:?})", x),
            LinePlane::Line(line) => write!(formatter, "Line({:?})", line),
        }
    }
}

/// Intersection of a line and a plane.
impl<S> Intersection<Plane<S>> for Line<S>
where
    S: EuclideanSpace + FiniteDimensional,
    <S as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
{
    /// The _time of impact_ of a point intersection or the line if it lies within the plane.
    ///
    /// The time of impact $t$ describes the distance from the line's origin point at which the
    /// intersection occurs.
    type Output = LinePlane<S>;

    /// Determines if a line intersects a plane at a point or lies within the plane. Computes the
    /// _time of impact_ of a `Line` for a point intersection.
    ///
    /// Given a line formed from an origin $P_0$ and a unit direction $\hat{u}$, the point of
    /// intersection with the plane is $P_0 + t\hat{u}$.
    fn intersection(&self, plane: &Plane<S>) -> Option<Self::Output> {
        let line = self;
        let direction = *line.direction.get();
        let normal = *plane.normal.get();
        let orientation = direction.dot(normal);
        if abs_diff_eq!(orientation, Zero::zero()) {
            // The line and plane are parallel.
            if abs_diff_eq!((plane.origin - line.origin).dot(normal), Zero::zero()) {
                Some(LinePlane::Line(*line))
            }
            else {
                None
            }
        }
        else {
            // The line and plane are not parallel and must intersect at a
            // point.
            Some(LinePlane::TimeOfImpact(
                (plane.origin - line.origin).dot(normal) / orientation,
            ))
        }
    }
}
impl_symmetrical_intersection!(Line, Plane);

/// Ray or half-line.
///
/// Describes a decomposed line with an _origin_ or _initial point_ and a _direction_. Rays extend
/// infinitely from their origin. The origin $P_0$ and the point $P_0 + \hat{u}$ (where $\hat{u}$
/// is the direction of the ray) form a half-line originating from $P_0$.
#[derive(Clone, Copy, PartialEq)]
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
    /// Reversing a ray yields its _opposite_, with the same origin and the opposing half-line.
    pub fn reverse(self) -> Self {
        let Ray { origin, direction } = self;
        Ray {
            origin,
            direction: Unit::from_inner_unchecked(-direction.into_inner()),
        }
    }
}

impl<S> Debug for Ray<S>
where
    S: Debug + EuclideanSpace,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("Ray")
            .field("origin", &self.origin)
            .field("direction", &self.direction)
            .finish()
    }
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
/// Represents an $n$-dimensional volume along each basis vector of a Euclidean space. The bounding
/// box is defined by the region between its _origin_ and _endpoint_.
#[derive(Clone, Copy, PartialEq)]
pub struct Aabb<S>
where
    S: EuclideanSpace,
{
    /// The _origin_ of the bounding box.
    ///
    /// The origin does **not** necessarily represent the lower or upper bound of the `Aabb`. See
    /// `lower_bound` and `upper_bound`.
    pub origin: S,
    /// The _extent_ of the bounding box.
    ///
    /// The extent describes the endpoint as a translation from the origin. The endpoint $P_E$ is
    /// formed by $P_0 + \vec{v}$, where $P_0$ is the origin and $\vec{v}$ is the extent.
    pub extent: Vector<S>,
}

impl<S> Aabb<S>
where
    S: EuclideanSpace,
{
    /// Creates an `Aabb` from a set of points.
    ///
    /// The bounding box is formed from the lower and upper bounds of the points. If the set of
    /// points is empty, then the `Aabb` will sit at the origin with zero volume.
    pub fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator<Item = S>,
        Scalar<S>: EmptyOrd,
    {
        let mut min = S::origin();
        let mut max = S::origin();
        for point in points {
            min = min.per_item_min_or_empty(point);
            max = max.per_item_max_or_empty(point);
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
        Scalar<S>: EmptyOrd,
    {
        self.origin.per_item_max_or_empty(self.endpoint())
    }

    pub fn lower_bound(&self) -> S
    where
        Scalar<S>: EmptyOrd,
    {
        self.origin.per_item_min_or_empty(self.endpoint())
    }

    /// Gets the Lebesgue measure ($n$-dimensional volume) of the bounding box.
    ///
    /// This value is analogous to _length_, _area_, and _volume_ in one, two, and three
    /// dimensions, respectively.
    pub fn volume(&self) -> Scalar<S> {
        self.origin
            .zip_map(self.endpoint(), |a, b| (a - b).abs())
            .product()
    }

    pub fn union(&self, aabb: &Self) -> Self
    where
        Scalar<S>: EmptyOrd,
    {
        let origin = self.lower_bound().per_item_min_or_empty(aabb.lower_bound());
        let extent = self.upper_bound().per_item_max_or_empty(aabb.upper_bound()) - origin;
        Aabb { origin, extent }
    }
}

impl<S> Debug for Aabb<S>
where
    S: Debug + EuclideanSpace,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("Aabb")
            .field("origin", &self.origin)
            .field("extent", &self.extent)
            .finish()
    }
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

/// Intersection of an axis-aligned bounding box and a point.
impl<S> Intersection<S> for Aabb<S>
where
    S: EuclideanSpace,
    Scalar<S>: EmptyOrd + Signed,
{
    type Output = Vector<S>;

    fn intersection(&self, point: &S) -> Option<Self::Output> {
        let aabb = self;
        let lower = aabb.lower_bound().per_item_max_or_empty(*point);
        let upper = aabb.upper_bound().per_item_min_or_empty(*point);
        if lower == upper {
            Some(*point - aabb.lower_bound())
        }
        else {
            None
        }
    }
}
impl_symmetrical_intersection!(Aabb);

/// Intersection of axis-aligned bounding boxes.
impl<S> Intersection<Aabb<S>> for Aabb<S>
where
    S: EuclideanSpace,
    Scalar<S>: EmptyOrd + Signed,
{
    type Output = Self;

    fn intersection(&self, other: &Aabb<S>) -> Option<Self::Output> {
        let max_lower_bound = self
            .lower_bound()
            .per_item_max_or_empty(other.lower_bound());
        let min_upper_bound = self
            .upper_bound()
            .per_item_min_or_empty(other.upper_bound());
        let difference = min_upper_bound - max_lower_bound;
        if difference.all(|x| (!x.is_empty()) && x.is_positive()) {
            Some(Aabb {
                origin: max_lower_bound,
                extent: difference,
            })
        }
        else {
            None
        }
    }
}

/// Intersection of an axis-aligned bounding box and a ray.
impl<S> Intersection<Ray<S>> for Aabb<S>
where
    S: EuclideanSpace,
    Scalar<S>: Bounded + EmptyOrd + InfinityEncoding + Signed,
{
    /// The minimum and maximum _times of impact_ of the intersection.
    ///
    /// The times of impact $t_{min}$ and $t_{max}$ describe the distance along the half-line from
    /// the ray's origin at which the intersection occurs.
    type Output = (Scalar<S>, Scalar<S>);

    /// Determines the minimum and maximum _times of impact_ of a `Ray` intersection with an
    /// `Aabb`.
    ///
    /// Given a ray formed by an origin $P_0$ and a unit direction $\hat{u}$, the nearest point of
    /// intersection is $P_0 + t_{min}\hat{u}$.
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
    /// use theon::space::{EuclideanSpace, VectorSpace};
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
    ///     direction: Unit::x(),
    /// };
    /// let (min, _) = ray.intersection(&aabb).unwrap();
    /// let point = ray.origin + (ray.direction.get() * min);
    fn intersection(&self, ray: &Ray<S>) -> Option<Self::Output> {
        // Avoid computing `NaN`s. Note that multiplying by the inverse (instead of dividing)
        // avoids dividing zero by zero, but does not avoid multiplying zero by infinity.
        let pdiv = |a: Scalar<S>, b: Scalar<S>| {
            if abs_diff_eq!(a, Zero::zero()) {
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
        let min = origin.per_item_min_or_empty(endpoint).max_or_empty();
        let max = origin.per_item_max_or_empty(endpoint).min_or_empty();
        if max.is_negative() || min > max || min.is_empty() || max.is_empty() {
            None
        }
        else {
            Some((min, max))
        }
    }
}
impl_symmetrical_intersection!(Aabb, Ray);

//impl<S> PartialEq for Aabb<S>
//where
//    S: EuclideanSpace,
//{
//}

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

impl<S> Debug for Plane<S>
where
    S: Debug + EuclideanSpace,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("Plane")
            .field("origin", &self.origin)
            .field("normal", &self.normal)
            .finish()
    }
}

/// Intersection of a plane and a ray.
#[derive(Clone, Copy, PartialEq)]
pub enum PlaneRay<S>
where
    S: EuclideanSpace,
{
    TimeOfImpact(Scalar<S>),
    Ray(Ray<S>),
}

impl<S> PlaneRay<S>
where
    S: EuclideanSpace,
{
    pub fn into_time_of_impact(self) -> Option<Scalar<S>> {
        match self {
            PlaneRay::TimeOfImpact(time) => Some(time),
            _ => None,
        }
    }

    pub fn into_ray(self) -> Option<Ray<S>> {
        match self {
            PlaneRay::Ray(ray) => Some(ray),
            _ => None,
        }
    }
}

impl<S> Debug for PlaneRay<S>
where
    S: Debug + EuclideanSpace,
    Scalar<S>: Debug,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        match *self {
            PlaneRay::TimeOfImpact(x) => write!(formatter, "TimeOfImpact({:?})", x),
            PlaneRay::Ray(ray) => write!(formatter, "Ray({:?})", ray),
        }
    }
}

/// Intersection of a plane and a ray.
impl<S> Intersection<Ray<S>> for Plane<S>
where
    S: EuclideanSpace + FiniteDimensional,
    <S as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
    Scalar<S>: Signed,
{
    /// The _time of impact_ of a point intersection or the ray if it lies within the plane.
    ///
    /// The time of impact $t$ describes the distance along the half-line from the ray's origin at
    /// which the intersection occurs.
    type Output = PlaneRay<S>;

    /// Determines if a ray intersects a plane at a point or lies within the plane. Computes the
    /// _time of impact_ of a `Ray` for a point intersection.
    ///
    /// Given a ray formed by an origin $P_0$ and a unit direction $\hat{u}$, the point of
    /// intersection with the plane is $P_0 + t\hat{u}$.
    fn intersection(&self, ray: &Ray<S>) -> Option<Self::Output> {
        let plane = self;
        ray.into_line()
            .intersection(plane)
            .and_then(|embedding| match embedding {
                LinePlane::TimeOfImpact(t) => {
                    if t.is_positive() {
                        Some(PlaneRay::TimeOfImpact(t))
                    }
                    else {
                        None
                    }
                }
                LinePlane::Line(_) => Some(PlaneRay::Ray(*ray)),
            })
    }
}
impl_symmetrical_intersection!(Plane, Ray);

#[cfg(test)]
mod tests {
    use decorum::real::UnaryRealFunction;
    use decorum::ExtendedReal;
    use nalgebra::{Point2, Point3};

    use crate::adjunct::Converged;
    use crate::query::{Aabb, Intersection, Line, LineLine, Plane, PlaneRay, Ray, Unit};
    use crate::space::{EuclideanSpace, Vector, VectorSpace};

    type X64 = ExtendedReal<f64>;

    type E2 = Point2<f64>;
    type E3 = Point3<f64>;

    #[test]
    fn aabb_aabb_intersection_e2() {
        let aabb1 = Aabb::<E2> {
            origin: EuclideanSpace::origin(),
            extent: Converged::converged(2.0),
        };
        let aabb2 = Aabb::<E2> {
            origin: Converged::converged(1.0),
            extent: Converged::converged(2.0),
        };
        assert_eq!(Some(aabb1), aabb1.intersection(&aabb1));
        assert_eq!(
            Some(Aabb::<E2> {
                origin: Converged::converged(1.0),
                extent: Converged::converged(1.0),
            }),
            aabb1.intersection(&aabb2),
        );
        let aabb2 = Aabb::<E2> {
            origin: Converged::converged(-3.0),
            extent: Converged::converged(2.0),
        };
        assert_eq!(None, aabb1.intersection(&aabb2));
    }

    #[test]
    fn aabb_point_intersection_e2() {
        let aabb = Aabb::<E2> {
            origin: EuclideanSpace::origin(),
            extent: Converged::converged(2.0),
        };
        let point = E2::converged(1.0);
        assert_eq!(
            Some(Vector::<E2>::converged(1.0)),
            aabb.intersection(&point),
        );
        let point = E2::converged(3.0);
        assert_eq!(None, aabb.intersection(&point));
    }

    #[test]
    fn aabb_ray_intersection_e2() {
        let aabb = Aabb::<E2> {
            origin: EuclideanSpace::origin(),
            extent: Converged::converged(1.0),
        };
        let ray = Ray::<E2> {
            origin: EuclideanSpace::from_xy(-1.0, 0.5),
            direction: Unit::x(),
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
            direction: Unit::x(),
        };
        assert_eq!(Some((1.0, 2.0)), ray.intersection(&aabb));
        assert_eq!(None, ray.reverse().intersection(&aabb));
    }

    // Ensure that certain values do not produce `NaN`s when querying the intersection of `Aabb`
    // and `Ray`.
    #[test]
    fn aabb_ray_intersection_nan() {
        let aabb = Aabb::<Point2<X64>> {
            origin: EuclideanSpace::origin(),
            extent: Converged::converged(X64::ONE),
        };
        let ray = Ray::<Point2<X64>> {
            origin: EuclideanSpace::origin(),
            direction: Unit::x(),
        };
        assert_eq!(Some((X64::ZERO, X64::ONE)), ray.intersection(&aabb));
    }

    #[test]
    fn line_line_intersection_e2() {
        let line = Line::<E2>::x();
        assert_eq!(
            Some(LineLine::Point(E2::origin())),
            line.intersection(&Line::y()),
        );
        assert_eq!(Some(LineLine::Line(line)), line.intersection(&Line::x()));

        let line1 = Line::<E2> {
            origin: E2::origin(),
            direction: Unit::try_from_inner(Converged::converged(1.0)).unwrap(),
        };
        let line2 = Line::<E2> {
            origin: E2::from_xy(2.0, 0.0),
            direction: Unit::try_from_inner(Vector::<E2>::from_xy(-1.0, 1.0)).unwrap(),
        };
        assert_eq!(
            Some(LineLine::Point(Converged::converged(1.0))),
            line1.intersection(&line2),
        );

        let line1 = Line::<E2>::x();
        let line2 = Line::<E2> {
            origin: E2::from_xy(0.0, 1.0),
            direction: Unit::x(),
        };
        assert_eq!(None, line1.intersection(&line2));
    }

    #[test]
    fn plane_ray_intersection_e3() {
        let plane = Plane::<E3> {
            origin: EuclideanSpace::from_xyz(0.0, 0.0, 1.0),
            normal: Unit::z(),
        };
        let ray = Ray::<E3> {
            origin: EuclideanSpace::origin(),
            direction: Unit::z(),
        };
        assert_eq!(Some(PlaneRay::TimeOfImpact(1.0)), ray.intersection(&plane));
        assert_eq!(None, ray.reverse().intersection(&plane));
    }
}
