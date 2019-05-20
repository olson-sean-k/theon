use decorum::{Infinite, Real};
use num::{Bounded, One, Zero};
use std::ops::Neg;

use crate::ops::{Reduce, ZipMap};
use crate::space::{Basis, EuclideanSpace, InnerSpace, Scalar, Vector};
use crate::Lattice;

pub trait Intersection<T> {
    type Output;

    fn intersection(&self, _: &T) -> Option<Self::Output>;
}

pub trait ReciprocalIntersection<T> {
    type Output;

    fn intersection(&self, _: &T) -> Option<Self::Output>;
}

impl<T, U> ReciprocalIntersection<U> for T
where
    U: Intersection<T>,
{
    type Output = U::Output;

    fn intersection(&self, other: &U) -> Option<Self::Output> {
        other.intersection(self)
    }
}

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

#[derive(Clone)]
pub struct Ray<S>
where
    S: EuclideanSpace,
{
    pub origin: S,
    pub direction: Unit<Vector<S>>,
}

impl<S> Ray<S>
where
    S: EuclideanSpace,
{
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
    Vector<S>: Reduce<Scalar<S>> + ZipMap<Scalar<S>, Output = Vector<S>>,
{
    type Output = (Scalar<S>, Scalar<S>);

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

impl<S> Neg for Ray<S>
where
    S: EuclideanSpace,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.reverse()
    }
}

#[derive(Clone)]
pub struct Aabb<S>
where
    S: EuclideanSpace,
{
    pub origin: S,
    pub extent: Vector<S>,
}

impl<S> Aabb<S>
where
    S: EuclideanSpace,
{
    pub fn from_points<I>(points: I) -> Self
    where
        S: ZipMap<Scalar<S>, Output = S>,
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

    pub fn upper_bound(&self) -> S
    where
        S: ZipMap<Scalar<S>, Output = S>,
    {
        self.origin
            .zip_map(self.endpoint(), |a, b| crate::partial_max(a, b))
    }

    pub fn lower_bound(&self) -> S
    where
        S: ZipMap<Scalar<S>, Output = S>,
    {
        self.origin
            .zip_map(self.endpoint(), |a, b| crate::partial_min(a, b))
    }

    /// Gets the Lebesgue measure (n-dimensional volume) of the `Aabb`.
    pub fn volume(&self) -> Scalar<S>
    where
        S: Reduce<Scalar<S>> + ZipMap<Scalar<S>, Output = S>,
    {
        self.origin
            .zip_map(self.endpoint(), |a, b| (a - b).abs())
            .reduce(One::one(), |product, a| product * a)
    }

    pub fn union(&self, aabb: &Self) -> Self
    where
        S: ZipMap<Scalar<S>, Output = S>,
    {
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
    fn aabb_ray_intersection_2() {
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
    fn aabb_ray_intersection_3() {
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
