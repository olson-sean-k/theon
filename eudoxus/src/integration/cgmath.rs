#![cfg(feature = "geometry-cgmath")]

use typenum::{U1, U2, U3, U4};

use crate::adjunct::{Adjunct, Converged, ExtendInto, Fold, Linear, Map, TruncateInto, ZipMap};
use crate::space::{
    AffineSpace, Basis, DualSpace, EuclideanSpace, FiniteDimensional, Homogeneous, InnerSpace,
    Scalar, VectorSpace,
};
use crate::{AsPosition, AsPositionMut};

#[doc(hidden)]
pub use cgmath::*;

impl<T> Adjunct for Vector1<T> {
    type Item = T;
}

impl<T> Adjunct for Vector2<T> {
    type Item = T;
}

impl<T> Adjunct for Vector3<T> {
    type Item = T;
}

impl<T> Adjunct for Vector4<T> {
    type Item = T;
}

impl<T> Basis for Vector1<T>
where
    T: BaseFloat,
{
    type Bases = [Self; 1];

    fn canonical_basis() -> Self::Bases {
        [Self::canonical_basis_component(0).unwrap()]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector1::unit_x()),
            _ => None,
        }
    }
}

impl<T> Basis for Vector2<T>
where
    T: BaseFloat,
{
    type Bases = [Self; 2];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector2::unit_x()),
            1 => Some(Vector2::unit_y()),
            _ => None,
        }
    }
}

impl<T> Basis for Vector3<T>
where
    T: BaseFloat,
{
    type Bases = [Self; 3];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
            Self::canonical_basis_component(2).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector3::unit_x()),
            1 => Some(Vector3::unit_y()),
            2 => Some(Vector3::unit_z()),
            _ => None,
        }
    }
}

impl<T> Basis for Vector4<T>
where
    T: BaseFloat,
{
    type Bases = [Self; 4];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
            Self::canonical_basis_component(2).unwrap(),
            Self::canonical_basis_component(3).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        match index {
            0 => Some(Vector4::unit_x()),
            1 => Some(Vector4::unit_y()),
            2 => Some(Vector4::unit_z()),
            3 => Some(Vector4::unit_w()),
            _ => None,
        }
    }
}

impl<T> Converged for Vector1<T> {
    fn converged(value: Self::Item) -> Self {
        Vector1::new(value)
    }
}

impl<T> Converged for Vector2<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Vector2::new(value, value)
    }
}

impl<T> Converged for Vector3<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Vector3::new(value, value, value)
    }
}

impl<T> Converged for Vector4<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Vector4::new(value, value, value, value)
    }
}

impl<T> DualSpace for Vector1<T>
where
    T: BaseFloat,
{
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl<T> DualSpace for Vector2<T>
where
    T: BaseFloat,
{
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl<T> DualSpace for Vector3<T>
where
    T: BaseFloat,
{
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl<T> DualSpace for Vector4<T>
where
    T: BaseFloat,
{
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl<T> ExtendInto<Vector2<T>> for Vector1<T> {
    fn extend(self, y: T) -> Vector2<T> {
        Vector2::new(self.x, y)
    }
}

impl<T> ExtendInto<Vector3<T>> for Vector2<T> {
    fn extend(self, z: T) -> Vector3<T> {
        Vector3::new(self.x, self.y, z)
    }
}

impl<T> ExtendInto<Vector4<T>> for Vector3<T> {
    fn extend(self, w: T) -> Vector4<T> {
        Vector4::new(self.x, self.y, self.z, w)
    }
}

impl<T> FiniteDimensional for Vector1<T> {
    type N = U1;
}

impl<T> FiniteDimensional for Vector2<T> {
    type N = U2;
}

impl<T> FiniteDimensional for Vector3<T> {
    type N = U3;
}

impl<T> FiniteDimensional for Vector4<T> {
    type N = U4;
}

impl<T> Fold for Vector1<T> {
    fn fold<U, F>(self, seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        f(seed, self.x)
    }
}

impl<T> Fold for Vector2<T> {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in [self.x, self.y] {
            seed = f(seed, a);
        }
        seed
    }
}

impl<T> Fold for Vector3<T> {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in [self.x, self.y, self.z] {
            seed = f(seed, a);
        }
        seed
    }
}

impl<T> Fold for Vector4<T> {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in [self.x, self.y, self.z, self.w] {
            seed = f(seed, a);
        }
        seed
    }
}

impl<T> Homogeneous for Vector1<T>
where
    T: BaseFloat,
{
    type ProjectiveSpace = Vector2<T>;
}

impl<T> Homogeneous for Vector2<T>
where
    T: BaseFloat,
{
    type ProjectiveSpace = Vector3<T>;
}

impl<T> Homogeneous for Vector3<T>
where
    T: BaseFloat,
{
    type ProjectiveSpace = Vector4<T>;
}

impl<T> InnerSpace for Vector1<T> where T: BaseFloat {}

impl<T> InnerSpace for Vector2<T> where T: BaseFloat {}

impl<T> InnerSpace for Vector3<T> where T: BaseFloat {}

impl<T> InnerSpace for Vector4<T> where T: BaseFloat {}

impl<T> Linear for Vector1<T> {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.x),
            _ => None,
        }
    }
}

impl<T> Linear for Vector2<T> {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.y),
            _ => None,
        }
    }
}

impl<T> Linear for Vector3<T> {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.y),
            2 => Some(&self.z),
            _ => None,
        }
    }
}

impl<T> Linear for Vector4<T> {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.y),
            2 => Some(&self.z),
            3 => Some(&self.w),
            _ => None,
        }
    }
}

impl<T, U> Map<U> for Vector1<T> {
    type Output = Vector1<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector1::new(f(self.x))
    }
}

impl<T, U> Map<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector2::new(f(self.x), f(self.y))
    }
}

impl<T, U> Map<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector3::new(f(self.x), f(self.y), f(self.z))
    }
}

impl<T, U> Map<U> for Vector4<T> {
    type Output = Vector4<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Vector4::new(f(self.x), f(self.y), f(self.z), f(self.w))
    }
}

impl<T> TruncateInto<Vector1<T>> for Vector2<T> {
    fn truncate(self) -> (Vector1<T>, T) {
        (Vector1::new(self.x), self.y)
    }
}

impl<T> TruncateInto<Vector2<T>> for Vector3<T> {
    fn truncate(self) -> (Vector2<T>, T) {
        (Vector2::new(self.x, self.y), self.z)
    }
}

impl<T> TruncateInto<Vector3<T>> for Vector4<T> {
    fn truncate(self) -> (Vector3<T>, T) {
        (Vector3::new(self.x, self.y, self.z), self.w)
    }
}

impl<T> VectorSpace for Vector1<T>
where
    T: BaseFloat,
{
    type Scalar = T;

    fn from_x(x: Self::Scalar) -> Self {
        Vector1::new(x)
    }

    fn into_x(self) -> Self::Scalar {
        self.x
    }

    fn x(&self) -> Self::Scalar {
        self.x
    }
}

impl<T> VectorSpace for Vector2<T>
where
    T: BaseFloat,
{
    type Scalar = T;

    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self {
        Vector2::new(x, y)
    }

    fn into_xy(self) -> (Self::Scalar, Self::Scalar) {
        (self.x, self.y)
    }

    fn x(&self) -> Self::Scalar {
        self.x
    }

    fn y(&self) -> Self::Scalar {
        self.y
    }
}

impl<T> VectorSpace for Vector3<T>
where
    T: BaseFloat,
{
    type Scalar = T;

    fn from_xyz(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self {
        Vector3::new(x, y, z)
    }

    fn into_xyz(self) -> (Self::Scalar, Self::Scalar, Self::Scalar) {
        (self.x, self.y, self.z)
    }

    fn x(&self) -> Self::Scalar {
        self.x
    }

    fn y(&self) -> Self::Scalar {
        self.y
    }

    fn z(&self) -> Self::Scalar {
        self.z
    }
}

impl<T> VectorSpace for Vector4<T>
where
    T: BaseFloat,
{
    type Scalar = T;

    fn from_xyzw(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar, w: Self::Scalar) -> Self {
        Vector4::new(x, y, z, w)
    }

    fn into_xyzw(self) -> (Self::Scalar, Self::Scalar, Self::Scalar, Self::Scalar) {
        (self.x, self.y, self.z, self.w)
    }

    fn x(&self) -> Self::Scalar {
        self.x
    }

    fn y(&self) -> Self::Scalar {
        self.y
    }

    fn z(&self) -> Self::Scalar {
        self.z
    }

    fn w(&self) -> Self::Scalar {
        self.w
    }
}

impl<T, U> ZipMap<U> for Vector1<T> {
    type Output = Vector1<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector1::new(f(self.x, other.x))
    }
}

impl<T, U> ZipMap<U> for Vector2<T> {
    type Output = Vector2<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector2::new(f(self.x, other.x), f(self.y, other.y))
    }
}

impl<T, U> ZipMap<U> for Vector3<T> {
    type Output = Vector3<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
    }
}

impl<T, U> ZipMap<U> for Vector4<T> {
    type Output = Vector4<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Vector4::new(
            f(self.x, other.x),
            f(self.y, other.y),
            f(self.z, other.z),
            f(self.w, other.w),
        )
    }
}

impl<T> Adjunct for Point1<T> {
    type Item = T;
}

impl<T> Adjunct for Point2<T> {
    type Item = T;
}

impl<T> Adjunct for Point3<T> {
    type Item = T;
}

impl<T> AffineSpace for Point1<T>
where
    T: BaseFloat,
{
    type Translation = Vector1<T>;
}

impl<T> AffineSpace for Point2<T>
where
    T: BaseFloat,
{
    type Translation = Vector2<T>;
}

impl<T> AffineSpace for Point3<T>
where
    T: BaseFloat,
{
    type Translation = Vector3<T>;
}

impl<T> AsPosition for Point1<T>
where
    T: BaseFloat,
{
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl<T> AsPosition for Point2<T>
where
    T: BaseFloat,
{
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl<T> AsPosition for Point3<T>
where
    T: BaseFloat,
{
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl<T> AsPositionMut for Point1<T>
where
    T: BaseFloat,
{
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl<T> AsPositionMut for Point2<T>
where
    T: BaseFloat,
{
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl<T> AsPositionMut for Point3<T>
where
    T: BaseFloat,
{
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl<T> Converged for Point1<T> {
    fn converged(value: Self::Item) -> Self {
        Point1::new(value)
    }
}

impl<T> Converged for Point2<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Point2::new(value, value)
    }
}

impl<T> Converged for Point3<T>
where
    T: Copy,
{
    fn converged(value: Self::Item) -> Self {
        Point3::new(value, value, value)
    }
}

impl<T> ExtendInto<Point2<T>> for Point1<T> {
    fn extend(self, y: T) -> Point2<T> {
        let Point1 { x } = self;
        Point2::new(x, y)
    }
}

impl<T> ExtendInto<Point3<T>> for Point2<T> {
    fn extend(self, z: T) -> Point3<T> {
        let Point2 { x, y } = self;
        Point3::new(x, y, z)
    }
}

impl<T> EuclideanSpace for Point1<T>
where
    T: BaseFloat,
{
    type CoordinateSpace = Vector1<T>;

    fn x(&self) -> Scalar<Self> {
        self.x
    }
}

impl<T> EuclideanSpace for Point2<T>
where
    T: BaseFloat,
{
    type CoordinateSpace = Vector2<T>;

    fn x(&self) -> Scalar<Self> {
        self.x
    }

    fn y(&self) -> Scalar<Self> {
        self.y
    }
}

impl<T> EuclideanSpace for Point3<T>
where
    T: BaseFloat,
{
    type CoordinateSpace = Vector3<T>;

    fn x(&self) -> Scalar<Self> {
        self.x
    }

    fn y(&self) -> Scalar<Self> {
        self.y
    }

    fn z(&self) -> Scalar<Self> {
        self.z
    }
}

impl<T> FiniteDimensional for Point1<T> {
    type N = U1;
}

impl<T> FiniteDimensional for Point2<T> {
    type N = U2;
}

impl<T> FiniteDimensional for Point3<T> {
    type N = U3;
}

impl<T> Fold for Point1<T> {
    fn fold<U, F>(self, seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        f(seed, self.x)
    }
}

impl<T> Fold for Point2<T> {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in [self.x, self.y] {
            seed = f(seed, a);
        }
        seed
    }
}

impl<T> Fold for Point3<T> {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for a in [self.x, self.y, self.z] {
            seed = f(seed, a);
        }
        seed
    }
}

impl<T> Linear for Point1<T> {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.x),
            _ => None,
        }
    }
}

impl<T> Linear for Point2<T> {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.y),
            _ => None,
        }
    }
}

impl<T> Linear for Point3<T> {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.y),
            2 => Some(&self.z),
            _ => None,
        }
    }
}

impl<T, U> Map<U> for Point1<T> {
    type Output = Point1<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Point1::new(f(self.x))
    }
}

impl<T, U> Map<U> for Point2<T> {
    type Output = Point2<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Point2::new(f(self.x), f(self.y))
    }
}

impl<T, U> Map<U> for Point3<T> {
    type Output = Point3<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        Point3::new(f(self.x), f(self.y), f(self.z))
    }
}

impl<T> TruncateInto<Point1<T>> for Point2<T> {
    fn truncate(self) -> (Point1<T>, T) {
        let Point2 { x, y } = self;
        (Point1::new(x), y)
    }
}

impl<T> TruncateInto<Point2<T>> for Point3<T> {
    fn truncate(self) -> (Point2<T>, T) {
        let Point3 { x, y, z } = self;
        (Point2::new(x, y), z)
    }
}

impl<T, U> ZipMap<U> for Point1<T> {
    type Output = Point1<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Point1::new(f(self.x, other.x))
    }
}

impl<T, U> ZipMap<U> for Point2<T> {
    type Output = Point2<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Point2::new(f(self.x, other.x), f(self.y, other.y))
    }
}

impl<T, U> ZipMap<U> for Point3<T> {
    type Output = Point3<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        Point3::new(f(self.x, other.x), f(self.y, other.y), f(self.z, other.z))
    }
}
