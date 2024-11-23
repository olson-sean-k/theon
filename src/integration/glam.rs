#![cfg(feature = "glam")]

use decorum::R64;
use glam::{Vec2, Vec3, Vec3A, Vec4};
use typenum::consts::{U2, U3, U4};

use crate::adjunct::{Adjunct, Converged, Extend, Fold, Map, Truncate, ZipMap};
use crate::ops::{Cross, Dot, Interpolate};
use crate::space::{
    AffineSpace, Basis, DualSpace, EuclideanSpace, FiniteDimensional, Homogeneous, InnerSpace,
    VectorSpace,
};
use crate::{AsPosition, AsPositionMut};

impl Adjunct for Vec2 {
    type Item = f32;
}

impl Adjunct for Vec3 {
    type Item = f32;
}

impl Adjunct for Vec3A {
    type Item = f32;
}

impl Adjunct for Vec4 {
    type Item = f32;
}

impl AffineSpace for Vec2 {
    type Translation = Self;
}

impl AffineSpace for Vec3 {
    type Translation = Self;
}

impl AffineSpace for Vec3A {
    type Translation = Self;
}

impl AsPosition for Vec2 {
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl AsPosition for Vec3 {
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl AsPosition for Vec3A {
    type Position = Self;

    fn as_position(&self) -> &Self::Position {
        self
    }
}

impl AsPositionMut for Vec2 {
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl AsPositionMut for Vec3 {
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl AsPositionMut for Vec3A {
    fn as_position_mut(&mut self) -> &mut Self::Position {
        self
    }
}

impl Basis for Vec2 {
    type Bases = [Self; 2];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        Self::AXES.get(index).copied()
    }
}

impl Basis for Vec3 {
    type Bases = [Self; 3];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
            Self::canonical_basis_component(2).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        Self::AXES.get(index).copied()
    }
}

impl Basis for Vec3A {
    type Bases = [Self; 3];

    fn canonical_basis() -> Self::Bases {
        [
            Self::canonical_basis_component(0).unwrap(),
            Self::canonical_basis_component(1).unwrap(),
            Self::canonical_basis_component(2).unwrap(),
        ]
    }

    fn canonical_basis_component(index: usize) -> Option<Self> {
        Self::AXES.get(index).copied()
    }
}

impl Basis for Vec4 {
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
        Self::AXES.get(index).copied()
    }
}

impl Converged for Vec2 {
    fn converged(value: Self::Item) -> Self {
        Self::splat(value)
    }
}

impl Converged for Vec3 {
    fn converged(value: Self::Item) -> Self {
        Self::splat(value)
    }
}

impl Converged for Vec3A {
    fn converged(value: Self::Item) -> Self {
        Self::splat(value)
    }
}

impl Converged for Vec4 {
    fn converged(value: Self::Item) -> Self {
        Self::splat(value)
    }
}

impl Cross for Vec3 {
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Self::cross(self, other)
    }
}

impl Cross for Vec3A {
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Self::cross(self, other)
    }
}

impl Dot for Vec2 {
    type Output = f32;

    fn dot(self, other: Self) -> Self::Output {
        Self::dot(self, other)
    }
}

impl Dot for Vec3 {
    type Output = f32;

    fn dot(self, other: Self) -> Self::Output {
        Self::dot(self, other)
    }
}

impl Dot for Vec3A {
    type Output = f32;

    fn dot(self, other: Self) -> Self::Output {
        Self::dot(self, other)
    }
}

impl Dot for Vec4 {
    type Output = f32;

    fn dot(self, other: Self) -> Self::Output {
        Self::dot(self, other)
    }
}

impl DualSpace for Vec2 {
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl DualSpace for Vec3 {
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl DualSpace for Vec3A {
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl DualSpace for Vec4 {
    type Dual = Self;

    fn transpose(self) -> Self::Dual {
        self
    }
}

impl Extend<Vec3> for Vec2 {
    fn extend(self, w: Self::Item) -> Vec3 {
        self.extend(w)
    }
}

impl Extend<Vec3A> for Vec2 {
    fn extend(self, w: Self::Item) -> Vec3A {
        self.extend(w).into()
    }
}

impl Extend<Vec4> for Vec3 {
    fn extend(self, w: Self::Item) -> Vec4 {
        self.extend(w)
    }
}

impl Extend<Vec4> for Vec3A {
    fn extend(self, w: Self::Item) -> Vec4 {
        self.extend(w)
    }
}

impl EuclideanSpace for Vec2 {
    type CoordinateSpace = Self;

    fn origin() -> Self {
        Self::zero()
    }
}

impl EuclideanSpace for Vec3 {
    type CoordinateSpace = Self;

    fn origin() -> Self {
        Self::zero()
    }
}

impl EuclideanSpace for Vec3A {
    type CoordinateSpace = Self;

    fn origin() -> Self {
        Self::zero()
    }
}

impl FiniteDimensional for Vec2 {
    type N = U2;
}

impl FiniteDimensional for Vec3 {
    type N = U3;
}

impl FiniteDimensional for Vec3A {
    type N = U3;
}

impl FiniteDimensional for Vec4 {
    type N = U4;
}

impl Fold for Vec2 {
    fn fold<T, F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T,
    {
        <[f32; 2]>::from(self).iter().cloned().fold(seed, f)
    }
}

impl Fold for Vec3 {
    fn fold<T, F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T,
    {
        <[f32; 3]>::from(self).iter().cloned().fold(seed, f)
    }
}

impl Fold for Vec3A {
    fn fold<T, F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T,
    {
        <[f32; 3]>::from(self).iter().cloned().fold(seed, f)
    }
}

impl Fold for Vec4 {
    fn fold<T, F>(self, seed: T, f: F) -> T
    where
        F: FnMut(T, Self::Item) -> T,
    {
        <[f32; 4]>::from(self).iter().cloned().fold(seed, f)
    }
}

impl Homogeneous for Vec2 {
    type ProjectiveSpace = Vec3;
}

impl Homogeneous for Vec3 {
    type ProjectiveSpace = Vec4;
}

impl Homogeneous for Vec3A {
    type ProjectiveSpace = Vec4;
}

impl InnerSpace for Vec2 {}

impl InnerSpace for Vec3 {}

impl InnerSpace for Vec3A {}

impl InnerSpace for Vec4 {}

impl Interpolate for Vec2 {
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Self::lerp(self, other, f64::from(f) as f32)
    }
}

impl Interpolate for Vec3 {
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Self::lerp(self, other, f64::from(f) as f32)
    }
}

impl Interpolate for Vec3A {
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Self::lerp(self, other, f64::from(f) as f32)
    }
}

impl Interpolate for Vec4 {
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Self::lerp(self, other, f64::from(f) as f32)
    }
}

impl Map<f32> for Vec2 {
    type Output = Self;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> f32,
    {
        let [x, y]: [f32; 2] = self.into();
        [f(x), f(y)].into()
    }
}

impl Map<f32> for Vec3 {
    type Output = Self;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> f32,
    {
        let [x, y, z]: [f32; 3] = self.into();
        [f(x), f(y), f(z)].into()
    }
}

impl Map<f32> for Vec3A {
    type Output = Self;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> f32,
    {
        let [x, y, z]: [f32; 3] = self.into();
        [f(x), f(y), f(z)].into()
    }
}

impl Map<f32> for Vec4 {
    type Output = Self;

    #[allow(clippy::many_single_char_names)]
    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> f32,
    {
        let [x, y, z, w]: [f32; 4] = self.into();
        [f(x), f(y), f(z), f(w)].into()
    }
}

impl Truncate<Vec2> for Vec3 {
    fn truncate(self) -> (Vec2, Self::Item) {
        let z = self.z();
        (self.truncate(), z)
    }
}

impl Truncate<Vec2> for Vec3A {
    fn truncate(self) -> (Vec2, Self::Item) {
        let z = self.z();
        (self.truncate(), z)
    }
}

impl Truncate<Vec3> for Vec4 {
    fn truncate(self) -> (Vec3, Self::Item) {
        let z = self.z();
        (self.truncate(), z)
    }
}

impl Truncate<Vec3A> for Vec4 {
    fn truncate(self) -> (Vec3A, Self::Item) {
        let z = self.z();
        (self.truncate().into(), z)
    }
}

impl VectorSpace for Vec2 {
    type Scalar = f32;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar> {
        match index {
            0 => Some(self.x()),
            1 => Some(self.y()),
            _ => None,
        }
    }

    fn into_xy(self) -> (Self::Scalar, Self::Scalar) {
        self.into()
    }

    fn zero() -> Self {
        Self::ZERO
    }
}

impl VectorSpace for Vec3 {
    type Scalar = f32;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar> {
        match index {
            0 => Some(self.x()),
            1 => Some(self.y()),
            2 => Some(self.z()),
            _ => None,
        }
    }

    fn into_xyz(self) -> (Self::Scalar, Self::Scalar, Self::Scalar) {
        self.into()
    }

    fn zero() -> Self {
        Self::ZERO
    }
}

impl VectorSpace for Vec3A {
    type Scalar = f32;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar> {
        match index {
            0 => Some(self.x()),
            1 => Some(self.y()),
            2 => Some(self.z()),
            _ => None,
        }
    }

    fn into_xyz(self) -> (Self::Scalar, Self::Scalar, Self::Scalar) {
        self.into()
    }

    fn zero() -> Self {
        Self::ZERO
    }
}

impl VectorSpace for Vec4 {
    type Scalar = f32;

    fn scalar_component(&self, index: usize) -> Option<Self::Scalar> {
        let [x, y, z, w] = self.to_array();
        match index {
            0 => Some(x),
            1 => Some(y),
            2 => Some(z),
            3 => Some(w),
            _ => None,
        }
    }

    fn zero() -> Self {
        Self::ZERO
    }
}

impl ZipMap<f32> for Vec2 {
    type Output = Self;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> f32,
    {
        let [x1, y1]: [f32; 2] = self.into();
        let [x2, y2]: [f32; 2] = other.into();
        From::from([f(x1, x2), f(y1, y2)])
    }
}

impl ZipMap<f32> for Vec3 {
    type Output = Self;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> f32,
    {
        let [x1, y1, z1]: [f32; 3] = self.into();
        let [x2, y2, z2]: [f32; 3] = other.into();
        From::from([f(x1, x2), f(y1, y2), f(z1, z2)])
    }
}

impl ZipMap<f32> for Vec3A {
    type Output = Self;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> f32,
    {
        let [x1, y1, z1]: [f32; 3] = self.into();
        let [x2, y2, z2]: [f32; 3] = other.into();
        From::from([f(x1, x2), f(y1, y2), f(z1, z2)])
    }

    fn per_item_sum(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = f32>,
    {
        // The `Add` implementation uses `_mm_add_ps`.
        self + other
    }

    fn per_item_product(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = f32>,
    {
        // The `Mul` implementation uses `_mm_mul_ps`.
        self * other
    }
}

impl ZipMap<f32> for Vec4 {
    type Output = Self;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> f32,
    {
        let [x1, y1, z1, w1]: [f32; 4] = self.into();
        let [x2, y2, z2, w2]: [f32; 4] = other.into();
        From::from([f(x1, x2), f(y1, y2), f(z1, z2), f(w1, w2)])
    }

    fn per_item_sum(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = f32>,
    {
        // The `Add` implementation uses `_mm_add_ps`.
        self + other
    }

    fn per_item_product(self, other: Self) -> Self::Output
    where
        Self: Adjunct<Item = f32>,
    {
        // The `Mul` implementation uses `_mm_mul_ps`.
        self * other
    }
}
