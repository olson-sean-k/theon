use arrayvec::{Array, ArrayVec};
use itertools::izip;
use num::Integer;
use std::ops::{Index, IndexMut};
use std::slice;
use typenum::U3;

use crate::array::ArrayScalar;
use crate::ops::{Fold, Map, Zip, ZipMap};
use crate::query::{Intersection, Line, Plane};
use crate::space::{EuclideanSpace, FiniteDimensional, Scalar, Vector};
use crate::{Composite, Converged, FromItems, IntoItems};

pub trait Arity {
    const ARITY: usize = 1;
}

// TODO: It should be possible to implement this for all `NGon`s, but that
//       implementation would likely be inefficient.
pub trait Rotate {
    fn rotate(self, n: isize) -> Self;
}

pub trait Flatten: Sized {
    fn flatten(self) -> Option<Self>;
}

/// Statically sized $n$-gon.
///
/// `NGon` represents a polygonal structure as an array. Each array element
/// represents vertex data in order, with neighboring elements being connected
/// by an implicit undirected edge. For example, an `NGon` with three vertices
/// (`NGon<[T; 3]>`) would represent a triangle (trigon). Generally these
/// elements are labeled $A$, $B$, $C$, etc.
///
/// **`NGon`s with less than three vertices are a degenerate case.** An `NGon`
/// with two vertices (`NGon<[T; 2]>`) is considered a _monogon_ despite common
/// definitions specifying a single vertex. Such an `NGon` is not considered a
/// _digon_, as it represents a single undirected edge rather than two distinct
/// (but collapsed) edges. Single-vertex `NGon`s are unsupported. See the
/// `Edge` type definition.
///
/// Monogons and digons are not generally considered polygons, and `NGon` does
/// not implement the `Polygonal` trait in these cases.
///
/// See the `Edge`, `Trigon`, and `Tetragon` type aliases.
#[derive(Clone, Copy, Debug)]
pub struct NGon<A>(pub A)
where
    A: Array;

impl<A> NGon<A>
where
    A: Array,
{
    pub fn into_array(self) -> A {
        self.0
    }

    fn into_array_vec(self) -> ArrayVec<A> {
        ArrayVec::from(self.into_array())
    }

    #[cfg(all(feature = "array", target_os = "linux"))]
    pub fn into_plane(self) -> Option<Plane<A::Item>>
    where
        Self: Composite<Item = A::Item>,
        A::Item: EuclideanSpace + FiniteDimensional<N = U3>,
        Scalar<A::Item>: ArrayScalar,
        Vector<A::Item>: FromItems + IntoItems,
    {
        Plane::from_points(self.into_array_vec())
    }
}

/// Gets a slice over the data in an `NGon`.
///
/// Slicing an `NGon` can be used to iterate over references to its data:
///
/// ```rust
/// use theon::ngon::Trigon;
/// use theon::Converged;
///
/// let trigon = Trigon::converged(0u32);
/// for vertex in trigon.as_ref() {
///     // ...
/// }
/// ```
impl<A> AsRef<[<A as Array>::Item]> for NGon<A>
where
    A: Array,
{
    fn as_ref(&self) -> &[A::Item] {
        unsafe { slice::from_raw_parts(self.0.as_ptr(), A::capacity()) }
    }
}

/// Gets a mutable slice over the data in an `NGon`.
///
/// Slicing an `NGon` can be used to iterate over references to its data:
///
/// ```rust
/// use theon::ngon::Tetragon;
/// use theon::Converged;
///
/// let mut tetragon = Tetragon::converged(1u32);
/// for mut vertex in tetragon.as_mut() {
///     *vertex = 0;
/// }
/// ```
impl<A> AsMut<[<A as Array>::Item]> for NGon<A>
where
    A: Array,
{
    fn as_mut(&mut self) -> &mut [A::Item] {
        unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr(), A::capacity()) }
    }
}

impl<A> Composite for NGon<A>
where
    A: Array,
{
    type Item = A::Item;
}

impl<A, U> Fold<U> for NGon<A>
where
    Self: IntoItems,
    A: Array,
{
    fn fold<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for vertex in self.into_items() {
            seed = f(seed, vertex);
        }
        seed
    }
}

impl<A> From<A> for NGon<A>
where
    A: Array,
{
    fn from(array: A) -> Self {
        NGon(array)
    }
}

impl<A> FromItems for NGon<A>
where
    A: Array,
{
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        items
            .into_iter()
            .collect::<ArrayVec<A>>()
            .into_inner()
            .ok()
            .map(|array| NGon(array))
    }
}

impl<A> Index<usize> for NGon<A>
where
    A: Array + AsRef<[<A as Array>::Item]>,
{
    type Output = A::Item;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.as_ref().index(index)
    }
}

impl<A> IndexMut<usize> for NGon<A>
where
    A: Array + AsRef<[<A as Array>::Item]> + AsMut<[<A as Array>::Item]>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.as_mut().index_mut(index)
    }
}

impl<A> IntoItems for NGon<A>
where
    A: Array,
{
    type Output = ArrayVec<A>;

    fn into_items(self) -> Self::Output {
        self.into_array_vec()
    }
}

impl<A> IntoIterator for NGon<A>
where
    A: Array,
{
    type Item = <A as Array>::Item;
    type IntoIter = <<Self as IntoItems>::Output as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_items().into_iter()
    }
}

macro_rules! impl_zip_ngon {
    (composite => $c:ident, length => $n:expr) => (
        impl_zip_ngon!(composite => $c, length => $n, items => (A, B));
        impl_zip_ngon!(composite => $c, length => $n, items => (A, B, C));
        impl_zip_ngon!(composite => $c, length => $n, items => (A, B, C, D));
        impl_zip_ngon!(composite => $c, length => $n, items => (A, B, C, D, E));
        impl_zip_ngon!(composite => $c, length => $n, items => (A, B, C, D, E, F));
    );
    (composite => $c:ident, length => $n:expr, items => ($($i:ident),*)) => (
        #[allow(non_snake_case)]
        impl<$($i),*> Zip for ($($c<[$i; $n]>),*) {
            type Output = $c<[($($i),*); $n]>;

            fn zip(self) -> Self::Output {
                let ($($i,)*) = self;
                FromItems::from_items(izip!($($i.into_items()),*)).unwrap()
            }
        }
    );
}

macro_rules! impl_ngon {
    (length => $n:expr) => (
        impl<T> Converged for NGon<[T; $n]>
        where
            T: Copy,
        {
            fn converged(item: T) -> Self {
                NGon([item; $n])
            }
        }

        impl<T, U> Map<U> for NGon<[T; $n]> {
            type Output = NGon<[U; $n]>;

            fn map<F>(self, f: F) -> Self::Output
            where
                F: FnMut(Self::Item) -> U,
            {
                FromItems::from_items(self.into_iter().map(f)).unwrap()
            }
        }

        impl_zip_ngon!(composite => NGon, length => $n);

        impl<T, U> ZipMap<U> for NGon<[T; $n]> {
            type Output = NGon<[U; $n]>;

            fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
            where
                F: FnMut(Self::Item, Self::Item) -> U,
            {
                FromItems::from_items(self.into_iter().zip(other).map(|(a, b)| f(a, b))).unwrap()
            }
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        $(impl_ngon!(length => $n);)*
    );
}
impl_ngon!(lengths => 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

macro_rules! impl_arity_ngon {
    (length => $n:expr) => (
        impl<T> Arity for NGon<[T; $n]> {
            const ARITY: usize = $n;
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        impl<T> Arity for NGon<[T; 2]> {
            const ARITY: usize = 1;
        }

        $(impl_arity_ngon!(length => $n);)*
    );
}
impl_arity_ngon!(lengths => 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

macro_rules! impl_flatten_ngon {
    (length => $n:expr) => (
        impl<S> Flatten for NGon<[S; $n]>
        where
            S: EuclideanSpace + FiniteDimensional<N = U3>,
            Scalar<S>: ArrayScalar,
            Vector<S>: FromItems + IntoItems,
        {
            fn flatten(self) -> Option<Self> {
                let mut flat = self;
                self.into_plane().map(move |plane| {
                    for position in flat.as_mut() {
                        let line = Line::<S> {
                            origin: *position,
                            direction: plane.normal,
                        };
                        if let Some(distance) = plane.intersection(&line) {
                            let translation = *line.direction.get() * distance;
                            *position = *position + translation;
                        }
                    }
                    flat
                })
            }
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        impl<S> Flatten for NGon<[S; 2]>
        where
            S: EuclideanSpace + FiniteDimensional<N = U3>,
        {
            fn flatten(self) -> Option<Self> {
                Some(self)
            }
        }

        impl<S> Flatten for NGon<[S; 3]>
        where
            S: EuclideanSpace + FiniteDimensional<N = U3>,
        {
            fn flatten(self) -> Option<Self> {
                Some(self)
            }
        }

        $(impl_flatten_ngon!(length => $n);)*
    );
}
#[cfg(all(feature = "array", target_os = "linux"))]
impl_flatten_ngon!(lengths => 4, 5, 6, 7, 8, 9, 10, 11, 12);

pub type Edge<T> = NGon<[T; 2]>;

impl<T> Edge<T> {
    pub fn new(a: T, b: T) -> Self {
        NGon([a, b])
    }
}

impl<T> Rotate for Edge<T> {
    fn rotate(self, n: isize) -> Self {
        if n % 2 != 0 {
            let [a, b] = self.into_array();
            Edge::new(b, a)
        }
        else {
            self
        }
    }
}

pub type Trigon<T> = NGon<[T; 3]>;

impl<T> Trigon<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        NGon([a, b, c])
    }
}

impl<T> Rotate for Trigon<T> {
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let [a, b, c] = self.into_array();
            Trigon::new(b, c, a)
        }
        else if n == 2 {
            let [a, b, c] = self.into_array();
            Trigon::new(c, a, b)
        }
        else {
            self
        }
    }
}

pub type Tetragon<T> = NGon<[T; 4]>;

impl<T> Tetragon<T> {
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        NGon([a, b, c, d])
    }
}

impl<T> Rotate for Tetragon<T> {
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let [a, b, c, d] = self.into_array();
            Tetragon::new(b, c, d, a)
        }
        else if n == 2 {
            let [a, b, c, d] = self.into_array();
            Tetragon::new(c, d, a, b)
        }
        else if n == 3 {
            let [a, b, c, d] = self.into_array();
            Tetragon::new(d, a, b, c)
        }
        else {
            self
        }
    }
}

fn umod<T>(n: T, m: T) -> T
where
    T: Copy + Integer,
{
    ((n % m) + m) % m
}
