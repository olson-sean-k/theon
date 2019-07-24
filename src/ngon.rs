use arrayvec::{Array, ArrayVec};
use itertools::izip;
use num::Integer;
use smallvec::SmallVec;
use std::ops::{Index, IndexMut};
use std::slice;
use typenum::type_operators::Cmp;
use typenum::{Greater, U2, U3};

use crate::ops::{Cross, Fold, Map, Zip, ZipMap};
use crate::query::{Intersection, Line, Plane, Unit};
use crate::space::{
    EmbeddingSpace, EuclideanSpace, FiniteDimensional, Scalar, Vector, VectorSpace,
};
use crate::{AsPosition, Composite, Converged, FromItems, IntoItems, Position};

pub trait Topological: Composite<Item = <Self as Topological>::Vertex> {
    type Vertex;

    fn arity(&self) -> usize;
}

pub trait NGon:
    AsMut<[<Self as Topological>::Vertex]>
    + AsRef<[<Self as Topological>::Vertex]>
    + IntoIterator<Item = <Self as Topological>::Vertex>
    + Sized
    + Topological
{
    /// Embeds an $n$-gon from $\Reals^2$ into $\Reals^3$.
    ///
    /// The scalar for the additional basis is normalized to the given value.
    ///
    /// # Examples
    ///
    /// Embedding a triangle into the $xy$-plane at $z=1$:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate theon;
    /// #
    /// use nalgebra::Point2;
    /// use theon::ngon::{NGon, Trigon};
    /// use theon::space::EuclideanSpace;
    ///
    /// type E2 = Point2<f64>;
    ///
    /// let trigon = Trigon::embed_into_xy(
    ///     Trigon::from([
    ///         E2::from_xy(-1.0, 0.0),
    ///         E2::from_xy(0.0, 1.0),
    ///         E2::from_xy(1.0, 0.0)
    ///     ]),
    ///     1.0,
    /// );
    /// ```
    fn embed_into_xy<P>(ngon: P, z: Scalar<Self::Vertex>) -> Self
    where
        Self::Vertex: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + NGon,
        P::Vertex: EmbeddingSpace<Embedding = Self::Vertex> + FiniteDimensional<N = U2>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Self::Vertex>>,
    {
        Self::embed_into_xy_with(ngon, z, |position| position)
    }

    fn embed_into_xy_with<P, F>(ngon: P, z: Scalar<Position<Self::Vertex>>, mut f: F) -> Self
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + NGon,
        P::Vertex: EmbeddingSpace<Embedding = Position<Self::Vertex>> + FiniteDimensional<N = U2>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Position<Self::Vertex>>>,
        F: FnMut(Position<Self::Vertex>) -> Self::Vertex,
    {
        ngon.map(move |position| f(position.embed(z)))
    }

    /// Embeds an $n$-gon from $\Reals^2$ into $\Reals^3$.
    ///
    /// The $n$-gon is rotated into the given plane about the origin.
    ///
    /// # Examples
    ///
    /// Embedding a triangle into the $xy$-plane at $z=0$:
    ///
    /// ```rust,no_run
    /// # extern crate nalgebra;
    /// # extern crate theon;
    /// #
    /// use nalgebra::{Point2, Point3};
    /// use theon::ngon::{NGon, Trigon};
    /// use theon::query::{Plane, Unit};
    /// use theon::space::{Basis, EuclideanSpace};
    ///
    /// type E2 = Point2<f64>;
    /// type E3 = Point3<f64>;
    ///
    /// let trigon = Trigon::embed_into_plane(
    ///     Trigon::from([
    ///         E2::from_xy(-1.0, 0.0),
    ///         E2::from_xy(0.0, 1.0),
    ///         E2::from_xy(1.0, 0.0)
    ///     ]),
    ///     Plane::<E3> {
    ///         origin: EuclideanSpace::origin(),
    ///         normal: Unit::try_from_inner(Basis::z()).expect("non-zero"),
    ///     },
    /// );
    /// ```
    fn embed_into_plane<P>(ngon: P, plane: Plane<Self::Vertex>) -> Self
    where
        Self::Vertex: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + NGon,
        P::Vertex: EmbeddingSpace<Embedding = Self::Vertex> + FiniteDimensional<N = U2>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Self::Vertex>>,
    {
        Self::embed_into_plane_with(ngon, plane, |position| position)
    }

    fn embed_into_plane_with<P, F>(ngon: P, plane: Plane<Position<Self::Vertex>>, mut f: F) -> Self
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + NGon,
        P::Vertex: EmbeddingSpace<Embedding = Position<Self::Vertex>> + FiniteDimensional<N = U2>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Position<Self::Vertex>>>,
        F: FnMut(Position<Self::Vertex>) -> Self::Vertex,
    {
        // TODO: Rotate the embedded n-gon into the plane about the origin.
        let _ = (ngon, plane, f);
        unimplemented!()
    }

    /// Projects an $n$-gon into a plane.
    ///
    /// The positions in each vertex of the $n$-gon are translated along the
    /// normal of the plane.
    fn project_into_plane(mut self, plane: Plane<Position<Self::Vertex>>) -> Self
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional,
        <Position<Self::Vertex> as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
    {
        for vertex in self.as_mut() {
            let line = Line::<Position<Self::Vertex>> {
                origin: *vertex.as_position(),
                direction: plane.normal,
            };
            if let Some(distance) = plane.intersection(&line) {
                let translation = *line.direction.get() * distance;
                vertex.transform(|position| *position + translation);
            }
        }
        self
    }
}

pub trait Arity {
    const ARITY: usize = 1;
}

pub trait Rotate {
    fn rotate(self, n: isize) -> Self;
}

#[derive(Clone, Debug)]
pub struct DynamicNGon<T>(SmallVec<[T; 4]>);

/// Statically sized $n$-gon.
///
/// `StaticNGon` represents a polygonal structure as an array. Each array
/// element represents vertex data in order with neighboring elements being
/// connected by an implicit undirected edge. For example, an `StaticNGon` with
/// three vertices (`StaticNGon<[T; 3]>`) would represent a triangle (trigon).
/// Generally these elements are labeled $A$, $B$, $C$, etc.
///
/// **`StaticNGon`s with less than three vertices are a degenerate case.** An
/// `StaticNGon` with two vertices (`StaticNGon<[T; 2]>`) is considered a
/// _monogon_ despite common definitions specifying a single vertex. Such an
/// `StaticNGon` is not considered a _digon_, as it represents a single
/// undirected edge rather than two distinct (but collapsed) edges.
/// Single-vertex `StaticNGon`s are unsupported. See the `Edge` type
/// definition.
///
/// Polygons are defined in $\Reals^2$, but `StaticNGon` supports arbitrary
/// vertex data. This includes positional data in Euclidean spaces of arbitrary
/// dimension. As such, `StaticNGon` does not represent a "pure polygon", but
/// instead a superset defined by its topology. `StaticNGon`s in $\Reals^3$ are
/// useful for representing polygons embedded into three-dimensional space, but
/// **there are no restrictions on the geometry of vertices**.
#[derive(Clone, Copy, Debug)]
pub struct StaticNGon<A>(pub A)
where
    A: Array;

impl<A> StaticNGon<A>
where
    A: Array,
{
    pub fn into_array(self) -> A {
        self.0
    }

    fn into_array_vec(self) -> ArrayVec<A> {
        ArrayVec::from(self.into_array())
    }
}

/// Gets a slice over the data in an `StaticNGon`.
///
/// Slicing an `StaticNGon` can be used to iterate over references to its data:
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
impl<A> AsRef<[<A as Array>::Item]> for StaticNGon<A>
where
    A: Array,
{
    fn as_ref(&self) -> &[A::Item] {
        unsafe { slice::from_raw_parts(self.0.as_ptr(), A::capacity()) }
    }
}

/// Gets a mutable slice over the data in an `StaticNGon`.
///
/// Slicing an `StaticNGon` can be used to iterate over references to its data:
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
impl<A> AsMut<[<A as Array>::Item]> for StaticNGon<A>
where
    A: Array,
{
    fn as_mut(&mut self) -> &mut [A::Item] {
        unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr(), A::capacity()) }
    }
}

impl<A> Composite for StaticNGon<A>
where
    A: Array,
{
    type Item = A::Item;
}

impl<A, U> Fold<U> for StaticNGon<A>
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

impl<A> From<A> for StaticNGon<A>
where
    A: Array,
{
    fn from(array: A) -> Self {
        StaticNGon(array)
    }
}

impl<A> FromItems for StaticNGon<A>
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
            .map(|array| StaticNGon(array))
    }
}

impl<A> Index<usize> for StaticNGon<A>
where
    A: Array + AsRef<[<A as Array>::Item]>,
{
    type Output = A::Item;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.as_ref().index(index)
    }
}

impl<A> IndexMut<usize> for StaticNGon<A>
where
    A: Array + AsRef<[<A as Array>::Item]> + AsMut<[<A as Array>::Item]>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.as_mut().index_mut(index)
    }
}

impl<A> IntoItems for StaticNGon<A>
where
    A: Array,
{
    type Output = ArrayVec<A>;

    fn into_items(self) -> Self::Output {
        self.into_array_vec()
    }
}

impl<A> IntoIterator for StaticNGon<A>
where
    A: Array,
{
    type Item = <A as Array>::Item;
    type IntoIter = <<Self as IntoItems>::Output as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_items().into_iter()
    }
}

impl<A> NGon for StaticNGon<A> where A: Array {}

impl<A> Topological for StaticNGon<A>
where
    A: Array,
{
    type Vertex = A::Item;

    fn arity(&self) -> usize {
        A::capacity()
    }
}

macro_rules! impl_arity_static_ngon {
    (length => $n:expr) => (
        impl<T> Arity for StaticNGon<[T; $n]> {
            const ARITY: usize = $n;
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        impl<T> Arity for StaticNGon<[T; 2]> {
            const ARITY: usize = 1;
        }

        $(impl_arity_static_ngon!(length => $n);)*
    );
}
impl_arity_static_ngon!(lengths => 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

macro_rules! impl_zip_static_ngon {
    (composite => $c:ident, length => $n:expr) => (
        impl_zip_static_ngon!(composite => $c, length => $n, items => (A, B));
        impl_zip_static_ngon!(composite => $c, length => $n, items => (A, B, C));
        impl_zip_static_ngon!(composite => $c, length => $n, items => (A, B, C, D));
        impl_zip_static_ngon!(composite => $c, length => $n, items => (A, B, C, D, E));
        impl_zip_static_ngon!(composite => $c, length => $n, items => (A, B, C, D, E, F));
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

// TODO: Some inherent functions are not documented to avoid bloat.
macro_rules! impl_static_ngon {
    (length => $n:expr) => (
        impl<T> StaticNGon<[T; $n]> {
            #[doc(hidden)]
            pub fn positions(&self) -> StaticNGon<[&Position<T>; $n]>
            where
                T: AsPosition,
            {
                if let Ok(array) = self
                    .as_ref()
                    .iter()
                    .map(|vertex| vertex.as_position())
                    .collect::<ArrayVec<[_; $n]>>()
                    .into_inner()
                {
                    array.into()
                }
                else {
                    panic!()
                }
            }
        }

        impl<'a, T> StaticNGon<[&'a T; $n]> {
            #[doc(hidden)]
            pub fn cloned(self) -> StaticNGon<[T; $n]>
            where
                T: Clone,
            {
                self.map(|vertex| vertex.clone())
            }
        }

        impl<T> Converged for StaticNGon<[T; $n]>
        where
            T: Copy,
        {
            fn converged(item: T) -> Self {
                StaticNGon([item; $n])
            }
        }

        impl<T, U> Map<U> for StaticNGon<[T; $n]> {
            type Output = StaticNGon<[U; $n]>;

            fn map<F>(self, f: F) -> Self::Output
            where
                F: FnMut(Self::Item) -> U,
            {
                FromItems::from_items(self.into_iter().map(f)).unwrap()
            }
        }

        impl_zip_static_ngon!(composite => StaticNGon, length => $n);

        impl<T, U> ZipMap<U> for StaticNGon<[T; $n]> {
            type Output = StaticNGon<[U; $n]>;

            fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
            where
                F: FnMut(Self::Item, Self::Item) -> U,
            {
                FromItems::from_items(self.into_iter().zip(other).map(|(a, b)| f(a, b))).unwrap()
            }
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        $(impl_static_ngon!(length => $n);)*
    );
}
impl_static_ngon!(lengths => 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

pub type Edge<T> = StaticNGon<[T; 2]>;

impl<T> Edge<T> {
    pub fn new(a: T, b: T) -> Self {
        StaticNGon([a, b])
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

pub type Trigon<T> = StaticNGon<[T; 3]>;

impl<T> Trigon<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        StaticNGon([a, b, c])
    }

    pub fn plane(&self) -> Option<Plane<Position<T>>>
    where
        T: AsPosition,
        Position<T>: EuclideanSpace + FiniteDimensional<N = U3>,
        Vector<Position<T>>: Cross<Output = Vector<Position<T>>>,
    {
        let [a, b, c] = self.positions().cloned().into_array();
        let v = a - b;
        let u = a - c;
        Unit::try_from_inner(v.cross(u))
            .map(move |normal| Plane::<Position<T>> { origin: a, normal })
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

pub type Tetragon<T> = StaticNGon<[T; 4]>;

impl<T> Tetragon<T> {
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        StaticNGon([a, b, c, d])
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
