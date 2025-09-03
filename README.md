<div align="center">
    <img alt="Theon" src="https://raw.githubusercontent.com/olson-sean-k/theon/master/doc/theon.svg?sanitize=true" width="320"/>
</div>
<br/>

**Theon** is a Rust library that abstracts Euclidean spaces and integrates with
various crates and types in the Rust ecosystem. Theon can be used by libraries
to avoid choosing specific math or linear algebra crates on behalf of their
dependents.

[![GitHub](https://img.shields.io/badge/GitHub-olson--sean--k/theon-8da0cb?logo=github&style=for-the-badge)](https://github.com/olson-sean-k/theon)
[![docs.rs](https://img.shields.io/badge/docs.rs-theon-66c2a5?logo=rust&style=for-the-badge)](https://docs.rs/theon)
[![crates.io](https://img.shields.io/crates/v/theon.svg?logo=rust&style=for-the-badge)](https://crates.io/crates/theon)

## Geometric Traits

Theon provides geometric traits that model [Euclidean spaces][space]. These
traits are not always mathematically rigorous, but this allows them to be
implemented for many types. APIs are designed for computations in lower
dimensional Euclidean spaces, but traits and types are generic with respect to
dimension.

## Spatial Queries

Geometric queries can be performed using any types that implement the
appropriate geometric traits.

```rust
use nalgebra::Point2;
use theon::adjunct::Converged;
use theon::query::{Aabb, Intersection, Ray, Unit};
use theon::space::EuclideanSpace;

type E2 = Point2<f64>;

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
```

In the above example, it is possible to replace the `E2` type definition with
types from [`cgmath`], [`glam`], or any other type that implements
`EuclideanSpace`, etc. See below.

## Integrations

Theon provides reverse integrations with commonly used linear algebra crates in
the Rust ecosystem, including [`glam`] and [`ultraviolet`]. These
implementations can be enabled using feature flags (none are enabled by
default).

| Feature       | Crate           |
|---------------|-----------------|
| `cgmath`      | [`cgmath`]      |
| `glam`        | [`glam`]        |
| `mint`        | [`mint`]        |
| `nalgebra`    | [`nalgebra`]    |
| `ultraviolet` | [`ultraviolet`] |

Because a given version of Theon implements traits for specific versions of
integrated crates, **care must be taken to resolve to these supported
versions**. Ideally, integrations would be implemented in these linear algebra
crates, but Theon is still under development and may not be ready for forward
integration.

In libaries or middleware crates, these features can be forwarded so that users
can activate support for a particular implementation as needed. This can be done
in the crate manifest.

```toml
[package]
name = "example"

[features]
default = []

# Forward features to Theon.

cgmath = ["theon/cgmath"]
glam = ["theon/glam"]
mint = ["theon/mint"]
nalgebra = ["theon/nalgebra"]
ultraviolet = ["theon/ultraviolet"]

[dependencies]
theon = "^0.2.0"
```

Note that [`mint`] is also an interoperability layer, though it takes a
different approach from this crate and is designed to bridge between various
mathematics crates via conversions. Theon provides limited support for using its
data types directly.

## LAPACK

Some queries require solving linear systems of arbitrary and non-trivial size.
To support these queries, the `lapack` feature depends on [`ndarray`] and
[LAPACK][lapack] libraries. For example, `Plane::from_points` is enabled by the
`lapack` feature and computes a best-fit plane using a singular value
decomposition.

The `lapack` feature can only be used with the `x86_64` architecture on Linux,
MacOS, and Windows.

[space]: https://en.wikipedia.org/wiki/euclidean_space
[lapack]: https://en.wikipedia.org/wiki/lapack

[`cgmath`]: https://crates.io/crates/cgmath
[`glam`]: https://crates.io/crates/glam
[`mint`]: https://crates.io/crates/mint
[`nalgebra`]: https://crates.io/crates/nalgebra
[`ndarray`]: https://crates.io/crates/ndarray
[`ultraviolet`]: https://crates.io/crates/ultraviolet
