<div align="center">
    <img alt="Theon" src="https://raw.githubusercontent.com/olson-sean-k/theon/master/doc/theon.svg?sanitize=true" width="320"/>
</div>
<br/>

**Theon** is a Rust library that abstracts Euclidean spaces.

[![CI](https://github.com/olson-sean-k/theon/workflows/CI/badge.svg)](https://github.com/olson-sean-k/theon/actions)
[![Documentation](https://docs.rs/theon/badge.svg)](https://docs.rs/theon)
[![Crate](https://img.shields.io/crates/v/theon.svg)](https://crates.io/crates/theon)

## Geometric Traits

Theon provides geometric traits that model [Euclidean
spaces](https://en.wikipedia.org/wiki/euclidean_space). These traits are not
always mathematically rigorous, but this allows them to be implemented for many
types. Most features are limited to two- and three-dimensional Euclidean spaces,
but traits tend to be generic with respect to dimensionality.

Theon uses a _bring-your-own-types_ model, wherein a crate owner can use
features of Theon by implementing certain traits for their types. Theon also
provides optional implementations for commonly used crates in the Rust
ecosystem, including [`cgmath`](https://crates.io/crates/cgmath),
[`mint`](https://crates.io/crates/mint), and
[`nalgebra`](https://crates.io/crates/nalgebra). These implementations can be
enabled using Cargo features and integrated crates are re-exported in the
`integration` module.

| Feature             | Default | Crate      | Support  |
|---------------------|---------|------------|----------|
| `geometry-cgmath`   | No      | `cgmath`   | Complete |
| `geometry-mint`     | No      | `mint`     | Partial  |
| `geometry-nalgebra` | Yes     | `nalgebra` | Complete |

Note that these downstream implementations have an important limitation: code
that uses Theon must specify a compatible version of the given crate or depend
on re-exported crates in the `integration` module.

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
types from [`cgmath`](https://crates.io/crates/cgmath) or any other type that
implements `EuclideanSpace` and the necessary operational traits.

## LAPACK

Some queries require solving linear systems of arbitrary and non-trivial size.
To support these queries, the `lapack` feature depends on
[`ndarray`](https://crates.io/crates/ndarray) and
[LAPACK](https://en.wikipedia.org/wiki/lapack). For example,
`Plane::from_points` is enabled by the `lapack` feature and computes a best-fit
plane using a singular value decomposition.

These operations are exposed in terms of geometric traits and the implementing
types that are being used.

The `lapack` feature only supports Linux at this time.
