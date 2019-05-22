**Theon** is a Rust library that abstracts Euclidean spaces.

[![Build Status](https://travis-ci.org/olson-sean-k/theon.svg?branch=master)](https://travis-ci.org/olson-sean-k/theon)
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
enabled using Cargo features.

| Feature             | Default | Crate      | Support  |
|---------------------|---------|------------|----------|
| `geometry-cgmath`   | No      | `cgmath`   | Complete |
| `geometry-mint`     | No      | `mint`     | Partial  |
| `geometry-nalgebra` | Yes     | `nalgebra` | Complete |

## Spatial Queries

Geometric queries can be performed using any types that implement the
appropriate geometric traits.

```rust
use nalgebra::Point2;
use theon::query::{Aabb, Intersection, Ray, Unit};
use theon::space::{Basis, EuclideanSpace};
use theon::Converged;

type E2 = Point2<f64>;

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
```

In the above example, it is possible to replace the `E2` type definition with
types from [`cgmath`](https://crates.io/crates/cgmath) or any other type that
implements `EuclideanSpace` and the necessary operational traits.
