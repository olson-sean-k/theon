mod ncube;
mod tree;

pub use ncube::NCube;

pub trait Subdivide: Sized {
    type Output: AsRef<[Self]> + IntoIterator<Item = Self>;

    fn subdivide(&self) -> Self::Output;
}

pub trait Partition<S>: Subdivide {
    fn contains(&self, _: &S) -> bool;

    // TODO: Should this API handle elements that are not contained by the
    //       partition? Should this function return `Option<usize>`?
    fn index(&self, _: &S) -> usize;
}
