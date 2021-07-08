use typenum::{U1, U2, U3, U4};

use crate::adjunct::{
    Adjunct, Converged, ExtendInto, Fold, IntoIterator, Linear, Map, TruncateInto, TryFromIterator,
    ZipMap,
};
use crate::space::FiniteDimensional;

impl<T, const N: usize> Adjunct for [T; N] {
    type Item = T;
}

impl<T> Adjunct for (T,) {
    type Item = T;
}

impl<T> Adjunct for (T, T) {
    type Item = T;
}

impl<T> Adjunct for (T, T, T) {
    type Item = T;
}

impl<T> Adjunct for (T, T, T, T) {
    type Item = T;
}

impl<T, const N: usize> Converged for [T; N]
where
    T: Copy,
{
    fn converged(item: T) -> Self {
        [item; N]
    }
}

impl<T> FiniteDimensional for [T; 1] {
    type N = U1;
}

impl<T> FiniteDimensional for [T; 2] {
    type N = U2;
}

impl<T> FiniteDimensional for [T; 3] {
    type N = U3;
}

impl<T> FiniteDimensional for [T; 4] {
    type N = U4;
}

impl<T> FiniteDimensional for (T,) {
    type N = U1;
}

impl<T> FiniteDimensional for (T, T) {
    type N = U2;
}

impl<T> FiniteDimensional for (T, T, T) {
    type N = U3;
}

impl<T> FiniteDimensional for (T, T, T, T) {
    type N = U4;
}

// TODO: As of this writing, it is not possible to further constrain
//       `FiniteDimensional::N` on any constant array length `N`. Once this is
//       possible, ensure that these dimensions agree.
impl<T, const N: usize> IntoIterator for [T; N]
where
    Self: Adjunct<Item = T> + FiniteDimensional,
{
    type Output = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::Output {
        std::array::IntoIter::new(self)
    }
}

impl<T, const N: usize> Linear for [T; N] {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        self.as_ref().get(index)
    }
}

impl<T> Linear for (T,) {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        if index == 0 {
            Some(&self.0)
        }
        else {
            None
        }
    }
}

impl<T> Linear for (T, T) {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.0),
            1 => Some(&self.1),
            _ => None,
        }
    }
}

impl<T> Linear for (T, T, T) {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.0),
            1 => Some(&self.1),
            2 => Some(&self.2),
            _ => None,
        }
    }
}

impl<T> Linear for (T, T, T, T) {
    fn get(&self, index: usize) -> Option<&Self::Item> {
        match index {
            0 => Some(&self.0),
            1 => Some(&self.1),
            2 => Some(&self.2),
            3 => Some(&self.3),
            _ => None,
        }
    }
}

impl<T, U> ZipMap<U> for [T; 1] {
    type Output = [U; 1];

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        let [x] = self;
        let [a] = other;
        [f(x, a)]
    }
}

impl<T, U> ZipMap<U> for [T; 2] {
    type Output = [U; 2];

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        let [x, y] = self;
        let [a, b] = other;
        [f(x, a), f(y, b)]
    }
}

impl<T, U> ZipMap<U> for [T; 3] {
    type Output = [U; 3];

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        let [x, y, z] = self;
        let [a, b, c] = other;
        [f(x, a), f(y, b), f(z, c)]
    }
}

impl<T, U> ZipMap<U> for [T; 4] {
    type Output = [U; 4];

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        let [x, y, z, w] = self;
        let [a, b, c, d] = other;
        [f(x, a), f(y, b), f(z, c), f(w, d)]
    }
}

impl<T, U> ZipMap<U> for (T,) {
    type Output = (U,);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        (f(self.0, other.0),)
    }
}

impl<T, U> ZipMap<U> for (T, T) {
    type Output = (U, U);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        (f(self.0, other.0), f(self.1, other.1))
    }
}

impl<T, U> ZipMap<U> for (T, T, T) {
    type Output = (U, U, U);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        (f(self.0, other.0), f(self.1, other.1), f(self.2, other.2))
    }
}

impl<T, U> ZipMap<U> for (T, T, T, T) {
    type Output = (U, U, U, U);

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> U,
    {
        (
            f(self.0, other.0),
            f(self.1, other.1),
            f(self.2, other.2),
            f(self.3, other.3),
        )
    }
}

macro_rules! count {
    ($x:tt $($xs:tt)*) => (1usize + count!($($xs)*));
    () => (0usize);
}
macro_rules! substitute {
    ($_t:tt, $with:ty) => {
        $with
    };
}

macro_rules! impl_converged {
    (tuple => ($j:ident $(,$i:ident)* $(,)?)) => (
        impl<T> Converged for (T, $(substitute!(($i), T),)*)
        where
            T: Clone,
        {
            fn converged(value: Self::Item) -> Self {
                $(let $i = value.clone();)*
                ($($i,)* value,)
            }
        }
    );
}
impl_converged!(tuple => (x,));
impl_converged!(tuple => (x, y));
impl_converged!(tuple => (x, y, z));
impl_converged!(tuple => (x, y, z, w));

macro_rules! impl_extend_into {
    (array => [$($i:ident),+ $(,)?]) => (
        impl<T> ExtendInto<[T; { count!($($i)*) + 1 }]> for [T; { count!($($i)*) }] {
            fn extend(self, item: Self::Item) -> [T; { count!($($i)*) + 1 }] {
                let [$($i,)+] = self;
                [$($i,)+ item]
            }
        }
    );
    (tuple => ($($i:ident),+ $(,)?)) => (
        impl<T> ExtendInto<(T, $(substitute!(($i), T),)+)> for ($(substitute!(($i), T),)+) {
            fn extend(self, item: Self::Item) -> (T, $(substitute!(($i), T),)+) {
                let ($($i,)+) = self;
                ($($i,)+ item)
            }
        }
    );
}
impl_extend_into!(array => [x]);
impl_extend_into!(array => [x, y]);
impl_extend_into!(array => [x, y, z]);
impl_extend_into!(tuple => (x,));
impl_extend_into!(tuple => (x, y));
impl_extend_into!(tuple => (x, y, z));

macro_rules! impl_fold {
    (array => [$($i:ident),+ $(,)?]) => (
        impl<T> Fold for [T; { count!($($i)*) }] {
            fn fold<U, F>(self, mut seed: U, mut f: F) -> U
            where
                F: FnMut(U, Self::Item) -> U,
            {
                let [$($i,)+] = self;
                $(seed = f(seed, $i);)+
                seed
            }
        }
    );
    (tuple => ($($i:ident),+ $(,)?)) => (
        impl<T> Fold for ($(substitute!(($i), T),)+) {
            fn fold<U, F>(self, mut seed: U, mut f: F) -> U
            where
                F: FnMut(U, Self::Item) -> U,
            {
                let ($($i,)+) = self;
                $(seed = f(seed, $i);)+
                seed
            }
        }
    );
}
impl_fold!(array => [x]);
impl_fold!(array => [x, y]);
impl_fold!(array => [x, y, z]);
impl_fold!(array => [x, y, z, w]);
impl_fold!(tuple => (x,));
impl_fold!(tuple => (x, y));
impl_fold!(tuple => (x, y, z));
impl_fold!(tuple => (x, y, z, w));

macro_rules! impl_into_iterator {
    (tuple => ($($i:ident),+ $(,)?)) => (
        #[allow(non_snake_case)]
        impl<T> IntoIterator for ($(substitute!(($i), T),)+) {
            type Output = std::array::IntoIter<T, { count!($($i)*) }>;

            fn into_iter(self) -> Self::Output {
                let ($($i,)+) = self;
                std::array::IntoIter::new([$($i,)+])
            }
        }
    );
}
impl_into_iterator!(tuple => (x,));
impl_into_iterator!(tuple => (x, y));
impl_into_iterator!(tuple => (x, y, z));
impl_into_iterator!(tuple => (x, y, z, w));

macro_rules! impl_map {
    (array => [$($i:ident),+ $(,)?]) => (
        impl<T, U> Map<U> for [T; { count!($($i)*) }] {
            type Output = [U; { count!($($i)*) }];

            fn map<F>(self, mut f: F) -> Self::Output
            where
                F: FnMut(Self::Item) -> U,
            {
                let [$($i,)+] = self;
                [$(f($i),)+]
            }
        }
    );
    (tuple => ($($i:ident),+ $(,)?)) => (
        impl<T, U> Map<U> for ($(substitute!(($i), T),)+) {
            type Output = ($(substitute!(($i), U),)+);

            fn map<F>(self, mut f: F) -> Self::Output
            where
                F: FnMut(Self::Item) -> U,
            {
                let ($($i,)+) = self;
                ($(f($i),)+)
            }
        }
    );
}
impl_map!(array => [x]);
impl_map!(array => [x, y]);
impl_map!(array => [x, y, z]);
impl_map!(array => [x, y, z, w]);
impl_map!(tuple => (x,));
impl_map!(tuple => (x, y));
impl_map!(tuple => (x, y, z));
impl_map!(tuple => (x, y, z, w));

macro_rules! impl_truncate_into {
    (array => [$($i:ident),+ $(,)?]) => (
        impl<T> TruncateInto<[T; { count!($($i)*) }]> for [T; { count!($($i)*) + 1 }] {
            fn truncate(self) -> ([T; { count!($($i)*) }], T) {
                let [$($i,)+ last] = self;
                ([$($i,)+], last)
            }
        }
    );
    (tuple => ($($i:ident),+ $(,)?)) => (
        impl<T> TruncateInto<($(substitute!(($i), T),)+)> for (T, $(substitute!(($i), T),)+) {
            fn truncate(self) -> (($(substitute!(($i), T),)+), T) {
                let ($($i,)+ last) = self;
                (($($i,)+), last)
            }
        }
    );
}
impl_truncate_into!(array => [x]);
impl_truncate_into!(array => [x, y]);
impl_truncate_into!(array => [x, y, z]);
impl_truncate_into!(tuple => (x,));
impl_truncate_into!(tuple => (x, y));
impl_truncate_into!(tuple => (x, y, z));

macro_rules! impl_try_from_iterator {
    (tuple => ($($i:ident),+ $(,)?)) => (
        #[allow(non_snake_case)]
        impl<I, T> TryFromIterator<I> for ($(substitute!(($i), T),)+)
        where
            I: Iterator<Item = T>,
        {
            type Error = ();
            type Remainder = std::iter::Peekable<std::iter::Fuse<I>>;

            fn try_from_iter(items: I) -> Result<(Self, Option<Self::Remainder>), Self::Error> {
                let mut items = items.fuse();
                $(let $i = items.next().ok_or(())?;)+
                let tuple = (
                    $($i,)+
                );
                let mut remainder = items.peekable();
                if remainder.peek().is_some() {
                    Ok((tuple, Some(remainder)))
                }
                else {
                    Ok((tuple, None))
                }
            }
        }
    );
    (array => [$($i:ident),+ $(,)?]) => (
        #[allow(non_snake_case)]
        impl<I, T> TryFromIterator<I> for [T; { count!($($i)*) }]
        where
            I: Iterator<Item = T>,
        {
            type Error = ();
            type Remainder = std::iter::Peekable<std::iter::Fuse<I>>;

            fn try_from_iter(items: I) -> Result<(Self, Option<Self::Remainder>), Self::Error> {
                let mut items = items.fuse();
                $(let $i = items.next().ok_or(())?;)+
                let array = [
                    $($i,)+
                ];
                let mut remainder = items.peekable();
                if remainder.peek().is_some() {
                    Ok((array, Some(remainder)))
                }
                else {
                    Ok((array, None))
                }
            }
        }
    );
}
impl_try_from_iterator!(array => [x]);
impl_try_from_iterator!(array => [x, y]);
impl_try_from_iterator!(array => [x, y, z]);
impl_try_from_iterator!(array => [x, y, z, w]);
impl_try_from_iterator!(tuple => (x,));
impl_try_from_iterator!(tuple => (x, y));
impl_try_from_iterator!(tuple => (x, y, z));
impl_try_from_iterator!(tuple => (x, y, z, w));
