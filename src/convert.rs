use arrayvec::ArrayVec;

use crate::Category;

pub trait IntoObjects: Category {
    type Output: IntoIterator<Item = Self::Object>;

    fn into_objects(self) -> Self::Output;
}

impl<T> IntoObjects for (T, T) {
    type Output = ArrayVec<[T; 2]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.0, self.1])
    }
}

impl<T> IntoObjects for (T, T, T) {
    type Output = ArrayVec<[T; 3]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.0, self.1, self.2])
    }
}

pub trait FromObjects: Category + Sized {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>;
}

impl<T> FromObjects for (T, T) {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(2);
        match (objects.next(), objects.next()) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }
}

impl<T> FromObjects for (T, T, T) {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(3);
        match (objects.next(), objects.next(), objects.next()) {
            (Some(a), Some(b), Some(c)) => Some((a, b, c)),
            _ => None,
        }
    }
}
