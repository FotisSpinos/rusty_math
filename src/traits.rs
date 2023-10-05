use crate::vector::vector::Vector;

pub trait Fillable<FillValueType> {
    fn fill(value: FillValueType) -> Self;
}

pub trait Transposable {
    type Output;

    fn transpose(&self) -> Self::Output;
}

pub trait Identity {
    fn identity() -> Self;
}

pub trait Grid2D<ComponentType, const ROWS: usize, const COLUMNS: usize> {
    fn column(&self, index: usize) -> Vector<ComponentType, ROWS>;

    fn columns(&self) -> usize;

    fn components(&self) -> [[ComponentType; COLUMNS]; ROWS];

    fn len(&self) -> usize;

    fn row(&self, index: usize) -> Vector<ComponentType, COLUMNS>;

    fn rows(&self) -> usize;
}