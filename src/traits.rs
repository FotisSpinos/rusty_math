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
    fn columns(&self) -> usize;

    fn len(&self) -> usize;

    fn rows(&self) -> usize;
}