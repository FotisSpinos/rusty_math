mod matrix;
mod vector;
mod diagonal_matrix;

pub use matrix::matrix::*;
pub use diagonal_matrix::diagonal_matrix::*;
pub use vector::vector::*;

pub mod rusty_maths {

    pub mod traits {
        use crate::Vector;

        pub trait Grid2D<ComponentType, const ROWS: usize, const COLUMNS: usize> {
            type TransposeType;

            fn fill(value: ComponentType) -> Self;

            fn column(&self, index: usize) -> Vector<ComponentType, ROWS>;

            fn columns(&self) -> usize;

            fn components(&self) -> [[ComponentType; COLUMNS]; ROWS];

            fn len(&self) -> usize;

            fn row(&self, index: usize) -> Vector<ComponentType, COLUMNS>;

            fn rows(&self) -> usize;

            fn identity() -> Self;

            fn transpose(&self) -> Self::TransposeType;
        }
    }

}
