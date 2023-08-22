mod matrix;
mod vector;

pub use matrix::matrix::*;
pub use vector::rusty_maths::vector::*;

pub mod rusty_maths {

    pub mod traits {
        use crate::Vector;

        pub trait Transpose<ReturnType> {
            fn transpose(&self) -> ReturnType;
        }

        pub trait Fill<ReturnType, FillType>
        where
            FillType: num::Num,
        {
            fn fill(value: FillType) -> ReturnType;
        }

        pub trait Identity<ReturnType> {
            fn identity() -> ReturnType;
        }

        pub trait Array2D<ComponentType, const ROWS: usize, const COLUMNS: usize> {
            fn column(&self, index: usize) -> Vector<ComponentType, ROWS>;

            fn columns(&self) -> usize;

            fn components(&self) -> [[ComponentType; COLUMNS]; ROWS];

            fn len(&self) -> usize;

            fn row(&self, index: usize) -> Vector<ComponentType, COLUMNS>;

            fn rows(&self) -> usize;
        }
    }

}
