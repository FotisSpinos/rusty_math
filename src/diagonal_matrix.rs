pub mod matrix {

    use num::{one, zero, Num, Zero};

    use std::{
        cmp::PartialEq,
        ops::{
            Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign,
        },
    };

    use crate::{Vector, rusty_maths::traits::Grid2D};

    pub struct DiagonalMatrix<T, const SIZE: usize> {
        pub components: [T; SIZE],
    }

    impl<ComponentType, const SIZE: usize> Grid2D<ComponentType, SIZE, SIZE>
    for DiagonalMatrix<ComponentType, SIZE>
    where
        ComponentType: Clone + Copy + Num,
    {

        type TransposeType = Self;

        fn fill(value: ComponentType) -> Self {
            todo!()
        }

        fn column(&self, index: usize) -> Vector<ComponentType, SIZE> {
            todo!()
        }

        fn columns(&self) -> usize {
            todo!()
        }

        fn components(&self) -> [[ComponentType; SIZE]; SIZE] {
            todo!()
        }

        fn len(&self) -> usize {
            todo!()
        }

        fn row(&self, index: usize) -> Vector<ComponentType, SIZE> {
            todo!()
        }

        fn rows(&self) -> usize {
            todo!()
        }

        fn identity() -> Self {
            let components = [one::<ComponentType>(); SIZE];
            DiagonalMatrix::new(components)
        }

        fn transpose(&self) -> Self::TransposeType {
            todo!()
        }
    }

    impl<T, const SIZE: usize> DiagonalMatrix<T, SIZE> {
        pub fn new(components: [T; SIZE]) -> Self {
            DiagonalMatrix { components }
        }

        pub fn fill(mut self, value: T)
        where
            T: Clone,
        {
            for i in 0..SIZE {
                self.components[i] = value.clone();
            }
        }
    }

    impl<T, const SIZE: usize> Add for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy + Add,
    {
        type Output = DiagonalMatrix<T, SIZE>;

        fn add(self, rhs: Self) -> Self::Output {
            let mut components = [one::<T>(); SIZE];

            for i in 0..SIZE {
                components[i] = self.components[i] + rhs.components[i];
            }

            DiagonalMatrix::new(components)
        }
    }

    impl<T, const SIZE: usize> Add<Matrix<T, SIZE, SIZE>> for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy + Add,
    {
        type Output = Matrix<T, SIZE, SIZE>;

        fn add(self, rhs: Self::Output) -> Self::Output {
            let mut output = rhs.clone();

            for i in 0..SIZE {
                output.components[i][i] = output.components[i][i] + self.components[i];
            }

            output
        }
    }

    impl<T, const SIZE: usize> Mul<T> for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy,
        DiagonalMatrix<T, SIZE>: Clone,
    {
        type Output = DiagonalMatrix<T, SIZE>;

        fn mul(self, rhs: T) -> Self::Output {
            let mut output = self.clone();

            for i in 0..SIZE {
                output.components[i] = output.components[i] * rhs;
            }

            output
        }
    }

    impl<T, const SIZE: usize> Mul<DiagonalMatrix<T, SIZE>> for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy,
        DiagonalMatrix<T, SIZE>: Clone,
    {
        type Output = DiagonalMatrix<T, SIZE>;

        fn mul(self, rhs: DiagonalMatrix<T, SIZE>) -> Self::Output {
            let mut output = self.clone();

            for i in 0..SIZE {
                output.components[i] = output.components[i] * rhs.components[i];
            }

            output
        }
    }

    impl<T, const SIZE: usize> AddAssign for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy,
        DiagonalMatrix<T, SIZE>: Add,
    {
        fn add_assign(&mut self, rhs: Self) {
            for i in 0..SIZE {
                self.components[i] = self.components[i] + rhs.components[i];
            }
        }
    }

    impl<T, const SIZE: usize> Sub for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy + Sub,
    {
        type Output = DiagonalMatrix<T, SIZE>;

        fn sub(self, rhs: Self) -> Self::Output {
            let mut components = [one::<T>(); SIZE];

            for i in 0..SIZE {
                components[i] = self.components[i] - rhs.components[i];
            }

            DiagonalMatrix::new(components)
        }
    }

    impl<T, const SIZE: usize> Sub<Matrix<T, SIZE, SIZE>> for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy + Add,
    {
        type Output = Matrix<T, SIZE, SIZE>;

        fn sub(self, rhs: Self::Output) -> Self::Output {
            let mut output = rhs.clone();

            for i in 0..SIZE {
                output.components[i][i] = output.components[i][i] - self.components[i];
            }

            output
        }
    }

    impl<T, const SIZE: usize> SubAssign for DiagonalMatrix<T, SIZE>
    where
        T: Num + Clone + Copy,
        DiagonalMatrix<T, SIZE>: Add,
    {
        fn sub_assign(&mut self, rhs: Self) {
            for i in 0..SIZE {
                self.components[i] = self.components[i] - rhs.components[i];
            }
        }
    }
}
