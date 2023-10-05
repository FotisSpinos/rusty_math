use std::ops::{SubAssign, Add, Sub, AddAssign, Mul};

use num::{Num, Zero, one, zero};

use crate::{Vector, traits::{Fillable, Grid2D, Identity}};

use super::matrix::Matrix;

pub struct DiagonalMatrix<T, const SIZE: usize> {
    pub diagonal_components: [T; SIZE],
}

impl<ComponentType, const SIZE: usize> Grid2D<ComponentType, SIZE, SIZE>
for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Clone + Copy + Num,
{
    fn column(&self, index: usize) -> Vector<ComponentType, SIZE> {
        todo!()
    }

    fn columns(&self) -> usize {
        SIZE
    }

    fn components(&self) -> [[ComponentType; SIZE]; SIZE] {
        todo!()
    }

    fn len(&self) -> usize {
        SIZE * SIZE
    }

    fn row(&self, index: usize) -> Vector<ComponentType, SIZE> {
        todo!()
    }

    fn rows(&self) -> usize {
        todo!()
    }
}

impl<ComponentType, const SIZE: usize> Identity
for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Clone + Copy + Num,
{
    fn identity() -> Self {
        let components = [one::<ComponentType>(); SIZE];
        DiagonalMatrix::new(components)
    }
}

impl<ComponentType, const SIZE: usize> Fillable<ComponentType> for DiagonalMatrix<ComponentType, SIZE>
where ComponentType: Zero + std::marker::Copy {

    fn fill(value: ComponentType) -> Self {
        let components = [zero(); SIZE];
        components.into_iter().for_each(|mut _component: ComponentType| {
            _component = value
        });
        DiagonalMatrix::new(components)
    }
}

impl<T, const SIZE: usize> DiagonalMatrix<T, SIZE> {
    pub fn new(components: [T; SIZE]) -> Self {
        DiagonalMatrix { diagonal_components: components }
    }

    pub fn fill(mut self, value: T)
    where
        T: Clone,
    {
        for i in 0..SIZE {
            self.diagonal_components[i] = value.clone();
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
            components[i] = self.diagonal_components[i] + rhs.diagonal_components[i];
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
            output.components[i][i] = output.components[i][i] + self.diagonal_components[i];
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
            output.diagonal_components[i] = output.diagonal_components[i] * rhs;
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
            output.diagonal_components[i] = output.diagonal_components[i] * rhs.diagonal_components[i];
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
            self.diagonal_components[i] = self.diagonal_components[i] + rhs.diagonal_components[i];
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
            components[i] = self.diagonal_components[i] - rhs.diagonal_components[i];
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
            output.components[i][i] = output.components[i][i] - self.diagonal_components[i];
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
            self.diagonal_components[i] = self.diagonal_components[i] - rhs.diagonal_components[i];
        }
    }
}
