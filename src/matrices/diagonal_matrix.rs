use std::ops::{Add, AddAssign, Index, Mul, Sub, SubAssign};

use num::{Num, Zero, one, zero};

use crate::traits::{Fillable, Grid2D, Identity};
use crate::vectors::vector::Vector;

use super::matrix::Matrix;

#[derive(Debug, Copy, Clone)]
pub struct DiagonalMatrix<ComponentType, const SIZE: usize> {
    pub diagonal_components: [ComponentType; SIZE],
}

impl<ComponentType, const SIZE: usize> Grid2D<ComponentType, SIZE, SIZE>
for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Clone + Copy + Num,
{
    fn rows(&self) -> usize {
        SIZE
    }

    fn columns(&self) -> usize {
        SIZE
    }

    fn count(&self) -> usize {
        SIZE * SIZE
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

impl<ComponentType, const SIZE: usize> DiagonalMatrix<ComponentType, SIZE> {
    pub fn new(components: [ComponentType; SIZE]) -> Self {
        DiagonalMatrix { diagonal_components: components }
    }

    pub fn fill(mut self, value: ComponentType)
    where
        ComponentType: Clone,
    {
        for i in 0..SIZE {
            self.diagonal_components[i] = value.clone();
        }
    }
}

impl<ComponentType, const SIZE: usize> Add for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy + Add,
{
    type Output = DiagonalMatrix<ComponentType, SIZE>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut components = [one::<ComponentType>(); SIZE];

        for (i, component) in components.iter_mut().enumerate().take(SIZE) {
            *component = self.diagonal_components[i] + rhs.diagonal_components[i];
        }

        DiagonalMatrix::new(components)
    }
}

impl<ComponentType, const SIZE: usize> Add<Matrix<ComponentType, SIZE, SIZE>> for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy + Add,
{
    type Output = Matrix<ComponentType, SIZE, SIZE>;

    fn add(self, rhs: Self::Output) -> Self::Output {
        let mut output = rhs;

        for i in 0..SIZE {
            output.components[i][i] = output.components[i][i] + self.diagonal_components[i];
        }

        output
    }
}

impl<ComponentType, const SIZE: usize> Mul<ComponentType> for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy,
    DiagonalMatrix<ComponentType, SIZE>: Clone,
{
    type Output = DiagonalMatrix<ComponentType, SIZE>;

    fn mul(self, rhs: ComponentType) -> Self::Output {
        let mut output = self;

        for i in 0..SIZE {
            output.diagonal_components[i] = output.diagonal_components[i] * rhs;
        }

        output
    }
}

impl<ComponentType, const SIZE: usize> Mul<DiagonalMatrix<ComponentType, SIZE>> for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy,
    DiagonalMatrix<ComponentType, SIZE>: Clone,
{
    type Output = DiagonalMatrix<ComponentType, SIZE>;

    fn mul(self, rhs: DiagonalMatrix<ComponentType, SIZE>) -> Self::Output {
        let mut output = self;

        for i in 0..SIZE {
            output.diagonal_components[i] = output.diagonal_components[i] * rhs.diagonal_components[i];
        }

        output
    }
}

impl<ComponentType, const SIZE: usize> Mul<Matrix<ComponentType, SIZE, SIZE>> for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy,
    DiagonalMatrix<ComponentType, SIZE>: Clone,
{
    type Output = Matrix<ComponentType, SIZE, SIZE>;
    
    fn mul(self, rhs: Matrix<ComponentType, SIZE, SIZE>) -> Self::Output {
        let mut output = rhs;

        for i in 0..SIZE {
            output[i] = (Vector::new(output[i]) * self.diagonal_components[i]).components;
        }

        output
    }
}

impl<ComponentType, const SIZE: usize> AddAssign for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy,
    DiagonalMatrix<ComponentType, SIZE>: Add,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..SIZE {
            self.diagonal_components[i] = self.diagonal_components[i] + rhs.diagonal_components[i];
        }
    }
}

impl<ComponentType, const SIZE: usize> Sub for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy + Sub,
{
    type Output = DiagonalMatrix<ComponentType, SIZE>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut components = [one::<ComponentType>(); SIZE];

        for (i, component) in components.iter_mut().enumerate().take(SIZE) {
            *component = self.diagonal_components[i] - rhs.diagonal_components[i];
        }

        DiagonalMatrix::new(components)
    }
}

impl<ComponentType, const SIZE: usize> Sub<Matrix<ComponentType, SIZE, SIZE>> for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy + Add,
{
    type Output = Matrix<ComponentType, SIZE, SIZE>;

    fn sub(self, rhs: Self::Output) -> Self::Output {
        let mut output = rhs;

        for i in 0..SIZE {
            output.components[i][i] = output.components[i][i] - self.diagonal_components[i];
        }

        output
    }
}

impl<ComponentType, const SIZE: usize> SubAssign for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy,
    DiagonalMatrix<ComponentType, SIZE>: Add,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..SIZE {
            self.diagonal_components[i] = self.diagonal_components[i] - rhs.diagonal_components[i];
        }
    }
}

impl<ComponentType, const SIZE: usize> Index<usize> for DiagonalMatrix<ComponentType, SIZE>
where
    ComponentType: Num + Clone + Copy,
{
    type Output = ComponentType;

    fn index(&self, index: usize) -> &Self::Output {
        &self.diagonal_components[index]
    }
}
