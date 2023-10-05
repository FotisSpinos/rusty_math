use num::{one, zero, Num, Zero};

use std::{
    cmp::PartialEq,
    ops::{
        Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign,
    },
};

use crate::{traits::{Fillable, Transposable, Grid2D, Identity}, vector::Vector};

pub type SquareMatrix<ComponentType, const SIZE: usize> = Matrix<ComponentType, SIZE, SIZE>;
pub type Matrix2x2<ComponentType> = SquareMatrix<ComponentType, 2>;
pub type Matrix3x3<ComponentType> = SquareMatrix<ComponentType, 3>;
pub type Matrix4x4<ComponentType> = SquareMatrix<ComponentType, 4>;

trait MatrixTrait<ComponentType, const ROWS: usize, const COLUMNS: usize> : Transposable + Fillable<ComponentType> + Grid2D<ComponentType, ROWS, COLUMNS> {

}

#[derive(Debug, Copy, Clone)]
pub struct Matrix<ComponentType, const ROWS: usize, const COLUMNS: usize>
where
    ComponentType: Clone,
{
    pub components: [[ComponentType; COLUMNS]; ROWS],
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Clone + Copy,
{
    pub fn new(components: [[ComponentType; COLUMNS]; ROWS]) -> Self {
        Matrix::<ComponentType, ROWS, COLUMNS> { components }
    }

    pub fn pow(matrix: Self, exponent: usize) -> Self
    where Self : Mul<Self, Output = Self> {
        let mut current = matrix;

        (0..exponent - 1).for_each(|_: usize| {
            current = current * matrix;
        });

        current
    }

    pub fn diagonal_matrix_mul(lhs: Self, rhs: Vector<ComponentType, COLUMNS>) -> Vector<ComponentType, ROWS>
    where
        ComponentType: Zero + Mul<Output = ComponentType>,
    {
        let mut result = Vector::<ComponentType, ROWS>::new([zero(); ROWS]);

        for i in 0..COLUMNS {
            for v in 0..i {
                result.components[i] =
                    result.components[i] + lhs.components[v][i] * rhs.components[v];
            }
            for j in i..ROWS {
                result.components[i] =
                    result.components[i] + lhs.components[i][j] * rhs.components[j];
            }
        }

        result
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Grid2D<ComponentType, ROWS, COLUMNS>
    for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Clone + Copy + Num,
{
    fn column(&self, index: usize) -> Vector<ComponentType, ROWS> {
        let mut output = [zero::<ComponentType>(); ROWS];

        for i in 0..ROWS {
            output[i] = self.components[i][index];
        }

        Vector::<ComponentType, ROWS>::new(output)
    }

    fn columns(&self) -> usize {
        COLUMNS
    }

    fn components(&self) -> [[ComponentType; COLUMNS]; ROWS] {
        self.components
    }

    fn len(&self) -> usize {
        self.rows() * self.columns()
    }

    fn row(&self, index: usize) -> Vector<ComponentType, COLUMNS> {
        Vector::new(self.components[index])
    }

    fn rows(&self) -> usize {
        ROWS
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Identity
for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Clone + Copy + Num,
{
    fn identity() -> Self {
        let mut components: [[ComponentType; COLUMNS]; ROWS] = [[zero::<ComponentType>(); COLUMNS]; ROWS];

        for row in 0..ROWS {
            components[row][row] = one::<ComponentType>();
        }

        Matrix::new(components)
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Transposable for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Clone + Copy + Num
{
    type Output = Matrix<ComponentType, COLUMNS, ROWS>;

    fn transpose(&self) -> Self::Output {
        let mut output = Matrix::<ComponentType, COLUMNS, ROWS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                output.components[x][y] = self.components[y][x];
            }
        }

        output
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Fillable<ComponentType> for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Clone + Copy + Num
{

    fn fill(value: ComponentType) -> Self {
        let mut components: [[ComponentType; COLUMNS]; ROWS] = [[zero::<ComponentType>(); COLUMNS]; ROWS];

        for rows in components.iter_mut().take(ROWS) {
            for columns in rows.iter_mut().take(COLUMNS) {
                *columns = value;
            }
        }

        Matrix::new(components)
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Zero for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Clone + Copy + Num
{
    fn zero() -> Self {
        let components: [[ComponentType; COLUMNS]; ROWS] = [[zero::<ComponentType>(); COLUMNS]; ROWS];
        Matrix::new(components)
    }

    fn set_zero(&mut self) {
        *self = Zero::zero();
    }

    fn is_zero(&self) -> bool
    where
        ComponentType: PartialEq + num::Zero
    {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                if self.components[y][x] != zero() {
                    return false
                }
            }
        }
        true
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Add for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    type Output = Matrix<ComponentType, ROWS, COLUMNS>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut matrix = Matrix::<ComponentType, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] + rhs.components[y][x];
            }
        }

        matrix
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> AddAssign for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] + rhs.components[y][x];
            }
        }
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Sub for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    type Output = Matrix<ComponentType, ROWS, COLUMNS>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut matrix = Matrix::<ComponentType, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] - rhs.components[y][x];
            }
        }

        matrix
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> SubAssign for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] - rhs.components[y][x];
            }
        }
    }
}

impl<ComponentType, const LHS_ROWS: usize, const LHS_COLUMNS: usize, const RHS_COLUMN: usize>
    Mul<Matrix<ComponentType, LHS_COLUMNS, RHS_COLUMN>> for Matrix<ComponentType, LHS_ROWS, LHS_COLUMNS>
where
    ComponentType: Num + Clone + Copy + AddAssign,
{
    type Output = Matrix<ComponentType, LHS_ROWS, RHS_COLUMN>;

    fn mul(
        self,
        rhs: Matrix<ComponentType, LHS_COLUMNS, RHS_COLUMN>,
    ) -> Matrix<ComponentType, LHS_ROWS, RHS_COLUMN> {
        let mut matrix = Matrix::<ComponentType, LHS_ROWS, RHS_COLUMN>::zero();

        for lhs_row in 0..LHS_ROWS {
            for rhs_column in 0..RHS_COLUMN {
                for lhs_column in 0..LHS_COLUMNS {
                    matrix[lhs_row][rhs_column] +=
                        self[lhs_row][lhs_column] * rhs[lhs_column][rhs_column];
                }
            }
        }

        matrix
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Mul<ComponentType> for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    type Output = Matrix<ComponentType, ROWS, COLUMNS>;

    fn mul(self, scalar: ComponentType) -> Self::Output {
        let mut matrix = Matrix::<ComponentType, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] * scalar;
            }
        }

        matrix
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Mul<Vector<ComponentType, COLUMNS>>
    for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    type Output = Vector<ComponentType, ROWS>;

    fn mul(self, rhs: Vector<ComponentType, COLUMNS>) -> Self::Output {
        let mut result = Self::Output::new([zero(); ROWS]);

        for i in 0..ROWS {
            for j in 0..COLUMNS {
                result.components[i] =
                    result.components[i] + self.components[i][j] * rhs.components[j];
            }
        }

        result
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> MulAssign<ComponentType> for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    fn mul_assign(&mut self, scalar: ComponentType) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] * scalar;
            }
        }
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Div<ComponentType> for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    type Output = Matrix<ComponentType, ROWS, COLUMNS>;

    fn div(self, scalar: ComponentType) -> Self::Output {
        let mut matrix = Matrix::<ComponentType, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] / scalar;
            }
        }

        matrix
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> DivAssign<ComponentType> for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    fn div_assign(&mut self, scalar: ComponentType) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] / scalar;
            }
        }
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> Index<usize> for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    type Output = [ComponentType; COLUMNS];

    fn index(&self, index: usize) -> &Self::Output {
        &self.components[index]
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> IndexMut<usize> for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.components[index]
    }
}

impl<ComponentType, const ROWS: usize, const COLUMNS: usize> PartialEq for Matrix<ComponentType, ROWS, COLUMNS>
where
    ComponentType: Num + Clone + Copy,
{
    fn eq(&self, other: &Self) -> bool {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                if self.components[y][x] != other[y][x] {
                    return false;
                }
            }
        }
        true
    }
}
