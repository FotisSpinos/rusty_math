pub type SquareMatrix<T, const SIZE: usize> = Matrix<T, SIZE, SIZE>;
pub type Matrix3x3<T> = SquareMatrix<T, 3>;
pub type Matrix4x4<T> = SquareMatrix<T, 4>;

use num::{one, zero, Num, Zero};

use std::{
    cmp::PartialEq,
    ops::{
        Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign,
    },
};

use crate::{rusty_maths::traits::Grid2D, vector::vector::Vector};

#[derive(Debug, Copy, Clone)]
pub struct Matrix<T, const ROWS: usize, const COLUMNS: usize>
where
    T: Clone,
{
    pub components: [[T; COLUMNS]; ROWS],
}

impl<T, const ROWS: usize, const COLUMNS: usize> Matrix<T, ROWS, COLUMNS>
where
    T: Clone + Copy,
{
    pub fn new(components: [[T; COLUMNS]; ROWS]) -> Self {
        Matrix::<T, ROWS, COLUMNS> { components }
    }

    pub fn diagonal_matrix_mul(lhs: Self, rhs: Vector<T, COLUMNS>) -> Vector<T, ROWS>
    where
        T: Zero + Mul<Output = T>,
    {
        let mut result = Vector::<T, ROWS>::new([zero(); ROWS]);

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

impl<T, const ROWS: usize, const COLUMNS: usize> Grid2D<T, ROWS, COLUMNS>
    for Matrix<T, ROWS, COLUMNS>
where
    T: Clone + Copy + Num,
{
    type TransposeType = Matrix<T, COLUMNS, ROWS>;
    fn fill(value: T) -> Self {
        let mut components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];

        for rows in components.iter_mut().take(ROWS) {
            for columns in rows.iter_mut().take(COLUMNS) {
                *columns = value;
            }
        }

        Matrix::new(components)
    }

    fn column(&self, index: usize) -> Vector<T, ROWS> {
        let mut output = [zero::<T>(); ROWS];

        for i in 0..ROWS {
            output[i] = self.components[i][index];
        }

        Vector::<T, ROWS>::new(output)
    }

    fn columns(&self) -> usize {
        COLUMNS
    }

    fn components(&self) -> [[T; COLUMNS]; ROWS] {
        self.components
    }

    fn len(&self) -> usize {
        self.rows() * self.columns()
    }

    fn row(&self, index: usize) -> Vector<T, COLUMNS> {
        Vector::new(self.components[index])
    }

    fn rows(&self) -> usize {
        ROWS
    }

    fn identity() -> Self {
        let mut components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];

        for row in 0..ROWS {
            components[row][row] = one::<T>();
        }

        Matrix::new(components)
    }

    fn transpose(&self) -> Self::TransposeType {
        let mut output = Matrix::<T, COLUMNS, ROWS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                output.components[x][y] = self.components[y][x];
            }
        }

        output
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> Zero for Matrix<T, ROWS, COLUMNS>
where
    T: Clone + Copy + Num
{
    fn zero() -> Self {
        let components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];
        Matrix::new(components)
    }

    fn set_zero(&mut self) {
        *self = Zero::zero();
    }

    fn is_zero(&self) -> bool
    where
        T: PartialEq + num::Zero
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

impl<T, const ROWS: usize, const COLUMNS: usize> Add for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    type Output = Matrix<T, ROWS, COLUMNS>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut matrix = Matrix::<T, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] + rhs.components[y][x];
            }
        }

        matrix
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> AddAssign for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] + rhs.components[y][x];
            }
        }
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> Sub for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    type Output = Matrix<T, ROWS, COLUMNS>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut matrix = Matrix::<T, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] - rhs.components[y][x];
            }
        }

        matrix
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> SubAssign for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] - rhs.components[y][x];
            }
        }
    }
}

impl<T, const LHS_ROWS: usize, const LHS_COLUMNS: usize, const RHS_COLUMN: usize>
    Mul<Matrix<T, LHS_COLUMNS, RHS_COLUMN>> for Matrix<T, LHS_ROWS, LHS_COLUMNS>
where
    T: Num + Clone + Copy + AddAssign,
{
    type Output = Matrix<T, LHS_ROWS, RHS_COLUMN>;

    fn mul(
        self,
        rhs: Matrix<T, LHS_COLUMNS, RHS_COLUMN>,
    ) -> Matrix<T, LHS_ROWS, RHS_COLUMN> {
        let mut matrix = Matrix::<T, LHS_ROWS, RHS_COLUMN>::zero();

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

impl<T, const ROWS: usize, const COLUMNS: usize> Mul<T> for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    type Output = Matrix<T, ROWS, COLUMNS>;

    fn mul(self, scalar: T) -> Self::Output {
        let mut matrix = Matrix::<T, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] * scalar;
            }
        }

        matrix
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> Mul<Vector<T, COLUMNS>>
    for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    type Output = Vector<T, ROWS>;

    fn mul(self, rhs: Vector<T, COLUMNS>) -> Self::Output {
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

impl<T, const ROWS: usize, const COLUMNS: usize> MulAssign<T> for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    fn mul_assign(&mut self, scalar: T) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] * scalar;
            }
        }
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> Div<T> for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    type Output = Matrix<T, ROWS, COLUMNS>;

    fn div(self, scalar: T) -> Self::Output {
        let mut matrix = Matrix::<T, ROWS, COLUMNS>::zero();

        for y in 0..ROWS {
            for x in 0..COLUMNS {
                matrix.components[y][x] = self.components[y][x] / scalar;
            }
        }

        matrix
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> DivAssign<T> for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    fn div_assign(&mut self, scalar: T) {
        for y in 0..ROWS {
            for x in 0..COLUMNS {
                self.components[y][x] = self.components[y][x] / scalar;
            }
        }
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> Index<usize> for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    type Output = [T; COLUMNS];

    fn index(&self, index: usize) -> &Self::Output {
        &self.components[index]
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> IndexMut<usize> for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.components[index]
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> PartialEq for Matrix<T, ROWS, COLUMNS>
where
    T: Num + Clone + Copy,
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
