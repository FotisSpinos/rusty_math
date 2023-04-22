pub mod rusty_maths {

    pub mod traits {
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
            fn column(&self, index: usize) -> [ComponentType; ROWS];

            fn columns(&self) -> usize;

            fn components(&self) -> [[ComponentType; COLUMNS]; ROWS];

            fn count(&self) -> usize;

            fn row(&self, index: usize) -> [ComponentType; COLUMNS];

            fn rows(&self) -> usize;
        }
    }

    pub mod matrix {

        use super::traits::{Array2D, Fill, Identity, Transpose};

        use num::{one, zero, Num, Zero};
        use std::{
            cmp::PartialEq, ops::Add, ops::AddAssign, ops::Div, ops::DivAssign, ops::Index,
            ops::IndexMut, ops::Mul, ops::MulAssign, ops::Sub, ops::SubAssign
        };

        #[derive(Copy, Clone)]
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
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Array2D<T, ROWS, COLUMNS> for Matrix<T, ROWS, COLUMNS>
        where
            T: Clone + Copy + Num,
        {
            fn row(&self, index: usize) -> [T; COLUMNS] {
                self.components[index]
            }

            fn column(&self, index: usize) -> [T; ROWS] {
                let mut output = [zero::<T>(); ROWS];

                for i in 0..ROWS {
                    output[i] = self.components[i][index];
                }

                output
            }

            fn rows(&self) -> usize {
                ROWS
            }

            fn columns(&self) -> usize {
                COLUMNS
            }

            fn count(&self) -> usize {
                self.rows() * self.columns()
            }

            fn components(&self) -> [[T; COLUMNS]; ROWS] {
                self.components
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Transpose<Matrix<T, COLUMNS, ROWS>>
            for Matrix<T, ROWS, COLUMNS>
        where
            T: Clone + Copy + Num,
        {
            fn transpose(&self) -> Matrix<T, COLUMNS, ROWS> {
                let mut output = Matrix::<T, COLUMNS, ROWS>::zero();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        output.components[x][y] = self.components[y][x];
                    }
                }

                output
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Fill<Matrix<T, ROWS, COLUMNS>, T>
            for Matrix<T, ROWS, COLUMNS>
        where
            T: Clone + Copy + Num,
        {
            fn fill(value: T) -> Self {
                let mut components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];

                for rows in components.iter_mut().take(ROWS) {
                    for columns in rows.iter_mut().take(COLUMNS) {
                        *columns = value;
                    }
                }

                Matrix::new(components)
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Identity<Matrix<T, ROWS, COLUMNS>>
            for Matrix<T, ROWS, COLUMNS>
        where
            T: Clone + Copy + Num,
        {
            fn identity() -> Self {
                let mut components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];

                for row in 0..ROWS {
                    components[row][row] = one::<T>();
                }

                Matrix::new(components)
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Zero for Matrix<T, ROWS, COLUMNS>
        where
            T: Clone + Copy + Num,
        {
            fn zero() -> Self {
                let components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];
                Matrix::new(components)
            }

            fn is_zero(&self) -> bool {
                todo!()
            }

            fn set_zero(&mut self) {
                *self = Zero::zero();
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

        impl<T, const LHS_ROWS: usize, const LHS_COLUMNS: usize, const RHS_COLUMN: usize> Mul<Matrix<T, LHS_COLUMNS, RHS_COLUMN>> for Matrix<T, LHS_ROWS, LHS_COLUMNS>
        where
            T: Num + Clone + Copy + AddAssign,
        {
            type Output = Matrix::<T, LHS_ROWS, RHS_COLUMN>;

            fn mul(self, rhs: Matrix<T, LHS_COLUMNS, RHS_COLUMN>) -> Matrix::<T, LHS_ROWS, RHS_COLUMN> {
                let mut matrix = Matrix::<T, LHS_ROWS, RHS_COLUMN>::zero();

                for lhs_row in 0..LHS_ROWS {
                    for rhs_column in 0..RHS_COLUMN {
                        for lhs_column in 0..LHS_COLUMNS {
                            matrix[lhs_row][rhs_column] += self[lhs_row][lhs_column] * rhs[lhs_column][rhs_column];
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
    }

    #[cfg(test)]
    mod matrix_tests {

        use num::{One, Zero};

        use crate::rusty_maths::{
            matrix::Matrix,
            traits::{Fill, Identity, Transpose},
        };

        use super::traits::Array2D;

        impl<T, const ROWS: usize, const COLUMNS: usize> std::fmt::Debug for Matrix<T, ROWS, COLUMNS>
        where
            T: Clone + Copy,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "")
            }
        }

        #[test]
        fn index() {
            let matrix = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

            for y in 0..matrix.rows() {
                for x in 0..matrix.columns() {
                    assert_eq!(matrix[y][x], x + (y * matrix.columns()) + 1)
                }
            }
        }

        #[test]
        fn init_zero() {
            let matrix = Matrix::<usize, 3, 3>::zero();
            for y in 0..matrix.rows() {
                for x in 0..matrix.columns() {
                    assert_eq!(matrix[y][x], 0);
                }
            }
        }

        #[test]
        fn add() {
            let lhs = Matrix::<usize, 3, 3>::fill(1);
            let rhs = Matrix::<usize, 3, 3>::fill(1);

            let add_matrix = lhs + rhs;

            for y in 0..add_matrix.rows() {
                for x in 0..add_matrix.columns() {
                    assert_eq!(add_matrix[y][x], 2);
                }
            }
        }

        #[test]
        fn add_assign() {
            let rhs = Matrix::<usize, 3, 3>::fill(1);
            let mut add_assign_matrix = Matrix::<usize, 3, 3>::fill(1);

            add_assign_matrix += rhs;

            for y in 0..add_assign_matrix.rows() {
                for x in 0..add_assign_matrix.columns() {
                    assert_eq!(add_assign_matrix[y][x], 2);
                }
            }
        }

        #[test]
        fn sub() {
            let lhs = Matrix::<usize, 3, 3>::fill(1);
            let rhs = Matrix::<usize, 3, 3>::fill(1);

            let add_matrix = lhs - rhs;

            for y in 0..add_matrix.rows() {
                for x in 0..add_matrix.columns() {
                    assert_eq!(add_matrix[y][x], 0);
                }
            }
        }

        #[test]
        fn sub_assign() {
            let _rhs = Matrix::<usize, 3, 3>::fill(1);
            let mut sub_assign_matrix = Matrix::<usize, 3, 3>::fill(3);

            sub_assign_matrix -= _rhs;

            for y in 0..sub_assign_matrix.rows() {
                for x in 0..sub_assign_matrix.columns() {
                    assert_eq!(sub_assign_matrix[y][x], 2);
                }
            }
        }

        #[test]
        fn matrix_mul() {
            let matrix = Matrix::new([
                [0, -1],
                [1, 0]
            ]);

            let result = matrix.clone() * matrix;

            assert_eq!(result.components(), [
                [-1, 0],
                [0, -1]
            ]);

        }

        #[test]
        fn scalar_mul() {
            let _lhs = Matrix::<usize, 3, 3>::fill(5);
            let _rhs = 2;

            let add_matrix = _lhs * _rhs;

            for y in 0..add_matrix.rows() {
                for x in 0..add_matrix.columns() {
                    assert_eq!(add_matrix[y][x], 10);
                }
            }
        }

        #[test]
        fn scalar_mul_assign() {
            let mut _lhs = Matrix::<usize, 3, 3>::fill(5);
            let _rhs = 2;

            _lhs *= _rhs;

            for y in 0.._lhs.rows() {
                for x in 0.._lhs.columns() {
                    assert_eq!(_lhs[y][x], 10);
                }
            }
        }

        #[test]
        fn scalar_div() {
            let _lhs = Matrix::<usize, 3, 3>::fill(10);
            let _rhs = 2;

            let add_matrix = _lhs / _rhs;

            for y in 0..add_matrix.rows() {
                for x in 0..add_matrix.columns() {
                    assert_eq!(add_matrix[y][x], 5);
                }
            }
        }

        #[test]
        fn scalar_div_assign() {
            let mut _lhs = Matrix::<usize, 3, 3>::fill(10);
            let _rhs = 2;

            _lhs /= _rhs;

            for y in 0.._lhs.rows() {
                for x in 0.._lhs.columns() {
                    assert_eq!(_lhs[y][x], 5);
                }
            }
        }

        #[test]
        fn partial_eq() {
            let mut _lhs = Matrix::<usize, 3, 3>::fill(10);
            let _rhs = Matrix::<usize, 3, 3>::fill(5);

            assert_ne!(_lhs, _rhs);
            assert_eq!(_lhs, _lhs);
        }

        #[test]
        fn row() {
            let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

            assert_eq!(matrix.row(0), [1, 2, 3]);
            assert_eq!(matrix.row(1), [4, 5, 6]);
        }

        #[test]
        fn column() {
            let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

            assert_eq!(matrix.column(0), [1, 4]);
            assert_eq!(matrix.column(1), [2, 5]);
            assert_eq!(matrix.column(2), [3, 6]);
        }

        #[test]
        fn count() {
            let matrix = Matrix::<usize, 4, 4>::fill(0);
            assert_eq!(matrix.count(), 16);
        }

        #[test]
        fn identity() {
            let matrix = Matrix::<usize, 3, 3>::identity();

            assert_eq!(matrix.components(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        }

        #[test]
        fn transpose() {
            let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
            let transposed = matrix.transpose();

            assert_eq!(transposed.components(), [[1, 4], [2, 5], [3, 6],]);
        }
    }
}
