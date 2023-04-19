pub mod rusty_maths {

    pub mod matrix {

        use num::{one, zero, Num, One, Zero};
        use std::{
            cmp::PartialEq, ops::Add, ops::AddAssign, ops::Div, ops::DivAssign, ops::Index,
            ops::IndexMut, ops::Mul, ops::MulAssign, ops::Sub, ops::SubAssign,
        };

        pub struct Matrix<T, const ROWS: usize, const COLUMNS: usize>
        where
            T: Clone,
        {
            components: [[T; COLUMNS]; ROWS],
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Matrix<T, ROWS, COLUMNS>
        where
            T: Zero + One + Clone + Copy,
        {
            pub fn new(components: [[T; COLUMNS]; ROWS]) -> Self {
                Matrix::<T, ROWS, COLUMNS> { components }
            }

            pub fn zeros() -> Self {
                let components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];
                Matrix::new(components)
            }

            pub fn ones() -> Self {
                let components: [[T; COLUMNS]; ROWS] = [[one::<T>(); COLUMNS]; ROWS];
                Matrix::new(components)
            }

            pub fn fill(value: T) -> Self {
                let mut components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];

                for rows in components.iter_mut().take(ROWS) {
                    for columns in rows.iter_mut().take(COLUMNS) {
                        *columns = value;
                    }
                }

                Matrix::new(components)
            }

            pub fn unit() -> Self {
                let mut components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];

                for row in 0..ROWS {
                    components[row][row] = one::<T>();
                }

                Matrix::new(components)
            }

            pub fn transpose(&self) -> Matrix<T, COLUMNS, ROWS> {
                let mut output = Matrix::<T, COLUMNS, ROWS>::zeros();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        output.components[x][y] = self.components[y][x];
                    }
                }

                output
            }

            pub fn row(&self, index: usize) -> &[T; COLUMNS] {
                &self.components[index]
            }

            pub fn column(&self, index: usize) -> [T; ROWS] {
                let mut output = [zero::<T>(); ROWS];

                for i in 0..ROWS {
                    output[i] = self.components[i][index];
                }

                output
            }

            pub fn rows(&self) -> usize {
                ROWS
            }

            pub fn columns(&self) -> usize {
                COLUMNS
            }

            pub fn components(&self) -> [[T; COLUMNS]; ROWS] {
                self.components
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Add for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn add(self, rhs: Self) -> Self::Output {
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

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
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

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

        impl<T, const ROWS: usize, const COLUMNS: usize> Mul<Self> for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn mul(self, rhs: Self) -> Self::Output {
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        matrix.components[y][x] = self.components[y][x] * rhs.components[y][x];
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
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        matrix.components[y][x] = self.components[y][x] * scalar;
                    }
                }

                matrix
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> MulAssign for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            fn mul_assign(&mut self, rhs: Self) {
                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        self.components[y][x] = self.components[y][x] * rhs.components[y][x];
                    }
                }
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

        impl<T, const ROWS: usize, const COLUMNS: usize> Div for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn div(self, rhs: Self) -> Self::Output {
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        matrix.components[y][x] = self.components[y][x] / rhs.components[y][x];
                    }
                }

                matrix
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> Div<T> for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn div(self, scalar: T) -> Self::Output {
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        matrix.components[y][x] = self.components[y][x] / scalar;
                    }
                }

                matrix
            }
        }

        impl<T, const ROWS: usize, const COLUMNS: usize> DivAssign for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            fn div_assign(&mut self, rhs: Self) {
                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        self.components[y][x] = self.components[y][x] / rhs.components[y][x];
                    }
                }
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

                return true;
            }
        }
    }
}

#[cfg(test)]
mod matrix_tests {

    use crate::rusty_maths::matrix::Matrix;

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
        let matrix = Matrix::<usize, 3, 3>::zeros();
        for y in 0..matrix.rows() {
            for x in 0..matrix.columns() {
                assert_eq!(matrix[y][x], 0);
            }
        }
    }

    #[test]
    fn init_one() {
        let matrix = Matrix::<usize, 3, 3>::ones();
        for y in 0..matrix.rows() {
            for x in 0..matrix.columns() {
                assert_eq!(matrix[y][x], 1);
            }
        }
    }

    #[test]
    fn add() {
        let lhs = Matrix::<usize, 3, 3>::ones();
        let rhs = Matrix::<usize, 3, 3>::ones();

        let add_matrix = lhs + rhs;

        for y in 0..add_matrix.rows() {
            for x in 0..add_matrix.columns() {
                assert_eq!(add_matrix[y][x], 2);
            }
        }
    }

    #[test]
    fn add_assign() {
        let rhs = Matrix::<usize, 3, 3>::ones();
        let mut add_assign_matrix = Matrix::<usize, 3, 3>::ones();

        add_assign_matrix += rhs;

        for y in 0..add_assign_matrix.rows() {
            for x in 0..add_assign_matrix.columns() {
                assert_eq!(add_assign_matrix[y][x], 2);
            }
        }
    }

    #[test]
    fn sub() {
        let lhs = Matrix::<usize, 3, 3>::ones();
        let rhs = Matrix::<usize, 3, 3>::ones();

        let add_matrix = lhs - rhs;

        for y in 0..add_matrix.rows() {
            for x in 0..add_matrix.columns() {
                assert_eq!(add_matrix[y][x], 0);
            }
        }
    }

    #[test]
    fn sub_assign() {
        let _rhs = Matrix::<usize, 3, 3>::ones();
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
        let _lhs = Matrix::<usize, 3, 3>::fill(5);
        let _rhs = Matrix::<usize, 3, 3>::fill(2);

        let add_matrix = _lhs * _rhs;

        for y in 0..add_matrix.rows() {
            for x in 0..add_matrix.columns() {
                assert_eq!(add_matrix[y][x], 10);
            }
        }
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
    fn matrix_mul_assign() {
        let _rhs = Matrix::<usize, 3, 3>::fill(2);
        let mut mul_assign_matrix = Matrix::<usize, 3, 3>::fill(5);

        mul_assign_matrix *= _rhs;

        for y in 0..mul_assign_matrix.rows() {
            for x in 0..mul_assign_matrix.columns() {
                assert_eq!(mul_assign_matrix[y][x], 10);
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
    fn matrix_div() {
        let _lhs = Matrix::<usize, 3, 3>::fill(10);
        let _rhs = Matrix::<usize, 3, 3>::fill(2);

        let add_matrix = _lhs / _rhs;

        for y in 0..add_matrix.rows() {
            for x in 0..add_matrix.columns() {
                assert_eq!(add_matrix[y][x], 5);
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
    fn matrix_div_assign() {
        let _rhs = Matrix::<usize, 3, 3>::fill(2);
        let mut mul_assign_matrix = Matrix::<usize, 3, 3>::fill(10);

        mul_assign_matrix /= _rhs;

        for y in 0..mul_assign_matrix.rows() {
            for x in 0..mul_assign_matrix.columns() {
                assert_eq!(mul_assign_matrix[y][x], 5);
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
    fn row() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(*matrix.row(0), [1, 2, 3]);
        assert_eq!(*matrix.row(1), [4, 5, 6]);
    }

    #[test]
    fn column() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(matrix.column(0), [1, 4]);
        assert_eq!(matrix.column(1), [2, 5]);
        assert_eq!(matrix.column(2), [3, 6]);
    }

    #[test]
    fn unit() {
        let matrix = Matrix::<usize, 3, 3>::unit();

        assert_eq!(matrix.components(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }

    #[test]
    fn transpose() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        let transposed = matrix.transpose();

        assert_eq!(transposed.components(), [[1, 4], [2, 5], [3, 6],]);
    }
}
