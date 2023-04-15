pub mod rusty_maths {
    use self::matrix::Matrix;

    pub mod matrix {

        use num::{one, zero, Num, One, Zero};
        use std::{
            ops::Add, ops::AddAssign, ops::Div, ops::DivAssign, ops::Index, ops::IndexMut,
            ops::Mul, ops::MulAssign, ops::Sub, ops::SubAssign,
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
                let matrix = Matrix::new(components);

                matrix
            }

            pub fn ones() -> Self {
                let components: [[T; COLUMNS]; ROWS] = [[one::<T>(); COLUMNS]; ROWS];
                Matrix::new(components)
            }

            pub fn fill(value: T) -> Self {
                let mut components: [[T; COLUMNS]; ROWS] = [[zero::<T>(); COLUMNS]; ROWS];

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        components[y][x] = value;
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

                return output;
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

            fn add(self, rhs: Matrix<T, ROWS, COLUMNS>) -> Self::Output {
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

            fn sub(self, rhs: Matrix<T, ROWS, COLUMNS>) -> Self::Output {
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

        impl<T, const ROWS: usize, const COLUMNS: usize> Mul for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn mul(self, rhs: Matrix<T, ROWS, COLUMNS>) -> Self::Output {
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        matrix.components[y][x] = self.components[y][x] * rhs.components[y][x];
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

        impl<T, const ROWS: usize, const COLUMNS: usize> Div for Matrix<T, ROWS, COLUMNS>
        where
            T: Num + Clone + Copy,
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn div(self, rhs: Matrix<T, ROWS, COLUMNS>) -> Self::Output {
                let mut matrix = Matrix::<T, ROWS, COLUMNS>::zeros();

                for y in 0..ROWS {
                    for x in 0..COLUMNS {
                        matrix.components[y][x] = self.components[y][x] / rhs.components[y][x];
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
    }

    fn dot_product<T, const ROWS: usize, const COLUMNS: usize>(
        lhs: Matrix<T, ROWS, COLUMNS>,
        rhs: Matrix<T, ROWS, COLUMNS>,
    ) -> Matrix<T, ROWS, COLUMNS>
    where
        T: Clone + Copy + num::One + num::Zero,
    {
        todo!();
    }
}

#[cfg(test)]
mod tests {

    use crate::rusty_maths::matrix::Matrix;

    #[test]
    fn matrix_index() {
        let matrix = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

        for y in 0..matrix.rows() {
            for x in 0..matrix.columns() {
                assert_eq!(matrix[y][x], x + (y * matrix.columns()) + 1)
            }
        }
    }

    #[test]
    fn matrix_zero() {
        let matrix = Matrix::<usize, 3, 3>::zeros();
        for y in 0..matrix.rows() {
            for x in 0..matrix.columns() {
                assert_eq!(matrix[y][x], 0);
            }
        }
    }

    #[test]
    fn matrix_one() {
        let matrix = Matrix::<usize, 3, 3>::ones();
        for y in 0..matrix.rows() {
            for x in 0..matrix.columns() {
                assert_eq!(matrix[y][x], 1);
            }
        }
    }

    #[test]
    fn matrix_add() {
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
    fn matrix_add_assign() {
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
    fn matrix_sub() {
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
    fn matrix_sub_assign() {
        let rhs = Matrix::<usize, 3, 3>::ones();
        let mut sub_assign_matrix = Matrix::<usize, 3, 3>::fill(3);

        sub_assign_matrix -= rhs;

        for y in 0..sub_assign_matrix.rows() {
            for x in 0..sub_assign_matrix.columns() {
                assert_eq!(sub_assign_matrix[y][x], 2);
            }
        }
    }

    #[test]
    fn matrix_mul() {
        let lhs = Matrix::<usize, 3, 3>::fill(5);
        let rhs = Matrix::<usize, 3, 3>::fill(2);

        let add_matrix = lhs * rhs;

        for y in 0..add_matrix.rows() {
            for x in 0..add_matrix.columns() {
                assert_eq!(add_matrix[y][x], 10);
            }
        }
    }

    #[test]
    fn matrix_mul_assign() {
        let rhs = Matrix::<usize, 3, 3>::fill(2);
        let mut mul_assign_matrix = Matrix::<usize, 3, 3>::fill(5);

        mul_assign_matrix *= rhs;

        for y in 0..mul_assign_matrix.rows() {
            for x in 0..mul_assign_matrix.columns() {
                assert_eq!(mul_assign_matrix[y][x], 10);
            }
        }
    }

    #[test]
    fn matrix_div() {
        let lhs = Matrix::<usize, 3, 3>::fill(10);
        let rhs = Matrix::<usize, 3, 3>::fill(2);

        let add_matrix = lhs / rhs;

        for y in 0..add_matrix.rows() {
            for x in 0..add_matrix.columns() {
                assert_eq!(add_matrix[y][x], 5);
            }
        }
    }

    #[test]
    fn matrix_div_assign() {
        let rhs = Matrix::<usize, 3, 3>::fill(2);
        let mut mul_assign_matrix = Matrix::<usize, 3, 3>::fill(10);

        mul_assign_matrix /= rhs;

        for y in 0..mul_assign_matrix.rows() {
            for x in 0..mul_assign_matrix.columns() {
                assert_eq!(mul_assign_matrix[y][x], 5);
            }
        }
    }

    #[test]
    fn matrix_row() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(*matrix.row(0), [1, 2, 3]);
        assert_eq!(*matrix.row(1), [4, 5, 6]);
    }

    #[test]
    fn matrix_columns() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(matrix.column(0), [1, 4]);
        assert_eq!(matrix.column(1), [2, 5]);
        assert_eq!(matrix.column(2), [3, 6]);
    }

    #[test]
    fn matrix_unit() {
        let matrix = Matrix::<usize, 3, 3>::unit();

        assert_eq!(matrix.components(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }

    #[test]
    fn matrix_transpose() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        let transposed = matrix.transpose();

        assert_eq!(transposed.components(), [[1, 4], [2, 5], [3, 6],]);
    }
}
