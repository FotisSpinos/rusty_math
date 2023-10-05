#[cfg(test)]
mod matrix_tests {
    use num::Zero;
    use rusty_math::{Matrix, Matrix3x3, Matrix4x4, traits::{Grid2D, Identity, Fillable, Transposable}, Vector};

    #[test]
    pub fn new() {
        let components = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::new(components);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.columns(), 3);
        assert_eq!(matrix.components, components);
    }

    #[test]
    pub fn pow() {
        let matrix = Matrix::<usize, 3, 3>::identity() * 2;
        let result = Matrix::<usize, 3, 3>::pow(matrix, 3);

        assert_eq!(result, Matrix::<usize, 3, 3>::identity() * 8);
    }

    #[test]
    pub fn clone() {
        let components = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::new(components);
        let clone = matrix.clone();

        assert_eq!(matrix, clone);
    }

    #[test]
    fn column() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(matrix.column(0), Vector::new([1, 4]));
        assert_eq!(matrix.column(1), Vector::new([2, 5]));
        assert_eq!(matrix.column(2), Vector::new([3, 6]));
    }

    #[test] fn columns() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(matrix.columns(), 3);
    }

    #[test]
    fn components() {
        let components = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::<usize, 2, 3>::new(components);

        assert_eq!(matrix.components(), components);
    }

    #[test]
    fn len() {
        let matrix = Matrix::<usize, 4, 4>::fill(0);
        assert_eq!(matrix.len(), 16);
    }

    #[test]
    fn row() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(matrix.row(0), Vector::new([1, 2, 3]));
        assert_eq!(matrix.row(1), Vector::new([4, 5, 6]));
    }

    #[test]
    fn rows() {
        let matrix = Matrix::<usize, 4, 3>::fill(0);
        assert_eq!(matrix.rows(), 4);
    }

    #[test]
    fn transpose() {
        let matrix = Matrix::<usize, 2, 3>::new([[1, 2, 3], [4, 5, 6]]);
        let transposed = matrix.transpose();

        assert_eq!(transposed.components(), [[1, 4], [2, 5], [3, 6],]);
    }

    #[test]
    fn fill() {
        let matrix = Matrix::<usize, 2, 3>::fill(5);

        assert_eq!(matrix.components(), [[5, 5, 5], [5, 5, 5]]);
    }

    #[test]
    fn identity() {
        let matrix = Matrix::<usize, 3, 3>::identity();

        assert_eq!(matrix.components(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }

    #[test]
    fn zero() {
        let matrix = Matrix::<usize, 2, 3>::zero();

        assert_eq!(matrix.components(), [[0, 0, 0], [0, 0, 0]]);
    }

    #[test]
    fn set_zero() {
        let mut matrix = Matrix::<usize, 2, 3>::fill(5);
        matrix.set_zero();

        assert_eq!(matrix.components(), [[0, 0, 0], [0, 0, 0]]);
    }

    #[test]
    fn is_zero() {
        let matrix = Matrix::<usize, 2, 3>::zero();

        assert!(matrix.is_zero());
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
        let matrix = Matrix::new([[0, -1], [1, 0]]);

        let result = matrix.clone() * matrix;

        assert_eq!(result.components(), [[-1, 0], [0, -1]]);
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
    fn index() {
        let matrix = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

        for y in 0..matrix.rows() {
            for x in 0..matrix.columns() {
                assert_eq!(matrix[y][x], x + (y * matrix.columns()) + 1)
            }
        }
    }

    #[test]
    fn index_mut() {
        let mut matrix = Matrix::<usize, 2, 2>::fill(1);

        for y in 0..matrix.rows() {
            for x in 0..matrix.columns() {
                matrix[y][x] = 0;
            }
        }

        assert_eq!(matrix.components(), [[0, 0], [0, 0]]);
    }

    #[test]
    fn partial_eq() {
        let mut _lhs = Matrix::<usize, 3, 3>::fill(10);
        let _rhs = Matrix::<usize, 3, 3>::fill(5);

        assert_ne!(_lhs, _rhs);
        assert_eq!(_lhs, _lhs);
    }

    #[test]
    fn matrix3x3() {
        let matrix = Matrix3x3::<usize>::identity();
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.columns(), 3);
    }

    #[test]
    fn matrix4x4() {
        let matrix = Matrix4x4::<usize>::identity();
        assert_eq!(matrix.rows(), 4);
        assert_eq!(matrix.columns(), 4);
    }
}
