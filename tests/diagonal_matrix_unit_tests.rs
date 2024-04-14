#[cfg(test)]
mod diagonal_matrix_tests {
    use rusty_math::{traits::{Grid2D, Identity}, DiagonalMatrix, Matrix3x3};

    #[test]
    pub fn new() {
        let diagonals = [1, 2, 3];
        let matrix = DiagonalMatrix::new(diagonals);
        assert_eq!(matrix.diagonal_components, diagonals)
    }

    #[test]
    pub fn clone() {
        let diagonals = [1, 2, 3];
        let matrix = DiagonalMatrix::new(diagonals);
        let clone = matrix.clone();

        assert_eq!(clone.diagonal_components, diagonals);
    }

    #[test]
    pub fn copy() {
        let diagonals = [1, 2, 3];
        let matrix = DiagonalMatrix::new(diagonals);
        let copy = matrix;

        assert_eq!(copy.diagonal_components, diagonals);
    }

    #[test]
    fn columns() {
        let matrix = DiagonalMatrix::new([1, 2, 3]);

        assert_eq!(matrix.columns(), 3);
    }

    #[test]
    fn rows() {
        let matrix = DiagonalMatrix::new([1, 2, 3]);

        assert_eq!(matrix.rows(), 3);
    }

    #[test]
    fn components() {
        let diagonals = [1, 2, 3];
        let matrix = DiagonalMatrix::new(diagonals);

        assert_eq!(matrix.diagonal_components, diagonals);
    }

    #[test]
    fn matrix_mul() {
        let diagonal_matrix = DiagonalMatrix::new([2, 5, 4]);
        let matrix = Matrix3x3::identity();
        let result = diagonal_matrix * matrix;

        assert_eq!(result[0], [2, 0, 0]);
        assert_eq!(result[1], [0, 5, 0]);
        assert_eq!(result[2], [0, 0, 4]);
    }

    #[test]
    fn diagonal_matrix_mul() {
        let lhs = DiagonalMatrix::new([2, 4, 6]);
        let rhs = DiagonalMatrix::new([2, 2, 2]);
        let result = lhs * rhs;

        assert_eq!(result[0], 4);
        assert_eq!(result[1], 8);
        assert_eq!(result[2], 12);
    }
}