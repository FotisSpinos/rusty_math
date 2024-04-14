#[cfg(test)]
mod diagonal_matrix_tests {
    use rusty_math::{traits::Grid2D, DiagonalMatrix};

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
}