#[cfg(test)]
mod diagonal_matrix_tests {
    use rusty_math::DiagonalMatrix;


    #[test]
    pub fn new() {
        let diagonal_components = [1, 2, 3];
        let matrix = DiagonalMatrix::new(diagonal_components);
    }
}