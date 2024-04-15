#[cfg(test)]
mod vector_tests {
    use rusty_math::{Vector2Int, Vector2, Vector3, Vector, traits::Grid2D};

    #[test]
    fn new() {
        let vector = Vector::new([1, 2, 3]);

        assert_eq!(vector.components, [1, 2, 3]);
    }

    #[test]
    fn len() {
        let components = [1, 2, 3];
        let vector = Vector::new([1, 2, 3]);

        assert_eq!(vector.count(), components.len());
    }

    #[test]
    fn dot() {
        let _lhs = Vector2Int::new([10, 5]);
        let _rhs = Vector2Int::new([5, 10]);

        let result = Vector2Int::dot(_lhs, _rhs);
        assert_eq!(result, 100);
    }

    #[test]
    fn unit_bases() {
        let vector = Vector::<usize, 3>::unit_bases(1);
        assert_eq!(vector.components, [0,1,0]);
    }

    #[test]
    fn magnitude() {
        let vector = Vector2::new([4.0, 3.0]);
        let magnitude = vector.magnitude();

        assert_eq!(magnitude, 5.0);
    }

    #[test]
    fn to_unit() {
        let vector = Vector2::new([1.0, 1.0]);
        let unit_vector = vector.to_unit();

        assert_eq!(unit_vector.components, [1.0 / f32::sqrt(2.0), 1.0 / f32::sqrt(2.0)]);
    }

    #[test]
    fn axpy() {
        let a = 2;
        let x = Vector2Int::new([1, 1]);
        let y = Vector2Int::new([5, 5]);

        let result = Vector2Int::axpy(a, x, y);
        assert_eq!(result.components, [7, 7]);
    }

    #[test]
    fn add() {
        let _lhs = Vector2::new([2.0, 3.0]);
        let _rhs = Vector2::new([3.0, 2.0]);
        let result = _lhs + _rhs;

        assert_eq!(result.components, [5.0, 5.0]);
    }

    #[test]
    fn add_assign() {
        let mut vector = Vector2::new([2.0, 3.0]);
        vector += Vector2::new([3.0, 2.0]);

        assert_eq!(vector.components, [5.0, 5.0]);
    }

    #[test]
    fn sub() {
        let _lhs = Vector2::new([2.0, 3.0]);
        let _rhs = Vector2::new([3.0, 2.0]);
        let result = _lhs - _rhs;

        assert_eq!(result.components, [-1.0, 1.0]);
    }

    #[test]
    fn sub_assign() {
        let mut vector = Vector2::new([2.0, 3.0]);
        vector -= Vector2::new([3.0, 2.0]);

        assert_eq!(vector.components, [-1.0, 1.0]);
    }

    #[test]
    fn scalar_mul() {
        let vector = Vector2::new([1.0, 2.0]);
        let result = vector * 2.0;

        assert_eq!(result.components, [2.0, 4.0]);
    }

    #[test]
    fn scalar_mul_assign() {
        let mut vector = Vector2::new([1.0, 2.0]);
        vector *= 2.0;

        assert_eq!(vector.components, [2.0, 4.0]);
    }

    #[test]
    fn scalar_div() {
        let vector = Vector2::new([2.0, 4.0]);
        let result = vector / 2.0;

        assert_eq!(result.components, [1.0, 2.0]);
    }

    #[test]
    fn scalar_div_assign() {
        let mut vector = Vector2::new([2.0, 4.0]);
        vector /= 2.0;

        assert_eq!(vector.components, [1.0, 2.0]);
    }

    #[test]
    fn index() {
        let vector = Vector3::new([1.0, 2.0, 3.0]);
        assert_eq!(vector[0], 1.0);
        assert_eq!(vector[1], 2.0);
        assert_eq!(vector[2], 3.0);
    }

    #[test]
    fn index_mut() {
        let mut vector = Vector3::new([1.0, 2.0, 3.0]);
        vector[0] = 0.0;
        vector[1] = 0.0;
        vector[2] = 0.0;

        assert_eq!(vector[0], 0.0);
        assert_eq!(vector[1], 0.0);
        assert_eq!(vector[2], 0.0);
    }

    #[test]
    fn as_column_vector() {
        let vector = Vector3::new([1.0, 2.0, 3.0]);
        let matrix = vector.as_column_vector();
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.columns(), 1);

        assert_eq!(matrix.components[0][0], 1.0);
        assert_eq!(matrix.components[1][0], 2.0);
        assert_eq!(matrix.components[2][0], 3.0);
    }

    #[test]
    #[should_panic]
    fn zero_size() {
        Vector::<usize, 0>::new([]);
    }
}
