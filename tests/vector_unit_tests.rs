#[cfg(test)]
mod vector_tests {
    use rusty_math::{Vector, Vector2Int, Vector2};

    #[test]
    fn new() {
        let vector = Vector::new([1, 2, 3]);

        assert_eq!(vector.components, [1, 2, 3]);
    }

    #[test]
    fn len() {
        let components = [1, 2, 3];
        let vector = Vector::new([1, 2, 3]);

        assert_eq!(vector.len(), components.len());
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
    fn dot() {
        let _lhs = Vector2Int::new([10, 5]);
        let _rhs = Vector2Int::new([5, 10]);

        let result = Vector2Int::dot(_lhs, _rhs);
        assert_eq!(result, 100);
    }

    #[test]
    fn magnitude() {
        let vector = Vector2::new([4.0, 3.0]);
        let length = vector.magnitude();

        assert_eq!(length, 5.0);
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
}
