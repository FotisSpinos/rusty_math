pub mod vector {
    use num::{one, traits::Pow, zero, Num, Zero};
    use std::ops::{Add, AddAssign, Sub, SubAssign, Div, DivAssign, Mul, MulAssign};

    pub type Vector2 = Vector<f32, 2>;
    pub type Vector3 = Vector<f32, 3>;
    pub type Vector4 = Vector<f32, 4>;

    pub type Vector2Int = Vector<i32, 2>;
    pub type Vector3Int = Vector<i32, 3>;
    pub type Vector4Int = Vector<i32, 4>;

    #[derive(Copy, Clone, Debug)]
    pub struct Vector<T, const SIZE: usize> {
        pub components: [T; SIZE],
    }

    impl<T, const SIZE: usize> Vector<T, SIZE> {
        pub fn new(components: [T; SIZE]) -> Self {
            Vector::<T, SIZE> { components }
        }

        pub fn len(&self) -> usize {
            self.components.len()
        }

        pub fn dot(lhs: Vector<T, SIZE>, rhs: Vector<T, SIZE>) -> T
        where
            T: Zero + Mul<Output = T> + Copy,
        {
            let mut result: T = zero();

            for i in 0..SIZE {
                result = result + lhs.components[i] * rhs.components[i];
            }

            result
        }

        pub fn unit_bases(index: usize) -> Self
        where
            T: Num + Copy,
        {
            let mut components = [zero(); SIZE];
            components[index] = one();

            Vector::<T, SIZE>::new(components)
        }

        pub fn magnitude(&self) -> T
        where
            T: Num + Copy + Pow<f32, Output = T>,
        {
            let mut square_sum: T = zero();

            for i in 0..SIZE {
                square_sum = square_sum + self.components[i] * self.components[i];
            }

            square_sum.pow(0.5)
        }

        pub fn unit(self) -> Self
        where
            T: Num + Copy + Pow<f32, Output = T>,
        {
            self / self.magnitude()
        }

        pub fn axpy(a: T, x: Vector<T, SIZE>, y: Vector<T, SIZE>) -> Self
        where
            Vector<T, SIZE>: Mul<T, Output = Vector<T, SIZE>>
                + Add<Vector<T, SIZE>, Output = Vector<T, SIZE>>,
        {
            y + (x * a)
        }
    }

    impl<T, const SIZE: usize> Add<Vector<T, SIZE>> for Vector<T, SIZE>
    where
        T: Add<Output = T> + Zero + Copy,
    {
        type Output = Vector<T, SIZE>;

        fn add(self, rhs: Self) -> Self::Output {
            let mut components = [zero::<T>(); SIZE];

            for i in 0..SIZE {
                components[i] = self.components[i] + rhs.components[i];
            }

            Vector::<T, SIZE> { components }
        }
    }

    impl<T, const SIZE: usize> AddAssign for Vector<T, SIZE>
    where
        Vector<T, SIZE>: Add<Output = Vector<T, SIZE>> + Clone
    {
        fn add_assign(&mut self, rhs: Self) {
            self.components = (self.clone() + rhs).components
        }
    }

    impl<T, const SIZE: usize> Sub<Vector<T, SIZE>> for Vector<T, SIZE>
    where
        T: Sub<Output = T> + Zero + Copy,
    {
        type Output = Vector<T, SIZE>;

        fn sub(self, rhs: Self) -> Self::Output {
            let mut components = [zero::<T>(); SIZE];

            for i in 0..SIZE {
                components[i] = self.components[i] - rhs.components[i];
            }

            Vector::<T, SIZE>::new(components)
        }
    }

    impl<T, const SIZE: usize> SubAssign<Vector<T, SIZE>> for Vector<T, SIZE>
    where 
        Vector<T, SIZE>: Sub<Output = Vector<T, SIZE>> + Clone
    {
        fn sub_assign(&mut self, rhs: Vector<T, SIZE>) {
            self.components = (self.clone() - rhs).components
        }
    }

    impl<T, const SIZE: usize> Mul<T> for Vector<T, SIZE>
    where
        T: Mul<Output = T> + Zero + Copy,
    {
        type Output = Vector<T, SIZE>;

        fn mul(self, rhs: T) -> Self::Output {
            let mut components = [zero::<T>(); SIZE];

            for i in 0..SIZE {
                components[i] = self.components[i] * rhs;
            }

            Vector::<T, SIZE>::new(components)
        }
    }

    impl<T, const SIZE: usize> MulAssign<T> for Vector<T, SIZE>
    where
        T: Mul<Output = T> + Zero + Copy,
    {
        fn mul_assign(&mut self, rhs: T) {
            self.components = (self.clone() * rhs).components
        }
    }

    impl<T, const SIZE: usize> Div<T> for Vector<T, SIZE>
    where
        T: Div<Output = T> + Zero + Copy,
    {
        type Output = Vector<T, SIZE>;

        fn div(self, rhs: T) -> Self::Output {
            let mut components = [zero::<T>(); SIZE];

            for i in 0..SIZE {
                components[i] = self.components[i] / rhs;
            }

            Vector::<T, SIZE>::new(components)
        }
    }

    impl<T, const SIZE: usize> DivAssign<T> for Vector<T, SIZE>
    where
        T: Div<Output = T> + Zero + Copy,
    {
        fn div_assign(&mut self, rhs: T) {
            self.components = (self.clone() / rhs).components
        }
    }

    impl<T, const SIZE: usize> PartialEq for Vector<T, SIZE>
    where
        T: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.components == other.components
        }

        fn ne(&self, other: &Self) -> bool {
            !self.eq(other)
        }
    }
}
