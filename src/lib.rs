pub mod rusty_maths {

    pub mod vectors {

        use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

        #[derive(Default, Clone, Copy)]
        pub struct Vec2<T> {
            pub x: T,
            pub y: T,
        }

        impl<T> Vec2<T> {
            pub fn new(x: T, y: T) -> Self {
                Vec2 { x, y }
            }

            pub fn dot_product(&self, other: Vec2<T>) -> T
            where
                T: Add<Output = T> + Mul<Output = T> + Copy,
            {
                (self.x * other.x) + (self.y * other.y)
            }
        }

        impl<T> Add for Vec2<T>
        where
            T: Add<Output = T>,
        {
            type Output = Vec2<T>;

            fn add(self, other: Self) -> Self::Output {
                Vec2::new(self.x + other.x, self.y + other.y)
            }
        }

        impl<T> AddAssign for Vec2<T>
        where
            T: AddAssign,
        {
            fn add_assign(&mut self, other: Self) {
                self.x += other.x;
                self.y += other.y;
            }
        }

        impl<T> Sub for Vec2<T>
        where
            T: Sub<Output = T>,
        {
            type Output = Vec2<T>;

            fn sub(self, other: Self) -> Self::Output {
                Vec2::new(self.x - other.x, self.y - other.y)
            }
        }

        impl<T> SubAssign for Vec2<T>
        where
            T: SubAssign,
        {
            fn sub_assign(&mut self, other: Self) {
                self.x -= other.x;
                self.y -= other.y;
            }
        }

        impl<T> Mul for Vec2<T>
        where
            T: Mul<Output = T>,
        {
            type Output = Vec2<T>;

            fn mul(self, other: Self) -> Self::Output {
                Vec2::new(self.x * other.x, self.y * other.y)
            }
        }

        impl<T> Mul<T> for Vec2<T>
        where
            T: Mul<Output = T> + Copy,
        {
            type Output = Vec2<T>;

            fn mul(self, scalar: T) -> Vec2<T> {
                Vec2::new(self.x * scalar, self.y * scalar)
            }
        }

        impl<T> MulAssign for Vec2<T>
        where
            T: MulAssign,
        {
            fn mul_assign(&mut self, other: Self) {
                self.x *= other.x;
                self.y *= other.y;
    }
}

impl<T> MulAssign<T> for Vec2<T> where T: MulAssign + Copy {
    fn mul_assign(&mut self, other: T) {
        self.x *= other;
        self.y *= other;
    }
}

impl<T> Div for Vec2<T> where T: Div<Output = T> {
    type Output = Vec2<T>;

    fn div(self, other: Self) -> Self::Output {
        Vec2::new(self.x / other.x, self.y / other.y)
    }
}

impl<T> Div<T> for Vec2<T> where T: Div<Output = T> + Copy {
    type Output = Vec2<T>;

    fn div(self, scalar: T) -> Vec2<T> {
        Vec2::new(self.x / scalar, self.y / scalar)
    }
}

impl<T> DivAssign for Vec2<T> where T: DivAssign {
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
    }
}

impl<T> DivAssign<T> for Vec2<T> where T: DivAssign + Copy {
    fn div_assign(&mut self, other: T) {
        self.x /= other;
        self.y /= other;
    }
}

impl<T> Neg for Vec2<T> where T : Neg<Output = T> {
    type Output = Vec2<T>;

    fn neg(self) -> Self::Output {
        Vec2::new(-self.x, -self.y)
    }
}

impl<T> PartialEq for Vec2<T> where T : PartialEq {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

}

}

#[cfg(test)]
mod tests {
    use crate::rusty_maths::vectors::Vec2;

    #[test]
    fn it_works() {
        let result = Vec2::new(1, 2) + Vec2::new(3, 4);
        assert_eq!(result.x, 4);
        assert_eq!(result.y, 6);
    }
}
