use core::panic;
use num::{one, traits::Pow, zero, Num, Zero};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::{traits::Fillable, Matrix};

pub type Vector2 = Vector<f32, 2>;
pub type Vector3 = Vector<f32, 3>;
pub type Vector4 = Vector<f32, 4>;

pub type Vector2Int = Vector<i32, 2>;
pub type Vector3Int = Vector<i32, 3>;
pub type Vector4Int = Vector<i32, 4>;

#[derive(Copy, Clone, Debug)]
pub struct Vector<ComponentType, const SIZE: usize> {
    pub components: [ComponentType; SIZE],
}

impl<ComponentType, const SIZE: usize> Vector<ComponentType, SIZE> {
    pub fn new(components: [ComponentType; SIZE]) -> Self {
        if SIZE == 0 {
            panic!("Vector size cannot be zero.")
        }
        Vector::<ComponentType, SIZE> { components }
    }

    pub fn count(&self) -> usize {
        self.components.len()
    }

    pub fn dot(lhs: Vector<ComponentType, SIZE>, rhs: Vector<ComponentType, SIZE>) -> ComponentType
    where
        ComponentType: Zero + Mul<Output = ComponentType> + Copy,
    {
        let mut result: ComponentType = zero();

        for i in 0..SIZE {
            result = result + lhs.components[i] * rhs.components[i];
        }

        result
    }

    pub fn unit_bases(index: usize) -> Self
    where
        ComponentType: Num + Copy,
    {
        let mut components = [zero(); SIZE];
        components[index] = one();

        Vector::<ComponentType, SIZE>::new(components)
    }

    pub fn magnitude(&self) -> ComponentType
    where
        ComponentType: Num + Copy + Pow<f32, Output = ComponentType>,
    {
        let mut square_sum: ComponentType = zero();

        for i in 0..SIZE {
            square_sum = square_sum + self.components[i] * self.components[i];
        }

        square_sum.pow(0.5)
    }

    pub fn to_unit(self) -> Self
    where
        ComponentType: Num + Copy + Pow<f32, Output = ComponentType>,
    {
        self / self.magnitude()
    }

    pub fn axpy(
        a: ComponentType,
        x: Vector<ComponentType, SIZE>,
        y: Vector<ComponentType, SIZE>,
    ) -> Self
    where
        Vector<ComponentType, SIZE>: Mul<ComponentType, Output = Vector<ComponentType, SIZE>>
            + Add<Vector<ComponentType, SIZE>, Output = Vector<ComponentType, SIZE>>,
    {
        y + (x * a)
    }

    pub fn as_column_vector(self) -> Matrix<ComponentType, SIZE, 1>
    where
        ComponentType: Clone + Copy + Zero,
        Matrix<ComponentType, SIZE, 1>: Fillable<ComponentType>,
    {
        let mut result = Matrix::<ComponentType, SIZE, 1>::fill(ComponentType::zero());
        for i in 0..SIZE {
            result.components[i][0] = self.components[i];
        }

        result
    }
}

impl<ComponentType, const SIZE: usize> Add<Vector<ComponentType, SIZE>>
    for Vector<ComponentType, SIZE>
where
    ComponentType: Add<Output = ComponentType> + Zero + Copy,
{
    type Output = Vector<ComponentType, SIZE>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut components = [zero::<ComponentType>(); SIZE];

        for (i, component) in components.iter_mut().enumerate().take(SIZE) {
            *component = self.components[i] + rhs.components[i];
        }

        Vector::<ComponentType, SIZE> { components }
    }
}

impl<ComponentType, const SIZE: usize> AddAssign for Vector<ComponentType, SIZE>
where
    Vector<ComponentType, SIZE>: Add<Output = Vector<ComponentType, SIZE>> + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        self.components = (self.clone() + rhs).components
    }
}

impl<ComponentType, const SIZE: usize> Sub<Vector<ComponentType, SIZE>>
    for Vector<ComponentType, SIZE>
where
    ComponentType: Sub<Output = ComponentType> + Zero + Copy,
{
    type Output = Vector<ComponentType, SIZE>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut components = [zero::<ComponentType>(); SIZE];

        for (i, item) in components.iter_mut().enumerate().take(SIZE) {
            *item = self.components[i] - rhs.components[i];
        }

        Vector::<ComponentType, SIZE>::new(components)
    }
}

impl<ComponentType, const SIZE: usize> SubAssign<Vector<ComponentType, SIZE>>
    for Vector<ComponentType, SIZE>
where
    Vector<ComponentType, SIZE>: Sub<Output = Vector<ComponentType, SIZE>> + Clone,
{
    fn sub_assign(&mut self, rhs: Vector<ComponentType, SIZE>) {
        self.components = (self.clone() - rhs).components
    }
}

impl<ComponentType, const SIZE: usize> Mul<ComponentType> for Vector<ComponentType, SIZE>
where
    ComponentType: Mul<Output = ComponentType> + Zero + Copy,
{
    type Output = Vector<ComponentType, SIZE>;

    fn mul(self, rhs: ComponentType) -> Self::Output {
        let mut components = [zero::<ComponentType>(); SIZE];

        for (i, item) in components.iter_mut().enumerate().take(SIZE) {
            *item = self.components[i] * rhs;
        }

        Vector::<ComponentType, SIZE>::new(components)
    }
}

impl<ComponentType, const SIZE: usize> MulAssign<ComponentType> for Vector<ComponentType, SIZE>
where
    ComponentType: Mul<Output = ComponentType> + Zero + Copy,
{
    fn mul_assign(&mut self, rhs: ComponentType) {
        self.components = (*self * rhs).components
    }
}

impl<ComponentType, const SIZE: usize> Div<ComponentType> for Vector<ComponentType, SIZE>
where
    ComponentType: Div<Output = ComponentType> + Zero + Copy,
{
    type Output = Vector<ComponentType, SIZE>;

    fn div(self, rhs: ComponentType) -> Self::Output {
        let mut components = [zero::<ComponentType>(); SIZE];

        for (i, item) in components.iter_mut().enumerate().take(SIZE) {
            *item = self.components[i] / rhs;
        }

        Vector::<ComponentType, SIZE>::new(components)
    }
}

impl<ComponentType, const SIZE: usize> DivAssign<ComponentType> for Vector<ComponentType, SIZE>
where
    ComponentType: Div<Output = ComponentType> + Zero + Copy,
{
    fn div_assign(&mut self, rhs: ComponentType) {
        self.components = (*self / rhs).components
    }
}

impl<ComponentType, const SIZE: usize> PartialEq for Vector<ComponentType, SIZE>
where
    ComponentType: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components
    }
}

impl<ComponentType, const SIZE: usize> Index<usize> for Vector<ComponentType, SIZE> {
    type Output = ComponentType;

    fn index(&self, index: usize) -> &Self::Output {
        &self.components[index]
    }
}

impl<ComponentType, const SIZE: usize> IndexMut<usize> for Vector<ComponentType, SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut ComponentType {
        &mut self.components[index]
    }
}
