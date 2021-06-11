use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Debug, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        [0.0; 3].into()
    }

    pub fn one() -> Self {
        [1.0; 3].into()
    }

    pub fn len(&self) -> f64 {
        self.dot(&self).sqrt()
    }

    #[inline]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Self {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }

    #[inline]
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        self.x = f(self.x);
        self.y = f(self.y);
        self.z = f(self.z);
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn apply_pbc(&mut self, box_length: f64) {
        self.x += if self.x < 0.0 {
            box_length
        } else if self.x >= box_length {
            -box_length
        } else {
            0.0
        };
        self.y += if self.y < 0.0 {
            box_length
        } else if self.y >= box_length {
            -box_length
        } else {
            0.0
        };
        self.z += if self.z < 0.0 {
            box_length
        } else if self.z >= box_length {
            -box_length
        } else {
            0.0
        };
    }

    #[inline]
    pub fn pbc(&mut self, box_length: f64) -> Self {
        let f = |v: f64| match v {
            v if v < 0.0 => v + box_length,
            v if v >= box_length => v - box_length,
            _ => v,
        };
        self.map(f)
    }

    #[inline]
    pub fn pbc_inplace(&mut self, box_length: f64) {
        let f = |v: f64| match v {
            v if v < 0.0 => v + box_length,
            v if v >= box_length => v - box_length,
            _ => v,
        };
        self.map_inplace(f)
    }

    #[inline]
    pub fn nearest_image(&self, box_length: f64) -> Self {
        let il = 1.0 / box_length;
        let f = |v: f64| v - box_length * f64::round(v * il);
        self.map(f)
    }

    #[inline]
    pub fn nearest_image_inplace(&mut self, box_length: f64) {
        let il = 1.0 / box_length;
        let f = |v: f64| v - box_length * f64::round(v * il);
        self.map_inplace(f)
    }
}

// Conversions

impl From<[f64; 3]> for Vec3 {
    fn from(v: [f64; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }
}

impl From<&[f64; 3]> for Vec3 {
    fn from(v: &[f64; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }
}

impl From<Vec3> for [f64; 3] {
    fn from(v: Vec3) -> Self {
        [v.x, v.y, v.z]
    }
}

/* MATH */
macro_rules! forward_val_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl $imp for Vec3 {
            type Output = Vec3;

            #[inline]
            fn $method(self, other: Vec3) -> Self::Output {
                Self::new(
                    self.x.$method(other.x),
                    self.y.$method(other.y),
                    self.z.$method(other.z),
                )
            }
        }
    };
}

macro_rules! forward_ref_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, 'b> $imp<&'b Vec3> for &'a Vec3 {
            type Output = Vec3;

            #[inline]
            fn $method(self, other: &Vec3) -> Self::Output {
                self.clone().$method(other.clone())
            }
        }
    };
}

macro_rules! forward_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a> $imp<Vec3> for &'a Vec3 {
            type Output = Vec3;

            #[inline]
            fn $method(self, other: Vec3) -> Self::Output {
                self.clone().$method(other)
            }
        }
    };
}

macro_rules! forward_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a> $imp<&'a Vec3> for Vec3 {
            type Output = Vec3;

            #[inline]
            fn $method(self, other: &Vec3) -> Self::Output {
                self.$method(other.clone())
            }
        }
    };
}

macro_rules! forward_val_float_binop {
    (impl $imp:ident, $method:ident) => {
        impl $imp<Vec3> for f64 {
            type Output = Vec3;

            #[inline]
            fn $method(self, other: Vec3) -> Self::Output {
                Vec3::new(
                    self.$method(other.x),
                    self.$method(other.y),
                    self.$method(other.z),
                )
            }
        }
    };
}

macro_rules! forward_float_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl $imp<f64> for Vec3 {
            type Output = Vec3;

            #[inline]
            fn $method(self, other: f64) -> Self::Output {
                Vec3::new(
                    self.x.$method(other),
                    self.y.$method(other),
                    self.z.$method(other),
                )
            }
        }
    };
}

macro_rules! forward_all_binop {
    (impl $imp:ident, $method:ident) => {
        forward_val_val_binop!(impl $imp, $method);
        forward_ref_ref_binop!(impl $imp, $method);
        forward_ref_val_binop!(impl $imp, $method);
        forward_val_ref_binop!(impl $imp, $method);
        forward_val_float_binop!(impl $imp, $method);
        forward_float_val_binop!(impl $imp, $method);
    };
}

forward_all_binop!(impl Add, add);
forward_all_binop!(impl Sub, sub);
forward_all_binop!(impl Mul, mul);
forward_all_binop!(impl Div, div);

// Assign Operators: +=, -=, /=, *=

macro_rules! op_assign {
    (impl $imp:ident, $method:ident) => {
        impl $imp for Vec3 {
            #[inline]
            fn $method(&mut self, other: Self) {
                self.x.$method(other.x);
                self.y.$method(other.y);
                self.z.$method(other.z);
            }
        }
    };
}

op_assign!(impl AddAssign, add_assign);
op_assign!(impl SubAssign, sub_assign);
op_assign!(impl MulAssign, mul_assign);
op_assign!(impl DivAssign, div_assign);

// Negation

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

// Display

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
}

// Indexing

impl Index<usize> for Vec3 {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds."),
        }
    }
}
