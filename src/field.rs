use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;

use crate::parameters;
use parameters::{Point2e1D, Xspace};

pub struct Field1D {
    pub amplitude: f64,
    pub omega: f64,
    pub x_envelop: f64,
}

impl Field1D {
    pub fn new(amplitude: f64, omega: f64, x_envelop: f64) -> Self {
        Self {
            amplitude,
            omega,
            x_envelop,
        }
    }

    // Временная зависимость электрического поля
    pub fn electric_field_time_dependence(&self, t: f64) -> f64 {
        let mut electric_field: f64 = 0.;

        if PI / self.omega - t > 0. {
            electric_field = -self.amplitude * f64::sin(self.omega * t).powi(2);
        }
        electric_field
    }

    // Пространственная огибающая электрического поля
    pub fn field_x_envelop(&self, x: f64) -> f64 {
        f64::cos(PI / 2. * x / self.x_envelop).powi(2)
    }

    // Проинтегрированная пространственная огибающая электрического поля
    pub fn integrated_field_x_envelop(&self, x: f64) -> f64 {
        0.5 * x + 0.25 * self.x_envelop * 2. / PI * f64::sin(PI * x / self.x_envelop)
    }

    // Электрическое поле в точке x в момент t
    pub fn electric_field(&self, t: f64, x: f64) -> f64 {
        self.electric_field_time_dependence(t) * self.field_x_envelop(x)
    }

    // Потенциал электрического поля координатный массив в момент времени t
    pub fn potential_as_array(&self, t: f64, x: &Array<f64, Ix1>) -> Array<f64, Ix1> {
        // phi(t, x) = -E(t) * integral(field_x_envelop(x))dx
        let time_part: f64 = self.electric_field_time_dependence(t);
        let mut space_part: Array<f64, Ix1> = x.clone();

        // Электрическое поле отлично от нуля в области пространства:
        // -x_envelop < x < x_envelop (*)
        // В этой области пространства производная потенциала этого поля отлична
        // от нуля. За пределами этой области потенциал -- константа, которая равна
        // потенциалу на границах области (*)
        space_part.par_iter_mut().for_each(|elem| match *elem {
            x if x < -self.x_envelop => *elem = self.integrated_field_x_envelop(-self.x_envelop),
            x if x > self.x_envelop => *elem = self.integrated_field_x_envelop(-self.x_envelop),
            _ => *elem = self.integrated_field_x_envelop(*elem),
        });
        -time_part * space_part
    }
}

// ==================================
// pub struct Field2e1D {
//     pub field1d: Field1D,
// }
//
// impl Field2e1D {
//     pub fn new(field1d: Field1D) -> Self {
//         Self { field1d }
//     }
//     pub fn potential(&self, t: f64, point: Point2e1D) -> f64 {
//         self.field1d.potential(t, point.x1) + self.field1d.potential(t, point.x2)
//     }
// }
