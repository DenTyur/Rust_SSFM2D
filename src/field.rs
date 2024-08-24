use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;

use crate::parameters;
use parameters::Xspace;

pub struct Field2d {
    pub e0: f64,
    pub omega: f64,
    pub x_envelop: f64,
}

impl Field2d {
    pub fn new(e0: f64, omega: f64, x_envelop: f64) -> Self {
        Self {
            e0,
            omega,
            x_envelop,
        }
    }
    pub fn electric_field_time_dependence(&self, t: f64) -> f64 {
        // Возвращает электрическое поле в момент времени t вдоль
        // каждой из пространственных осей x0, x1 и т.д.: массив размерности dim.
        // Каждый элемент этого массива содержит электрическое поле
        // в момент времени t вдоль соответствующей оси.
        // Например, E0 = electric_fielf(2.)[0] - электрическое
        // поле в момент времени t=2 вдоль оси x0.

        let mut e: f64 = 0.;

        if PI / self.omega - t > 0. {
            e = -self.e0 * f64::sin(self.omega * t).powi(2);
        }
        e
    }

    pub fn field_x_envelop(&self, x: f64) -> f64 {
        // Пространственная огибающая электрического поля вдоль каждой из осей.
        f64::cos(PI / 2. * x / self.x_envelop).powi(2)
    }

    pub fn integrated_field_x_envelop(&self, x: f64) -> f64 {
        0.5 * x + 0.25 * self.x_envelop * 2. / PI * f64::sin(PI * x / self.x_envelop)
    }

    pub fn electric_field(&self, t: f64, x: f64) -> [f64; 2] {
        // Электрическое поле вдоль каждой пространственной оси в момент времени t.
        [
            self.electric_field_time_dependence(t) * self.field_x_envelop(x),
            self.electric_field_time_dependence(t) * self.field_x_envelop(x),
        ]
    }

    pub fn potential(&self, t: f64, x: &Xspace) -> [Array<f64, Ix1>; 2] {
        // Потенциал электрического поля вдоль каждой из осей.
        // В рассматриваемом случае оси независимы (2 одномерных электрона).
        // Поэтому можно интегрировать в 1D.

        let mut env: Array<f64, Ix1> = x.grid[0].clone();
        env.par_iter_mut().for_each(|elem| {
            if *elem < -self.x_envelop {
                *elem = self.integrated_field_x_envelop(-self.x_envelop);
            } else if *elem > self.x_envelop {
                *elem = self.integrated_field_x_envelop(self.x_envelop);
            } else {
                *elem = self.integrated_field_x_envelop(*elem);
            }
        });
        [
            -self.electric_field_time_dependence(t) * env.clone(),
            -self.electric_field_time_dependence(t) * env,
        ]
    }

    pub fn save_potential(&self, path: &str, x: &Xspace, t: f64) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.potential(t, x)[0].write_npy(writer)?;
        Ok(())
    }
}
