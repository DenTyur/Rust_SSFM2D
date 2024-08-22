use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

use crate::parameters;
use parameters::Xspace;

pub struct Potentials {
    pub u: Array<Complex<f64>, Ix2>,
}

impl Potentials {
    pub fn oscillator2d(x: &Xspace) -> Self {
        // плохо написано. Лучше не создавать промежуточное u.
        let mut u: Array<Complex<f64>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let j = Complex::I;
        u.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut u_row, x_i)| {
                u_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(u_elem, y_j)| {
                        *u_elem = 0.5 * (x_i.powi(2) + y_j.powi(2)) + 0. * j;
                    })
            });
        Self { u: u }
    }

    pub fn save(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.u.write_npy(writer)?;
        Ok(())
    }
}
