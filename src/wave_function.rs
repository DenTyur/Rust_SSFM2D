use crate::parameters;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::Xspace;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

pub struct WaveFunction {
    pub psi: Array<Complex<f64>, Ix2>,
}

impl WaveFunction {
    pub fn oscillator2d(x: &Xspace) -> Self {
        let mut psi: Array<Complex<f64>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let j = Complex::I;
        psi.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, x_i)| {
                psi_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(psi_elem, y_j)| {
                        *psi_elem = (-0.5 * (x_i.powi(2) + y_j.powi(2))).exp() + 0. * j;
                    })
            });
        Self { psi: psi }
    }

    pub fn norm(&self, dx: &Vec<f64>) -> f64 {
        //Возвращает норму волновой функции

        let mut volume: f64 = 1.;
        dx.iter().for_each(|dx| volume *= dx);
        f64::sqrt(
            self.psi
                .mapv(|a| (a.re.powi(2) + a.im.powi(2)))
                .sum_axis(Axis(0))
                .sum()
                * volume,
        )
    }

    pub fn normalization_by_1(&mut self, dx: &Vec<f64>) {
        // Нормирует волновую функцию на 1
        //
        let norm: f64 = self.norm(dx);
        let j = Complex::I;
        self.psi *= (1. + 0. * j) / norm;
    }

    pub fn save(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.psi.write_npy(writer)?;
        Ok(())
    }
}
