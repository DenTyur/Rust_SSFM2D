use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use rayon::prelude::*;
use std::time::Instant;

pub fn x_evol_half(psi: &mut Array2<Complex<f64>>, u: &Array2<Complex<f64>>, dt: f64) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;
    psi.axis_iter_mut(Axis(0))
        .zip(u.axis_iter(Axis(0)))
        .par_bridge()
        .for_each(|(mut psi_row, u_row)| {
            psi_row
                .iter_mut()
                .zip(u_row.iter())
                .for_each(|(psi_elem, u_elem)| {
                    *psi_elem *= (-j * dt / 2. * u_elem).exp();
                });
        });
}

pub fn x_evol(psi: &mut Array2<Complex<f64>>, u: &Array2<Complex<f64>>, dt: f64) {
    // эволюция в координатном пространстве на временной шаг
    let j = Complex::I;
    psi.axis_iter_mut(Axis(0))
        .zip(u.axis_iter(Axis(0)))
        .par_bridge()
        .for_each(|(mut psi_row, u_row)| {
            psi_row
                .iter_mut()
                .zip(u_row.iter())
                .for_each(|(psi_elem, u_elem)| {
                    *psi_elem *= (-j * dt * u_elem).exp();
                });
        });
}
pub fn p_evol(psi: &mut Array2<Complex<f64>>, px: &Array<f64, Ix1>, py: &Array<f64, Ix1>, dt: f64) {
    // эволюция в импульсном пространстве
    let j = Complex::I;
    psi.axis_iter_mut(Axis(0))
        .zip(px.iter())
        .par_bridge()
        .for_each(|(mut psi_row, px_i)| {
            psi_row
                .iter_mut()
                .zip(py.iter())
                .for_each(|(psi_elem, py_k)| {
                    *psi_elem *= (-j * dt / 2. * (px_i.powi(2) + py_k.powi(2))).exp();
                });
        });
}

pub fn demodify_psi(
    psi: &mut Array2<Complex<f64>>,
    x: &Array<f64, Ix1>,
    y: &Array<f64, Ix1>,
    px0: f64,
    py0: f64,
    dx: f64,
    dy: f64,
) {
    // демодифицирует "psi для DFT" обратно в psi
    let j = Complex::I;
    psi.axis_iter_mut(Axis(0))
        .zip(x.iter())
        .par_bridge()
        .for_each(|(mut psi_row, x_i)| {
            psi_row
                .iter_mut()
                .zip(y.iter())
                .for_each(|(psi_elem, y_k)| {
                    *psi_elem *= dx * dy / (2. * 3.14) * (-j * (px0 * x_i + py0 * y_k)).exp();
                })
        });
}

pub fn modify_psi(
    psi: &mut Array2<Complex<f64>>,
    x: &Array<f64, Ix1>,
    y: &Array<f64, Ix1>,
    px0: f64,
    py0: f64,
    dx: f64,
    dy: f64,
) {
    // модифицирует psi для DFT (в нашем сучае FFT)
    let j = Complex::I;
    psi.axis_iter_mut(Axis(0))
        .zip(x.iter())
        .par_bridge()
        .for_each(|(mut psi_row, x_i)| {
            psi_row
                .iter_mut()
                .zip(y.iter())
                .for_each(|(psi_elem, y_k)| {
                    *psi_elem *= dx * dy / (2. * 3.14) * (-j * (px0 * x_i + py0 * y_k)).exp();
                })
        });
}

pub struct FftMaker {
    pub handler0: FftHandler<f64>,
    pub handler1: FftHandler<f64>,
    pub psi_temp: Array2<Complex<f64>>,
}

impl FftMaker {
    pub fn new(n0: usize, n1: usize) -> Self {
        Self {
            handler0: FftHandler::new(n0),
            handler1: FftHandler::new(n1),
            psi_temp: Array::zeros((n0, n1)),
        }
    }

    pub fn do_fft(&mut self, psi: &mut Array2<Complex<f64>>) {
        ndfft_par(psi, &mut self.psi_temp, &mut self.handler0, 0);
        ndfft_par(&self.psi_temp, psi, &mut self.handler1, 1);
    }
    pub fn do_ifft(&mut self, psi: &mut Array2<Complex<f64>>) {
        ndifft_par(psi, &mut self.psi_temp, &mut self.handler1, 1);
        ndifft_par(&self.psi_temp, psi, &mut self.handler0, 0);
    }
}
