use crate::field;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use field::Field2e1D;
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use parameters::*;
use potentials::AtomicPotential;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;
use wave_function::WaveFunction;

pub fn x_evol_half(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field2e1d: &Field2e1D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let ts = Instant::now();
    psi.psi
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|(index, psi_elem)| {
            *psi_elem *= (-j
                * 0.5
                * t.dt
                * (atomic_potential.potential[index]
                    - field2e1d.electric_field_potential(t.current, x.point(index))))
            .exp();
        });
    println!("TIME_evol_half = {:?}", ts.elapsed());
}

pub fn x_evol(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field2e1d: &Field2e1D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let ts = Instant::now();
    psi.psi
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|(index, psi_elem)| {
            *psi_elem *= (-j
                * t.dt
                * (atomic_potential.potential[index]
                    - field2e1d.electric_field_potential(t.current, x.point(index))))
            .exp();
        });
    println!("TIME_evol= {:?}", ts.elapsed());
}

pub fn p_evol(psi: &mut WaveFunction, p: &Pspace, dt: f64) {
    // эволюция в импульсном пространстве
    let j = Complex::I;

    psi.psi
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|(index, psi_elem)| {
            *psi_elem *= (-j * 0.5 * dt * p.point_abs_squared(index)).exp();
        });
}

pub fn demodify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // демодифицирует "psi для DFT" обратно в psi
    let j = Complex::I;
    psi.psi
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .par_bridge()
        .for_each(|(mut psi_row, x_i)| {
            psi_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(psi_elem, y_k)| {
                    *psi_elem *= (2. * PI) / (x.dx[0] * x.dx[1])
                        * (j * (p.p0[0] * x_i + p.p0[1] * y_k)).exp();
                })
        });
}

pub fn modify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // модифицирует psi для DFT (в нашем сучае FFT)
    let j = Complex::I;
    psi.psi
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .par_bridge()
        .for_each(|(mut psi_row, x_i)| {
            psi_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(psi_elem, y_k)| {
                    *psi_elem *= x.dx[0] * x.dx[1] / (2. * PI)
                        * (-j * (p.p0[0] * x_i + p.p0[1] * y_k)).exp();
                })
        });
}

pub struct FftMaker2d {
    pub handler: Vec<FftHandler<f64>>,
    pub psi_temp: Array2<Complex<f64>>,
}

impl FftMaker2d {
    pub fn new(n: &Vec<usize>) -> Self {
        Self {
            handler: Vec::from_iter(0..n.len()) // тоже костыль! как это сделать через функцию?
                .iter()
                .map(|&i| FftHandler::new(n[i]))
                .collect(),
            psi_temp: Array::zeros((n[0], n[1])),
        }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[0], 0);
    }
}
