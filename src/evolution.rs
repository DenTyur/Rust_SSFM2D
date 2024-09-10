use crate::field;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use field::Field1D;
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

pub fn time_step_evol(
    fft: &mut FftMaker2d,
    psi: &mut WaveFunction,
    field: &Field1D,
    u: &AtomicPotential,
    x: &Xspace,
    p: &Pspace,
    t: &mut Tspace,
) {
    modify_psi(psi, x, p);
    x_evol_half(psi, u, t, field, x);

    for _i in 0..t.n_steps - 1 {
        fft.do_fft(psi);
        // Можно оптимизировать p_evol
        p_evol(psi, p, t.dt);
        fft.do_ifft(psi);
        x_evol(psi, u, t, field, x);
        t.current += t.dt;
    }

    fft.do_fft(psi);
    p_evol(psi, p, t.dt);
    fft.do_ifft(psi);
    x_evol_half(psi, u, t, field, x);
    demodify_psi(psi, x, p);
    t.current += t.dt;
}

pub fn x_evol_half(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field: &Field1D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let ts = Instant::now();
    let electric_field_potential = field.potential_as_array(t.current, &x.grid[0]);

    multizip((
        psi.psi.axis_iter_mut(Axis(0)),
        atomic_potential.potential.axis_iter(Axis(0)),
        electric_field_potential.iter(),
    ))
    .par_bridge()
    .for_each(|(mut psi_row, atomic_potential_row, field_potential_1e)| {
        multizip((
            psi_row.iter_mut(),
            atomic_potential_row.iter(),
            electric_field_potential.iter(),
        ))
        .for_each(|(psi_elem, atomic_potential_elem, field_potential_2e)| {
            *psi_elem *= (-j
                * 0.5
                * t.dt
                * (atomic_potential_elem - field_potential_1e - field_potential_2e))
                .exp();
        });
    });
    println!("TIME_evol_half = {:?}", ts.elapsed());
}

pub fn x_evol(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field: &Field1D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let ts = Instant::now();
    let electric_field_potential = field.potential_as_array(t.current, &x.grid[0]);

    multizip((
        psi.psi.axis_iter_mut(Axis(0)),
        atomic_potential.potential.axis_iter(Axis(0)),
        electric_field_potential.iter(),
    ))
    .par_bridge()
    .for_each(|(mut psi_row, atomic_potential_row, field_potential_1e)| {
        multizip((
            psi_row.iter_mut(),
            atomic_potential_row.iter(),
            electric_field_potential.iter(),
        ))
        .for_each(|(psi_elem, atomic_potential_elem, field_potential_2e)| {
            *psi_elem *=
                (-j * t.dt * (atomic_potential_elem - field_potential_1e - field_potential_2e))
                    .exp();
        });
    });
    println!("TIME_evol = {:?}", ts.elapsed());
}

pub fn p_evol(psi: &mut WaveFunction, p: &Pspace, dt: f64) {
    // эволюция в импульсном пространстве
    let j = Complex::I;
    let ts = Instant::now();
    psi.psi
        .axis_iter_mut(Axis(0))
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(mut psi_row, px_i)| {
            psi_row
                .iter_mut()
                .zip(p.grid[1].iter())
                .for_each(|(psi_elem, py_k)| {
                    *psi_elem *= (-j * 0.5 * dt * (px_i.powi(2) + py_k.powi(2))).exp();
                });
        });
    println!("Time_p_evol = {:?}", ts.elapsed());
}

pub fn demodify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // демодифицирует "psi для DFT" обратно в psi
    let j = Complex::I;
    let ts = Instant::now();
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
    println!("Time_demodify_psi = {:?}", ts.elapsed());
}

pub fn modify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // модифицирует psi для DFT (в нашем сучае FFT)
    let j = Complex::I;
    let ts = Instant::now();
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
    println!("Time_modify_psi = {:?}", ts.elapsed());
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
        let ts = Instant::now();
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
        println!("Time_fft = {:?}", ts.elapsed());
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction) {
        let ts = Instant::now();
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[0], 0);
        println!("Time_ifft = {:?}", ts.elapsed());
    }
}
