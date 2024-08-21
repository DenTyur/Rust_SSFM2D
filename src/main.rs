#[macro_use]
extern crate fstrings;

mod evol;
mod hdf5_helper;
mod npy_helper;
mod parameters;
mod potentials;
mod wave_function;
use evol::*;
use parameters::{PspaceParameters, TimeParameters, XspaceParameters};

use ndarray::prelude::*;
use num_complex::Complex;
use std::time::Instant;

fn main() {
    // задаем параметры временной сетки
    let tprs = TimeParameters::new(0., 0.1, 10, 50);
    let _t_arr = tprs.get_grid();
    let mut t_current: f64 = tprs.t0;

    // задаем параметры сетки x и y и генерируем массивы этих сеток
    let xprs = XspaceParameters::new(-20., 0.04, 1024);
    let yprs = xprs;
    let x: Array<f64, Ix1> = xprs.get_grid();
    let y: Array<f64, Ix1> = yprs.get_grid();
    npy_helper::write_1d_f64("arrays_saved/x.npy", &x).unwrap();
    npy_helper::write_1d_f64("arrays_saved/y.npy", &y).unwrap();

    // генерация импульсной сетки на основе координатной сетки
    let pxprs = PspaceParameters::init(xprs);
    let pyprs = PspaceParameters::init(yprs);
    let px: Array<f64, Ix1> = pxprs.get_grid();
    let py: Array<f64, Ix1> = pyprs.get_grid();

    // генерация "атомного" потенциала
    let u: Array<Complex<f64>, Ix2> = potentials::oscillator(&x, &y);
    npy_helper::write_2d_c64("arrays_saved/u.npy", &u).unwrap();

    // генерация начальной волновой функции psi
    let mut psi: Array<Complex<f64>, Ix2> = wave_function::oscillator(&x, &y);
    wave_function::normalization_by_1(&mut psi, xprs.dx, yprs.dx);
    println!("norm psi = {}", wave_function::norm(&psi, xprs.dx, yprs.dx));

    // планировщик fft
    let mut fft = FftMaker::new(xprs.n, yprs.n);

    for i in 0..tprs.nt {
        let time = Instant::now();
        time_step_evol(
            &mut fft,
            &mut psi,
            &u,
            &x,
            &y,
            &px,
            &py,
            xprs,
            yprs,
            pxprs,
            pyprs,
            tprs,
            &mut t_current,
        );
        // сохранение временного среза волновой функции
        let psi_path = f!("arrays_saved/psi_t_{i}.npy");
        npy_helper::write_2d_c64(psi_path.as_str(), &psi).unwrap();

        println!("time_step_evol={}", time.elapsed().as_secs_f64());
        println!("t_current={}", t_current);
    }
}

fn time_step_evol(
    fft: &mut FftMaker,
    psi: &mut Array2<Complex<f64>>,
    u: &Array2<Complex<f64>>,
    x: &Array<f64, Ix1>,
    y: &Array<f64, Ix1>,
    px: &Array<f64, Ix1>,
    py: &Array<f64, Ix1>,
    xprs: XspaceParameters,
    yprs: XspaceParameters,
    pxprs: PspaceParameters,
    pyprs: PspaceParameters,
    tprs: TimeParameters,
    t_current: &mut f64,
) {
    modify_psi(psi, x, y, pxprs.p0, pyprs.p0, xprs.dx, yprs.dx);
    x_evol_half(psi, u, tprs.dt);

    for _i in 0..tprs.n_steps - 1 {
        fft.do_fft(psi);
        p_evol(psi, px, py, tprs.dt);
        fft.do_ifft(psi);
        x_evol(psi, u, tprs.dt);
        *t_current += tprs.dt;
    }

    fft.do_fft(psi);
    p_evol(psi, px, py, tprs.dt);
    fft.do_ifft(psi);
    x_evol_half(psi, u, tprs.dt);
    demodify_psi(psi, x, y, pxprs.p0, pyprs.p0, xprs.dx, yprs.dx);
    *t_current += tprs.dt;
}
