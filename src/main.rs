#[macro_use]
extern crate fstrings;

mod evol;
mod npy_helper;
mod parameters;
mod potentials;
mod wave_function;
use evol::*;
use parameters::{Pspace, Tspace, Xspace};
use potentials::Potentials;
use wave_function::WaveFunction;

use std::time::Instant;

fn main() {
    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.1, 10, 20);

    // задаем координатную сетку
    let x = Xspace::new(vec![-5.115, -5.115], vec![0.01, 0.01], vec![1024, 1024]);
    x.save("arrays_saved/tests").unwrap();

    // задаем импульсную сетку
    let p = Pspace::init(&x);

    // генерация "атомного" потенциала
    let u = Potentials::oscillator2d(&x);
    u.save("arrays_saved/tests/u.npy").unwrap();

    // генерация начальной волновой функции psi
    let mut psi = WaveFunction::oscillator2d(&x);
    psi.normalization_by_1(&x.dx);
    println!("norm psi = {}", psi.norm(&x.dx));

    // планировщик fft
    let mut fft = FftMaker2d::new(&x.n);

    for i in 0..t.nt {
        // сохранение временного среза волновой функции
        psi.save(f!("arrays_saved/tests/psi_t_{i}.npy").as_str())
            .unwrap();
        // эволюция на 1 шаг (t_step) по времени
        let time = Instant::now();
        time_step_evol(&mut fft, &mut psi, &u, &x, &p, &mut t);
        println!("time_step_evol={}", time.elapsed().as_secs_f64());
        println!("t.current={}, norm = {}", t.current, psi.norm(&x.dx));
    }
}

fn time_step_evol(
    fft: &mut FftMaker2d,
    psi: &mut WaveFunction,
    u: &Potentials,
    x: &Xspace,
    p: &Pspace,
    t: &mut Tspace,
) {
    modify_psi(psi, x, p);
    x_evol_half(psi, u, t.dt);

    for _i in 0..t.n_steps - 1 {
        fft.do_fft(psi);
        // Можно оптимизировать p_evol
        p_evol(psi, p, t.dt);
        fft.do_ifft(psi);
        x_evol(psi, u, t.dt);
        t.current += t.dt;
    }

    fft.do_fft(psi);
    p_evol(psi, p, t.dt);
    fft.do_ifft(psi);
    x_evol_half(psi, u, t.dt);
    demodify_psi(psi, x, p);
    t.current += t.dt;
}
