#[macro_use]
extern crate fstrings;

mod evol;
mod field;
mod parameters;
mod potentials;
mod wave_function;
use evol::*;
use field::*;
use parameters::{Pspace, Tspace, Xspace};
use potentials::Potentials;
use wave_function::WaveFunction;

use std::process::exit;
use std::time::Instant;

fn main() {
    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 150, 50);

    // задаем координатную сетку
    // let x = Xspace::new(vec![-100., -100.], vec![1., 1.], vec![200, 200]);
    let x = Xspace::load("./arrays_saved/test_br", 2);

    // инициализируем внешнее поле
    let field = Field2d::new(0.045, 0.002, 50.0001);

    // задаем импульсную сетку
    let p = Pspace::init(&x);

    // генерация "атомного" потенциала
    let u = Potentials::load2d("./arrays_saved/test_br/u.npy");

    // генерация начальной волновой функции psi
    let mut psi = WaveFunction::load2d("./arrays_saved/test_br/psi_bound_br_2d_4096_05.npy");
    psi.normalization_by_1(&x.dx);
    println!("norm psi = {}", psi.norm(&x.dx));

    // планировщик fft
    let mut fft = FftMaker2d::new(&x.n);

    for i in 0..t.nt {
        // сохранение временного среза волновой функции
        psi.save(f!("./arrays_saved/test_br_long/psi_t_{i}.npy").as_str())
            .unwrap();
        // эволюция на 1 шаг (t_step) по времени
        let time = Instant::now();
        time_step_evol(&mut fft, &mut psi, &field, &u, &x, &p, &mut t);
        println!("time_step_evol={}", time.elapsed().as_secs_f64());
        println!("t.current={}, norm = {}", t.current, psi.norm(&x.dx));
    }
}

fn time_step_evol(
    fft: &mut FftMaker2d,
    psi: &mut WaveFunction,
    field: &Field2d,
    u: &Potentials,
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
