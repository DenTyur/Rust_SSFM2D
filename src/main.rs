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
use potentials::AtomicPotential;
use wave_function::WaveFunction;

use std::process::exit;
use std::time::Instant;

fn main() {
    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 150, 50);

    // задаем координатную сетку
    // let x = Xspace::new(vec![-100., -100.], vec![1., 1.], vec![200, 200]);
    let x_dir_path = "./arrays_saved/test_br";
    let x = Xspace::load(x_dir_path, 2);

    // инициализируем импульсную сетку на основе координатной сетки
    let p = Pspace::init(&x);

    // инициализируем внешнее поле
    let field1d = Field1D {
        amplitude: 0.045,
        omega: 0.002,
        x_envelop: 50.0001,
    };
    let field2e1d = Field2e1D::new(field1d);

    // генерация "атомного" потенциала
    let atomic_potential_path = "./arrays_saved/test_br/u.npy";
    let atomic_potential = AtomicPotential::init_from_file(atomic_potential_path, &x);

    // генерация начальной волновой функции psi
    let psi_path = "./arrays_saved/test_br/psi_bound_br_2d_4096_05.npy";
    let mut psi = WaveFunction::init_from_file(psi_path, &x);
    psi.normalization_by_1();
    println!("norm psi = {}", psi.norm());
    // exit(0);

    // планировщик fft
    let mut fft = FftMaker2d::new(&x.n);

    for i in 0..t.nt {
        // сохранение временного среза волновой функции
        psi.save_psi(f!("./arrays_saved/tests/psi_t_{i}.npy").as_str())
            .unwrap();
        // эволюция на 1 шаг (t_step) по времени
        let time = Instant::now();
        time_step_evol(
            &mut fft,
            &mut psi,
            &field2e1d,
            &atomic_potential,
            &x,
            &p,
            &mut t,
        );
        println!("time_step_evol={}", time.elapsed().as_secs_f64());
        println!("t.current={}, norm = {}", t.current, psi.norm());
    }
}

fn time_step_evol(
    fft: &mut FftMaker2d,
    psi: &mut WaveFunction,
    field: &Field2e1D,
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
