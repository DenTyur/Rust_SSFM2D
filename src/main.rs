#[macro_use]
extern crate fstrings;

mod evolution;
mod field;
mod parameters;
mod potentials;
mod wave_function;
use evolution::{time_step_evol, FftMaker2d};
use field::*;
use parameters::{Pspace, Tspace, Xspace};
use potentials::AtomicPotential;
use wave_function::WaveFunction;

// use std::process::exit;
use std::time::Instant;

fn main() {
    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 5, 1);

    // задаем координатную сетку
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

    // генерация "атомного" потенциала
    let atomic_potential_path = "./arrays_saved/test_br/u.npy";
    let atomic_potential = AtomicPotential::init_from_file(atomic_potential_path, &x);

    // генерация начальной волновой функции psi
    let psi_path = "./arrays_saved/test_br/psi_bound_br_2d_4096_05.npy";
    let mut psi = WaveFunction::init_from_file(psi_path, &x);
    psi.normalization_by_1();

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
            &field1d,
            &atomic_potential,
            &x,
            &p,
            &mut t,
        );
        println!("time_step_evol={}", time.elapsed().as_secs_f64());
        println!("t.current={}, norm = {}", t.current, psi.norm());
    }
}
