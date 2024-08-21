use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

pub fn oscillator(x: &Array<f64, Ix1>, y: &Array<f64, Ix1>) -> Array<Complex<f64>, Ix2> {
    let mut u: Array<Complex<f64>, Ix2> = Array::zeros((x.len(), y.len()));
    let j = Complex::I;
    u.axis_iter_mut(Axis(0))
        .zip(x.iter())
        .par_bridge()
        .for_each(|(mut u_row, x_i)| {
            u_row.iter_mut().zip(y.iter()).for_each(|(u_elem, y_j)| {
                *u_elem = 0.5 * (x_i.powi(2) + y_j.powi(2)) + 0. * j;
            })
        });
    u
}
