use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

pub fn oscillator(x: &Array<f64, Ix1>, y: &Array<f64, Ix1>) -> Array<Complex<f64>, Ix2> {
    let mut psi: Array<Complex<f64>, Ix2> = Array::zeros((x.len(), y.len()));
    let j = Complex::I;
    psi.axis_iter_mut(Axis(0))
        .zip(x.iter())
        .par_bridge()
        .for_each(|(mut psi_row, x_i)| {
            psi_row
                .iter_mut()
                .zip(y.iter())
                .for_each(|(psi_elem, y_j)| {
                    *psi_elem = (-0.5 * (x_i.powi(2) + y_j.powi(2))).exp() + 0. * j;
                })
        });
    psi
}

pub fn norm(psi: &Array<Complex<f64>, Ix2>, dx: f64, dy: f64) -> f64 {
    f64::sqrt(
        psi.mapv(|a| (a.re.powi(2) + a.im.powi(2)))
            .sum_axis(Axis(0))
            .sum(),
    ) * dx
        * dy
}

pub fn normalization_by_1(psi: &mut Array<Complex<f64>, Ix2>, dx: f64, dy: f64) {
    let norm: f64 = norm(psi, dx, dy);
    *psi = &*psi / norm;
}
