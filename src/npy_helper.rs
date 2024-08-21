use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use std::fs::File;
use std::io::BufWriter;

pub fn write_2d_c64(path: &str, arr: &Array2<Complex<f64>>) -> Result<(), WriteNpyError> {
    let writer = BufWriter::new(File::create(path)?);
    arr.write_npy(writer)?;
    Ok(())
}

pub fn write_1d_f64(path: &str, arr: &Array<f64, Ix1>) -> Result<(), WriteNpyError> {
    let writer = BufWriter::new(File::create(path)?);
    arr.write_npy(writer)?;
    Ok(())
}
// fn read_c64() -> Result<(), ReadNpyError> {
//     let reader = File::open("array.npy")?;
//     let arr = Array2::<i32>::read_npy(reader)?;
//     println!("arr =\n{}", arr);
//     Ok(())
// }
