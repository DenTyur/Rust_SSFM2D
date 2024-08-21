use hdf5::File;
use ndarray::Array2;

pub fn write(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create("data.h5")?;

    let dataset = file
        .new_dataset::<f64>()
        .shape(data.dim())
        .create("data_real")?;

    dataset.write(data.as_slice().expect("Fail"))?;

    Ok(())
}
