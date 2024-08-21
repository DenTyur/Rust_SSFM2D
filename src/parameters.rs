use ndarray::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct TimeParameters {
    // параметры временной сетки
    pub t0: f64,
    pub dt: f64,
    pub n_steps: usize,
    pub nt: usize,
}

impl TimeParameters {
    pub fn new(t0: f64, dt: f64, n_steps: usize, nt: usize) -> Self {
        Self {
            t0,
            dt,
            n_steps,
            nt,
        }
    }

    pub fn t_step(&self) -> f64 {
        // возвращает временной шаг срезов волновой функции
        self.dt * self.n_steps as f64
    }

    pub fn last(&self) -> f64 {
        // возвращает последний элемент сетки временных срезов
        self.t0 + self.t_step() * (self.nt - 1) as f64
    }

    pub fn get_grid(&self) -> Array<f64, Ix1> {
        // возвращает временную сетку срезов
        Array::linspace(self.t0, self.last(), self.nt)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct XspaceParameters {
    // параметры координатной сетки
    pub x0: f64,
    pub dx: f64,
    pub n: usize,
}

impl XspaceParameters {
    pub fn new(x0: f64, dx: f64, n: usize) -> Self {
        Self { x0, dx, n }
    }

    pub fn last(&self) -> f64 {
        // Возвращает последний элемент координатной сетки
        self.x0 + self.dx * (self.n - 1) as f64
    }

    pub fn get_grid(&self) -> Array<f64, Ix1> {
        Array::linspace(self.x0, self.last(), self.n)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PspaceParameters {
    pub p0: f64,
    pub dp: f64,
    pub n: usize,
}

impl PspaceParameters {
    pub fn init(prs: XspaceParameters) -> Self {
        Self {
            p0: -3.14 / prs.dx,
            dp: 2. * 3.14 / (prs.n as f64 * prs.dx),
            n: prs.n,
        }
    }

    pub fn last(&self) -> f64 {
        // Возвращает последний элемент импульсной сетки
        self.p0 + self.dp * (self.n - 1) as f64
    }

    pub fn get_grid(&self) -> Array<f64, Ix1> {
        Array::linspace(self.p0, self.last(), self.n)
    }
}
