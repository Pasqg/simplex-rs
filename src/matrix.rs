use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct Matrix<T> {
    pub rows: usize,
    pub columns: usize,
    data: Vec<T>,
}

impl<T: Clone> Matrix<T> {
    pub fn with_default(rows: usize, columns: usize, default: T) -> Self {
        Self { rows, columns, data: vec!(default; rows * columns)}
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, row_index: usize) -> &Self::Output {
        &self.data[row_index * self.columns..(row_index + 1) * self.columns]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, row_index: usize) -> &mut Self::Output {
        &mut self.data[row_index * self.columns..(row_index + 1) * self.columns]
    }
}