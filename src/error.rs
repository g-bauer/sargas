use thiserror::Error;
#[cfg(feature = "chemfiles")]
use chemfiles;

#[derive(Error, Debug)]
pub enum SargasError {
    #[cfg(feature = "chemfiles")]
    #[error(transparent)]
    ChemfilesError(#[from] chemfiles::Error),
    #[error("End of trajectory reached.")]
    TrajectoryEnd,
    #[error("invalid value for cut off (maximum: {maximum:?}, found: {found:?})")]
    InvalidCutoff {
        maximum: f64,
        found: f64,
    },
}

#[cfg(feature = "python")]
mod python {
    use super::SargasError;
    use pyo3::exceptions::PyIOError;
    use pyo3::PyErr;

    impl From<SargasError> for PyErr {
        fn from(err: SargasError) -> Self {
            PyIOError::new_err(err.to_string())
        }
    }
}
