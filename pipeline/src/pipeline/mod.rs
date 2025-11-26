/// Execution engine for building ML pipelines and RAG workflows
pub mod context;
pub mod pipeline;
pub mod step;
pub mod steps;

pub use context::{ContextData, ExecutionContext};
pub use pipeline::{Pipeline, PipelineBuilder};
pub use step::{DynPipelineStep, PipelineStep, StepResult};
pub use steps::*;
