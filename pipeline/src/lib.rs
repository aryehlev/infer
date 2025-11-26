//! # Infer-Pipeline
//!
//! Execution engine for building ML pipelines and RAG (Retrieval-Augmented Generation) workflows.
//!
//! This crate provides a flexible pipeline system for composing multi-step ML workflows
//! with support for:
//! - Conditional branching
//! - Parallel execution
//! - Model inference
//! - Data transformations
//! - RAG (retrieval, ranking, prompt construction)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use infer_pipeline::prelude::*;
//! use std::sync::Arc;
//!
//! // Create a pipeline
//! let pipeline = Pipeline::builder("my_workflow")
//!     .add_fn("load_data", |ctx| {
//!         // Load data into context
//!         Ok(StepResult::Continue)
//!     })
//!     .add_fn("process", |ctx| {
//!         // Process data
//!         Ok(StepResult::Continue)
//!     })
//!     .build();
//!
//! // Execute the pipeline
//! let mut ctx = ExecutionContext::new();
//! pipeline.execute(&mut ctx)?;
//! # Ok::<(), infer_lib::InferError>(())
//! ```

pub mod pipeline;

// Re-export core types from lib
pub use infer_lib::{
    Result, InferError,
    ModelOutput, ModelInput,
    ModelRegistry, Model, ModelBackend,
    DataFrameTransformer,
};

pub use pipeline::{
    context::{ExecutionContext, ContextData},
    step::{PipelineStep, DynPipelineStep, StepResult, ConditionalStep, SequenceStep, ParallelStep},
    pipeline::{Pipeline, PipelineBuilder},
    steps,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::pipeline::{
        ExecutionContext, ContextData,
        PipelineStep, DynPipelineStep, StepResult,
        Pipeline, PipelineBuilder,
    };
    pub use crate::steps;

    // Re-export core lib types
    pub use infer_lib::prelude::*;
}
