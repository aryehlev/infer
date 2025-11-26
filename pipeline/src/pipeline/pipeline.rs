/// Pipeline for composing and executing multiple steps
use std::sync::Arc;
use crate::Result;
use super::context::ExecutionContext;
use super::step::{DynPipelineStep, PipelineStep, StepResult};

/// A pipeline that executes a sequence of steps
#[derive(Clone)]
pub struct Pipeline {
    name: String,
    steps: Vec<DynPipelineStep>,
}

impl Pipeline {
    /// Create a new pipeline builder
    pub fn builder(name: impl Into<String>) -> PipelineBuilder {
        PipelineBuilder::new(name)
    }

    /// Execute the pipeline with the given context
    pub fn execute(&self, ctx: &mut ExecutionContext) -> Result<()> {
        let mut i = 0;
        while i < self.steps.len() {
            let step = &self.steps[i];
            match step.execute(ctx)? {
                StepResult::Continue => {
                    i += 1;
                }
                StepResult::Skip(target) => {
                    // Find the target step and skip to it
                    if let Some(pos) = self.steps.iter().position(|s| s.name() == target) {
                        i = pos;
                    } else {
                        i += 1;
                    }
                }
                StepResult::Stop => return Ok(()),
            }
        }
        Ok(())
    }

    /// Get the name of this pipeline
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of steps in this pipeline
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the pipeline has no steps
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the names of all steps in this pipeline
    pub fn step_names(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.name()).collect()
    }
}

/// Builder for constructing pipelines
pub struct PipelineBuilder {
    name: String,
    steps: Vec<DynPipelineStep>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
        }
    }

    /// Add a step to the pipeline
    pub fn add_step(mut self, step: DynPipelineStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add multiple steps to the pipeline
    pub fn add_steps(mut self, steps: Vec<DynPipelineStep>) -> Self {
        self.steps.extend(steps);
        self
    }

    /// Add a step with a closure
    pub fn add_fn<F>(self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&mut ExecutionContext) -> Result<StepResult> + Send + Sync + 'static,
    {
        struct ClosureStep<F> {
            name: String,
            f: F,
        }

        impl<F> PipelineStep for ClosureStep<F>
        where
            F: Fn(&mut ExecutionContext) -> Result<StepResult> + Send + Sync,
        {
            fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
                (self.f)(ctx)
            }

            fn name(&self) -> &str {
                &self.name
            }
        }

        let step = Arc::new(ClosureStep {
            name: name.into(),
            f,
        });
        self.add_step(step)
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline {
        Pipeline {
            name: self.name,
            steps: self.steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::ContextData;

    struct TestStep {
        name: String,
        action: Arc<dyn Fn(&mut ExecutionContext) -> Result<StepResult> + Send + Sync>,
    }

    impl TestStep {
        fn new(
            name: impl Into<String>,
            action: impl Fn(&mut ExecutionContext) -> Result<StepResult> + Send + Sync + 'static,
        ) -> Arc<Self> {
            Arc::new(Self {
                name: name.into(),
                action: Arc::new(action),
            })
        }
    }

    impl PipelineStep for TestStep {
        fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
            (self.action)(ctx)
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_pipeline_basic() {
        let pipeline = Pipeline::builder("test")
            .add_step(TestStep::new("step1", |ctx| {
                ctx.insert("value", ContextData::Int(1));
                Ok(StepResult::Continue)
            }))
            .add_step(TestStep::new("step2", |ctx| {
                let val = ctx.get("value").unwrap().as_int().unwrap();
                ctx.insert("value", ContextData::Int(val + 1));
                Ok(StepResult::Continue)
            }))
            .build();

        let mut ctx = ExecutionContext::new();
        pipeline.execute(&mut ctx).unwrap();

        assert_eq!(ctx.get("value").unwrap().as_int(), Some(2));
    }

    #[test]
    fn test_pipeline_stop() {
        let pipeline = Pipeline::builder("test")
            .add_step(TestStep::new("step1", |ctx| {
                ctx.insert("step1", ContextData::Bool(true));
                Ok(StepResult::Stop)
            }))
            .add_step(TestStep::new("step2", |ctx| {
                ctx.insert("step2", ContextData::Bool(true));
                Ok(StepResult::Continue)
            }))
            .build();

        let mut ctx = ExecutionContext::new();
        pipeline.execute(&mut ctx).unwrap();

        assert!(ctx.contains("step1"));
        assert!(!ctx.contains("step2")); // Should not execute
    }

    #[test]
    fn test_pipeline_skip() {
        let pipeline = Pipeline::builder("test")
            .add_step(TestStep::new("step1", |ctx| {
                ctx.insert("step1", ContextData::Bool(true));
                Ok(StepResult::Skip("step3".to_string()))
            }))
            .add_step(TestStep::new("step2", |ctx| {
                ctx.insert("step2", ContextData::Bool(true));
                Ok(StepResult::Continue)
            }))
            .add_step(TestStep::new("step3", |ctx| {
                ctx.insert("step3", ContextData::Bool(true));
                Ok(StepResult::Continue)
            }))
            .build();

        let mut ctx = ExecutionContext::new();
        pipeline.execute(&mut ctx).unwrap();

        assert!(ctx.contains("step1"));
        assert!(!ctx.contains("step2")); // Should be skipped
        assert!(ctx.contains("step3"));
    }

    #[test]
    fn test_pipeline_metadata() {
        let pipeline = Pipeline::builder("my_pipeline")
            .add_step(TestStep::new("step1", |_| Ok(StepResult::Continue)))
            .add_step(TestStep::new("step2", |_| Ok(StepResult::Continue)))
            .build();

        assert_eq!(pipeline.name(), "my_pipeline");
        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());
        assert_eq!(pipeline.step_names(), vec!["step1", "step2"]);
    }

    #[test]
    fn test_pipeline_add_fn() {
        let pipeline = Pipeline::builder("test")
            .add_fn("increment", |ctx| {
                let val = ctx.get("value").and_then(|d| d.as_int()).unwrap_or(0);
                ctx.insert("value", ContextData::Int(val + 1));
                Ok(StepResult::Continue)
            })
            .add_fn("double", |ctx| {
                let val = ctx.get("value").unwrap().as_int().unwrap();
                ctx.insert("value", ContextData::Int(val * 2));
                Ok(StepResult::Continue)
            })
            .build();

        let mut ctx = ExecutionContext::new();
        pipeline.execute(&mut ctx).unwrap();

        // (0 + 1) * 2 = 2
        assert_eq!(ctx.get("value").unwrap().as_int(), Some(2));
    }
}
