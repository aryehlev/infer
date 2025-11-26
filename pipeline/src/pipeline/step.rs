/// Pipeline step trait and related types
use std::sync::Arc;
use crate::Result;
use super::context::ExecutionContext;

/// Result of executing a pipeline step
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Continue to the next step
    Continue,
    /// Skip to a named step
    Skip(String),
    /// Stop the pipeline execution
    Stop,
}

/// A single step in a pipeline
pub trait PipelineStep: Send + Sync {
    /// Execute the step with the given context
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult>;

    /// Get the name of this step
    fn name(&self) -> &str;

    /// Optional: Get a description of what this step does
    fn description(&self) -> Option<&str> {
        None
    }
}

/// Type-erased pipeline step
pub type DynPipelineStep = Arc<dyn PipelineStep>;

/// A conditional step that executes different branches based on a predicate
pub struct ConditionalStep {
    name: String,
    predicate: Arc<dyn Fn(&ExecutionContext) -> bool + Send + Sync>,
    then_step: DynPipelineStep,
    else_step: Option<DynPipelineStep>,
}

impl ConditionalStep {
    /// Create a new conditional step
    pub fn new(
        name: impl Into<String>,
        predicate: impl Fn(&ExecutionContext) -> bool + Send + Sync + 'static,
        then_step: DynPipelineStep,
    ) -> Self {
        Self {
            name: name.into(),
            predicate: Arc::new(predicate),
            then_step,
            else_step: None,
        }
    }

    /// Add an else branch
    pub fn with_else(mut self, else_step: DynPipelineStep) -> Self {
        self.else_step = Some(else_step);
        self
    }
}

impl PipelineStep for ConditionalStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        if (self.predicate)(ctx) {
            self.then_step.execute(ctx)
        } else if let Some(else_step) = &self.else_step {
            else_step.execute(ctx)
        } else {
            Ok(StepResult::Continue)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Conditional branch step")
    }
}

/// A step that executes multiple steps in sequence
pub struct SequenceStep {
    name: String,
    steps: Vec<DynPipelineStep>,
}

impl SequenceStep {
    /// Create a new sequence step
    pub fn new(name: impl Into<String>, steps: Vec<DynPipelineStep>) -> Self {
        Self {
            name: name.into(),
            steps,
        }
    }
}

impl PipelineStep for SequenceStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        for step in &self.steps {
            match step.execute(ctx)? {
                StepResult::Continue => continue,
                other => return Ok(other),
            }
        }
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Executes multiple steps in sequence")
    }
}

/// A step that executes multiple steps in parallel
pub struct ParallelStep {
    name: String,
    steps: Vec<DynPipelineStep>,
}

impl ParallelStep {
    /// Create a new parallel step
    pub fn new(name: impl Into<String>, steps: Vec<DynPipelineStep>) -> Self {
        Self {
            name: name.into(),
            steps,
        }
    }
}

impl PipelineStep for ParallelStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        use rayon::prelude::*;

        // Clone context for each parallel branch
        let results: Vec<_> = self.steps
            .par_iter()
            .map(|step| {
                let mut ctx_clone = ctx.clone();
                step.execute(&mut ctx_clone).map(|result| (result, ctx_clone))
            })
            .collect();

        // Merge results back into the main context
        for result in results {
            let (step_result, step_ctx) = result?;

            // Merge context data (later steps overwrite earlier ones)
            for key in step_ctx.keys() {
                if let Some(data) = step_ctx.get(key) {
                    ctx.insert(key.clone(), data.clone());
                }
            }

            // If any step returns Stop or Skip, propagate it
            match step_result {
                StepResult::Continue => continue,
                other => return Ok(other),
            }
        }

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Executes multiple steps in parallel")
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
        ) -> Self {
            Self {
                name: name.into(),
                action: Arc::new(action),
            }
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
    fn test_sequence_step() {
        let step1 = Arc::new(TestStep::new("step1", |ctx| {
            ctx.insert("value", ContextData::Int(1));
            Ok(StepResult::Continue)
        }));

        let step2 = Arc::new(TestStep::new("step2", |ctx| {
            let val = ctx.get("value").unwrap().as_int().unwrap();
            ctx.insert("value", ContextData::Int(val + 1));
            Ok(StepResult::Continue)
        }));

        let seq = SequenceStep::new("sequence", vec![step1, step2]);
        let mut ctx = ExecutionContext::new();

        seq.execute(&mut ctx).unwrap();
        assert_eq!(ctx.get("value").unwrap().as_int(), Some(2));
    }

    #[test]
    fn test_conditional_step() {
        let then_step = Arc::new(TestStep::new("then", |ctx| {
            ctx.insert("result", ContextData::String("then".to_string()));
            Ok(StepResult::Continue)
        }));

        let else_step = Arc::new(TestStep::new("else", |ctx| {
            ctx.insert("result", ContextData::String("else".to_string()));
            Ok(StepResult::Continue)
        }));

        let cond_step = ConditionalStep::new(
            "conditional",
            |ctx| ctx.get("flag").and_then(|d| d.as_bool()).unwrap_or(false),
            then_step,
        )
        .with_else(else_step);

        // Test true branch
        let mut ctx = ExecutionContext::new();
        ctx.insert("flag", ContextData::Bool(true));
        cond_step.execute(&mut ctx).unwrap();
        assert_eq!(ctx.get("result").unwrap().as_string(), Some("then"));

        // Test false branch
        let mut ctx = ExecutionContext::new();
        ctx.insert("flag", ContextData::Bool(false));
        cond_step.execute(&mut ctx).unwrap();
        assert_eq!(ctx.get("result").unwrap().as_string(), Some("else"));
    }

    #[test]
    fn test_parallel_step() {
        let step1 = Arc::new(TestStep::new("step1", |ctx| {
            ctx.insert("a", ContextData::Int(1));
            Ok(StepResult::Continue)
        }));

        let step2 = Arc::new(TestStep::new("step2", |ctx| {
            ctx.insert("b", ContextData::Int(2));
            Ok(StepResult::Continue)
        }));

        let parallel = ParallelStep::new("parallel", vec![step1, step2]);
        let mut ctx = ExecutionContext::new();

        parallel.execute(&mut ctx).unwrap();
        assert_eq!(ctx.get("a").unwrap().as_int(), Some(1));
        assert_eq!(ctx.get("b").unwrap().as_int(), Some(2));
    }
}
