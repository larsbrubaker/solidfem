//! The aggregation phases of the resumable setup (`setup.rs` holds the state machine and the rest
//! of the per-level sequence): the strength graph, the three aggregation passes and the grouping of
//! each aggregate's members, including the retry with a wider neighbour reach when a level stalls.

use super::super::aggregate::{strong_row, Aggregation};
use super::super::NEAR_NULL_MODES;
use super::MultigridError;
use super::{Flow, MultigridSetup, Phase, MAX_COARSE_FRACTION, MAX_RETRIES, RETRY_WIDENING};

impl MultigridSetup {
    pub(super) fn run_strength(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        while self.cursor < self.current.rows {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            strong_row(
                &self.current,
                &self.positions,
                self.reach,
                self.cursor,
                &mut self.strong_columns,
            );
            self.strong_starts.push(self.strong_columns.len());
            self.cursor += 1;
            *budget -= 1;
        }
        self.cursor = 0;
        self.aggregation = Some(Aggregation::new(self.current.rows));
        self.phase = Phase::SeedPass;
        Ok(Flow::Advanced)
    }

    pub(super) fn run_seed(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        while self.cursor < self.current.rows {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            let row = self.cursor;
            let start = self.strong_starts[row];
            let end = self.strong_starts[row + 1];
            if let Some(aggregation) = &mut self.aggregation {
                aggregation.seed(row, &self.strong_columns[start..end]);
            }
            self.cursor += 1;
            *budget -= 1;
        }
        self.cursor = 0;
        self.phase = Phase::JoinPass;
        Ok(Flow::Advanced)
    }

    pub(super) fn run_join(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        while self.cursor < self.current.rows {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            let row = self.cursor;
            let start = self.strong_starts[row];
            let end = self.strong_starts[row + 1];
            if let Some(aggregation) = &mut self.aggregation {
                aggregation.join(row, &self.current, &self.strong_columns[start..end]);
            }
            self.cursor += 1;
            *budget -= 1;
        }
        self.cursor = 0;
        self.phase = Phase::SweepPass;
        Ok(Flow::Advanced)
    }

    pub(super) fn run_sweep(
        &mut self,
        budget: &mut usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Flow, MultigridError> {
        while self.cursor < self.current.rows {
            if *budget == 0 {
                return Ok(Flow::Paused);
            }
            if cancel() {
                return Ok(Flow::Cancelled);
            }
            let row = self.cursor;
            let start = self.strong_starts[row];
            let end = self.strong_starts[row + 1];
            if let Some(aggregation) = &mut self.aggregation {
                aggregation.sweep(row, &self.strong_columns[start..end]);
            }
            self.cursor += 1;
            *budget -= 1;
        }
        self.cursor = 0;
        self.phase = Phase::Group;
        Ok(Flow::Advanced)
    }

    /// Lists the members of each aggregate, in row order. Linear in the row count, one shot.
    pub(super) fn run_group(&mut self) -> Flow {
        let aggregates = self.aggregate_count();
        let coarse_values = (aggregates * NEAR_NULL_MODES) as f64;
        let coarsens = coarse_values <= MAX_COARSE_FRACTION * self.current.dofs() as f64;
        if !coarsens && self.retries < MAX_RETRIES {
            // Stalled: a wider reach joins more neighbours, the same decision made on evidence.
            self.retries += 1;
            self.reach *= RETRY_WIDENING;
            self.strong_starts.clear();
            self.strong_starts.push(0);
            self.strong_columns.clear();
            self.aggregation = None;
            self.phase = Phase::Strength;
            return Flow::Advanced;
        }
        if !coarsens || aggregates == 0 {
            // Coarsening has stalled for good: this level stops the hierarchy.
            self.is_coarsest = true;
            self.phase = Phase::Spectral;
            return Flow::Advanced;
        }
        let rows = self.current.rows;
        let mut starts = vec![0usize; aggregates + 1];
        if let Some(aggregation) = &self.aggregation {
            for aggregate in &aggregation.aggregate {
                starts[*aggregate as usize + 1] += 1;
            }
        }
        for i in 0..aggregates {
            starts[i + 1] += starts[i];
        }
        let mut filled = starts.clone();
        let mut members = vec![0u32; rows];
        if let Some(aggregation) = &self.aggregation {
            for (row, aggregate) in aggregation.aggregate.iter().enumerate() {
                let aggregate = *aggregate as usize;
                members[filled[aggregate]] = row as u32;
                filled[aggregate] += 1;
            }
        }
        let mut centroids = vec![[0.0; 3]; aggregates];
        for aggregate in 0..aggregates {
            let group = &members[starts[aggregate]..starts[aggregate + 1]];
            for member in group {
                let point = self.positions[*member as usize];
                for axis in 0..3 {
                    centroids[aggregate][axis] += point[axis];
                }
            }
            let count = group.len() as f64;
            for value in &mut centroids[aggregate] {
                *value /= count;
            }
        }
        self.coarse_positions = centroids;
        self.member_starts = starts;
        self.members = members;
        let block = self.current.row_block * NEAR_NULL_MODES;
        self.tentative_values = Vec::with_capacity(rows * block);
        self.tentative_positions = vec![0; rows];
        self.coarse_null = Vec::with_capacity(aggregates * 36);
        self.phase = Phase::Tentative;
        Flow::Advanced
    }
}
