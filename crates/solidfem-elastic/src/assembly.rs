//! Global stiffness assembly over free nodes. Fixed nodes are eliminated outright: they get no
//! row at all, so the system stays symmetric positive definite and smaller. The cancel check is
//! read once per element, so a large assembly stops promptly.

use solidfem_mesh::QuadraticTetMesh;
use solidfem_solve::CsrMatrix;

use crate::element::element_stiffness;
use crate::error::ModelError;
use crate::material::MaterialField;
use crate::quadrature::Quadrature;
use crate::shape::NODE_COUNT;

/// Marks a node with no free index.
pub(crate) const FIXED: u32 = u32::MAX;

/// Assembles the full stiffness matrix with no node eliminated, three degrees of freedom per
/// node in node order. It has the model's six rigid-body null vectors.
pub fn assemble_stiffness(
    mesh: &QuadraticTetMesh,
    field: &impl MaterialField,
    rule: Quadrature,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<CsrMatrix, ModelError> {
    let free: Vec<u32> = (0..mesh.nodes().len() as u32).collect();
    assemble_free(mesh, field, rule, &free, mesh.nodes().len(), cancel)
}

/// Assembles the stiffness over free nodes; `free[node]` is the node's free index or [`FIXED`].
pub(crate) fn assemble_free(
    mesh: &QuadraticTetMesh,
    field: &impl MaterialField,
    rule: Quadrature,
    free: &[u32],
    free_count: usize,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<CsrMatrix, ModelError> {
    let mapped: Vec<([u32; NODE_COUNT], usize)> = mesh
        .tets()
        .iter()
        .map(|tet| {
            let mut kept = [0u32; NODE_COUNT];
            let mut count = 0;
            for node in tet {
                let index = free[*node as usize];
                if index != FIXED {
                    kept[count] = index;
                    count += 1;
                }
            }
            (kept, count)
        })
        .collect();
    let mut matrix = CsrMatrix::from_block_connectivity(
        free_count,
        3,
        mapped.iter().map(|(kept, count)| &kept[..*count]),
    );

    let mut ke = [0.0; 900];
    for (tet_index, tet) in mesh.tets().iter().enumerate() {
        if cancel() {
            return Err(ModelError::Cancelled);
        }
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        element_stiffness(&nodes, field, rule, &mut ke).map_err(|fault| ModelError::Element {
            tet: tet_index as u32,
            fault,
        })?;
        for a in 0..NODE_COUNT {
            let row_free = free[tet[a] as usize];
            if row_free == FIXED {
                continue;
            }
            for b in 0..NODE_COUNT {
                let column_free = free[tet[b] as usize];
                if column_free == FIXED {
                    continue;
                }
                for i in 0..3 {
                    for k in 0..3 {
                        let row = 3 * row_free as usize + i;
                        // Fits in u32: the free node count is at most the u32 node count, and a
                        // mesh of a billion nodes would not fit in memory anyway.
                        let column = (3 * column_free as usize + k) as u32;
                        matrix.add(row, column, ke[(3 * a + i) * 30 + (3 * b + k)]);
                    }
                }
            }
        }
    }
    Ok(matrix)
}
