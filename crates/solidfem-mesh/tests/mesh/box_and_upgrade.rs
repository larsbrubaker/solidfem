// The box builder, the 10-node upgrade and boundary extraction, checked against closed-form
// counts for a structured box: a wrong count here means shared edges or faces were not shared.

use solidfem_mesh::{box_tet_mesh, tet_volume, MeshError, QuadraticTetMesh, TetMesh};

#[test]
fn a_box_fills_its_volume_with_positive_tets() {
    let mesh = box_tet_mesh([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [2, 3, 4]).expect("box");
    assert_eq!(mesh.tets().len(), 6 * 2 * 3 * 4);
    let mut total = 0.0;
    for tet in mesh.tets() {
        let [a, b, c, d] = tet.map(|n| mesh.positions()[n as usize]);
        let volume = tet_volume(a, b, c, d);
        assert!(volume > 0.0);
        total += volume;
    }
    // Volumes sum to 24 exactly but for rounding in 144 additions of short decimals.
    assert!((total - 24.0).abs() < 1e-12, "volume {total}");
}

#[test]
fn the_upgrade_shares_each_edge_midpoint_and_counts_match_a_structured_grid() {
    let (nx, ny, nz) = (3usize, 2usize, 2usize);
    let mesh = QuadraticTetMesh::from_linear(
        &box_tet_mesh([0.0; 3], [3.0, 2.0, 2.0], [nx, ny, nz]).expect("box"),
    );
    // A six-tet-per-cell grid has one edge per grid line plus face and body diagonals, and the
    // quadratic node grid is exactly the doubled grid.
    assert_eq!(
        mesh.nodes().len(),
        (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1)
    );
    assert_eq!(mesh.vertex_count(), (nx + 1) * (ny + 1) * (nz + 1));
}

#[test]
fn the_boundary_is_each_outer_grid_face_cut_in_two() {
    let (nx, ny, nz) = (3usize, 2usize, 2usize);
    let mesh = QuadraticTetMesh::from_linear(
        &box_tet_mesh([0.0; 3], [3.0, 2.0, 2.0], [nx, ny, nz]).expect("box"),
    );
    let faces = mesh.boundary_faces();
    assert_eq!(faces.len(), 4 * (nx * ny + ny * nz + nz * nx));
    // Every boundary face's outward normal must point away from its tet's far corner.
    for face in &faces {
        let tet = mesh.tets()[face.tet as usize];
        let p = |n: u32| mesh.nodes()[n as usize];
        let (a, b, c) = (p(face.nodes[0]), p(face.nodes[1]), p(face.nodes[2]));
        let far = p(tet[face.face as usize]);
        assert!(
            tet_volume(a, b, c, far) < 0.0,
            "face {face:?} points inward"
        );
    }
    assert_eq!(mesh.face_connected_bodies().len(), 1);
}

#[test]
fn a_tet_flat_to_rounding_is_refused_by_name_at_every_scale() {
    // Four corners in one plane but for 1e-16 of a unit, so the determinant is positive only by
    // an amount the coordinates cannot resolve; the ratio must not depend on the part's size.
    for scale in [1e-3, 1.0, 1e3] {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.25 * scale, 0.25 * scale, 1e-16 * scale],
        ];
        assert!(tet_volume(positions[0], positions[1], positions[2], positions[3]) > 0.0);
        let error = TetMesh::new(positions, vec![[0, 1, 2, 3]]).expect_err("flat to rounding");
        assert!(
            matches!(error, MeshError::FlatTet { tet: 0, .. }),
            "scale {scale}: {error}"
        );
    }
    // A very thin but genuine element is kept: 1e-6 is a million times what rounding can reach.
    let thin = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.25, 0.25, 1e-6],
    ];
    TetMesh::new(thin, vec![[0, 1, 2, 3]]).expect("thin is not flat");
}

#[test]
fn an_inverted_tet_is_refused_by_name() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let error = TetMesh::new(positions, vec![[0, 2, 1, 3]]).expect_err("inverted");
    assert!(matches!(error, MeshError::DegenerateTet { tet: 0, .. }));
}

#[test]
fn an_unused_node_is_refused_by_name() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [9.0, 9.0, 9.0],
    ];
    let error = TetMesh::new(positions, vec![[0, 1, 2, 3]]).expect_err("unused");
    assert_eq!(error, MeshError::UnusedNode { node: 4 });
}
