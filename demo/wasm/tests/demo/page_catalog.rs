// Keeps the page's committed `catalogJson()` fixture (demo/src/testdata/catalog.json, which the
// page's `bun test` checks every control range against) identical to what the engine offers, so
// those tests can never go on passing against ranges the engine has moved away from.
//
// After a deliberate catalog change, refresh the fixture by running this test with
// SOLIDFEM_BLESS_CATALOG=1 set.

use solidfem_wasm::catalog_json;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/testdata/catalog.json");

#[test]
fn the_pages_catalog_fixture_matches_the_engine() {
    let expected = catalog_json();
    if std::env::var_os("SOLIDFEM_BLESS_CATALOG").is_some() {
        std::fs::write(FIXTURE, format!("{expected}\n")).expect("write the fixture");
    }
    let committed = std::fs::read_to_string(FIXTURE).expect("read the fixture");
    assert_eq!(
        committed.trim_end(),
        expected,
        "demo/src/testdata/catalog.json is out of date: rerun this test with \
         SOLIDFEM_BLESS_CATALOG=1 after a deliberate catalog change"
    );
}
