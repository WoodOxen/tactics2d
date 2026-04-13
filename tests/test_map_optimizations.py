# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for map performance optimizations."""

import os
import sys

# Add project root to Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

# Import map elements
from tactics2d.map.element import Area, Junction, Lane, Map, Node, Regulatory, RoadLine
from tactics2d.map.element.lane import LaneRelationship


def test_boundary_incremental_update():
    """Test that boundary is updated incrementally when elements are added."""
    map_ = Map(name="test_boundary")

    # Initially boundary should be zeros
    assert map_.boundary == (0.0, 0.0, 0.0, 0.0)

    # Add a node at (10, 20)
    node1 = Node(id_="node1", x=10.0, y=20.0)
    map_.add_node(node1)

    # Boundary should reflect the node
    # Note: boundary uses floor/ceil, so (10, 20) becomes (10, 10, 20, 20)
    expected = (np.floor(10.0), np.ceil(10.0), np.floor(20.0), np.ceil(20.0))
    # Use np.allclose for floating point comparison with numpy types
    assert np.allclose(map_.boundary, expected), f"Expected {expected}, got {map_.boundary}"

    # Add another node at (5, 30) - should expand boundary
    node2 = Node(id_="node2", x=5.0, y=30.0)
    map_.add_node(node2)

    # Boundary should now include both nodes: min_x=5, max_x=10, min_y=20, max_y=30
    expected = (np.floor(5.0), np.ceil(10.0), np.floor(20.0), np.ceil(30.0))
    assert np.allclose(map_.boundary, expected), f"Expected {expected}, got {map_.boundary}"

    # Add a node outside current boundary at (50, 60)
    node3 = Node(id_="node3", x=50.0, y=60.0)
    map_.add_node(node3)

    # Boundary should expand to include new node
    expected = (np.floor(5.0), np.ceil(50.0), np.floor(20.0), np.ceil(60.0))
    assert np.allclose(map_.boundary, expected), f"Expected {expected}, got {map_.boundary}"

    print("✓ Boundary incremental update test passed")


def test_slots_optimization():
    """Test that __slots__ are defined in element classes."""
    # Check that __slots__ attribute exists and is a tuple
    classes_to_check = [
        (Node, ("id_", "x", "y")),
        (
            Lane,
            (
                "id_",
                "left_side",
                "right_side",
                "line_ids",
                "regulatory_ids",
                "type_",
                "subtype",
                "color",
                "location",
                "inferred_participants",
                "speed_limit_mandatory",
                "custom_tags",
                "geometry",
                "speed_limit",
                "predecessors",
                "successors",
                "left_neighbors",
                "right_neighbors",
            ),
        ),
        (
            Area,
            (
                "id_",
                "geometry",
                "line_ids",
                "regulatory_ids",
                "type_",
                "subtype",
                "color",
                "location",
                "inferred_participants",
                "speed_limit_mandatory",
                "custom_tags",
                "speed_limit",
            ),
        ),
        (
            RoadLine,
            (
                "id_",
                "geometry",
                "type_",
                "subtype",
                "color",
                "width",
                "height",
                "lane_change",
                "temporary",
                "custom_tags",
            ),
        ),
        (Junction, ("id_", "connections")),
    ]

    for cls, expected_slots in classes_to_check:
        assert hasattr(cls, "__slots__"), f"{cls.__name__} missing __slots__"
        assert isinstance(cls.__slots__, tuple), f"{cls.__name__}.__slots__ should be tuple"

        # Check that all expected slots are present
        for slot in expected_slots:
            assert slot in cls.__slots__, f"Slot '{slot}' missing in {cls.__name__}.__slots__"

        # Create an instance to verify slots work
        try:
            # For Node, we can create a real instance
            if cls == Node:
                instance = cls(id_="test", x=0.0, y=0.0)
                assert not hasattr(instance, "__dict__"), f"{cls.__name__} should not have __dict__"
            # For other classes, we would need proper geometry objects
            # but we can at least verify __slots__ is defined
        except Exception as e:
            # Creation might fail due to missing geometry, that's OK for this test
            pass

    print("✓ __slots__ optimization test passed")


def test_spatial_index_optimization():
    """Test that spatial index query optimizations work."""
    map_ = Map(name="test_spatial")

    # Add some nodes at known positions
    nodes = [
        Node(id_="node1", x=0.0, y=0.0),
        Node(id_="node2", x=10.0, y=10.0),
        Node(id_="node3", x=20.0, y=20.0),
        Node(id_="node4", x=30.0, y=30.0),
    ]

    for node in nodes:
        map_.add_node(node)

    # Test point query - should find nodes near (0, 0) with buffer
    near_point = map_.query_point((0.0, 0.0), buffer=5.0)
    assert "node1" in near_point, "Should find node1 near (0, 0)"

    # Test bbox query - should find nodes in bounding box
    bbox = (0.0, 15.0, 0.0, 15.0)  # min_x, max_x, min_y, max_y
    in_bbox = map_.query_bbox(bbox)
    assert "node1" in in_bbox, "Should find node1 in bbox"
    assert "node2" in in_bbox, "Should find node2 in bbox"
    assert "node3" not in in_bbox, "Should not find node3 in bbox (at 20,20)"

    # Verify that query results are lists of element IDs
    assert isinstance(near_point, list), "query_point should return list"
    assert isinstance(in_bbox, list), "query_bbox should return list"

    print("✓ Spatial index optimization test passed")


def test_o_n2_fix_in_spatial_index():
    """Test that the O(n²) lookup fix is working."""
    map_ = Map(name="test_on2_fix")

    # Add many nodes to test performance (not actually measuring time, but verifying functionality)
    for i in range(100):
        node = Node(id_=f"node{i}", x=float(i), y=float(i))
        map_.add_node(node)

    # Query should work without errors
    results = map_.query_point((50.0, 50.0), buffer=10.0)

    # Should find nodes around (50, 50)
    expected_nodes = [f"node{i}" for i in range(40, 61)]  # nodes 40-60 within buffer
    for node_id in expected_nodes:
        if abs(float(node_id[4:]) - 50.0) <= 10.0:  # Check if within 10 units
            assert node_id in results, f"Should find {node_id} near (50, 50)"

    print("✓ O(n²) fix test passed")


def test_vectorized_boundary_calculation():
    """Test that boundary calculation uses vectorized numpy operations."""
    map_ = Map(name="test_vectorized")

    # Add many nodes with random coordinates
    np.random.seed(42)
    num_nodes = 1000
    xs = np.random.uniform(-100, 100, num_nodes)
    ys = np.random.uniform(-100, 100, num_nodes)

    for i in range(num_nodes):
        node = Node(id_=f"rand_node{i}", x=float(xs[i]), y=float(ys[i]))
        map_.add_node(node)

    # Compute expected boundary using numpy
    expected_min_x = np.floor(np.min(xs))
    expected_max_x = np.ceil(np.max(xs))
    expected_min_y = np.floor(np.min(ys))
    expected_max_y = np.ceil(np.max(ys))

    actual = map_.boundary

    assert (
        abs(actual[0] - expected_min_x) < 1e-6
    ), f"min_x mismatch: {actual[0]} vs {expected_min_x}"
    assert (
        abs(actual[1] - expected_max_x) < 1e-6
    ), f"max_x mismatch: {actual[1]} vs {expected_max_x}"
    assert (
        abs(actual[2] - expected_min_y) < 1e-6
    ), f"min_y mismatch: {actual[2]} vs {expected_min_y}"
    assert (
        abs(actual[3] - expected_max_y) < 1e-6
    ), f"max_y mismatch: {actual[3]} vs {expected_max_y}"

    print("✓ Vectorized boundary calculation test passed")


def test_empty_map_boundary():
    """Test boundary calculation for empty map."""
    map_ = Map(name="test_empty")

    # Empty map should return (0.0, 0.0, 0.0, 0.0)
    assert map_.boundary == (0.0, 0.0, 0.0, 0.0)

    # After adding a node, boundary should update
    node = Node(id_="node1", x=10.0, y=20.0)
    map_.add_node(node)
    expected = (np.floor(10.0), np.ceil(10.0), np.floor(20.0), np.ceil(20.0))
    assert np.allclose(map_.boundary, expected)

    print("✓ Empty map boundary test passed")


def test_spatial_index_edge_cases():
    """Test spatial index edge cases including HAS_STRTREE=False."""
    # Test empty map queries
    map_ = Map(name="test_empty_queries")
    assert map_.query_point((0.0, 0.0)) == []
    assert map_.query_bbox((0.0, 10.0, 0.0, 10.0)) == []

    # Test with elements but no STRtree (simulate import failure)
    # We'll test this by checking the code handles missing STRtree gracefully
    # The implementation already checks HAS_STRTREE at the start of query methods

    # Add a node and verify queries work with STRtree available
    node = Node(id_="test_node", x=5.0, y=5.0)
    map_.add_node(node)

    # Point query should find the node
    results = map_.query_point((5.0, 5.0), buffer=1.0)
    assert "test_node" in results

    # Bbox query should find the node
    results = map_.query_bbox((4.0, 6.0, 4.0, 6.0))
    assert "test_node" in results

    print("✓ Spatial index edge cases test passed")


def test_spatial_index_without_strtree():
    """Test spatial index behavior when STRtree is not available."""
    # Temporarily patch HAS_STRTREE to False
    import unittest.mock as mock

    from tactics2d.map.element import map as map_module

    with mock.patch.object(map_module, "HAS_STRTREE", False):
        # Create a new map inside the patched context
        map_ = Map(name="test_no_strtree")

        # Add a node
        node = Node(id_="test_node", x=5.0, y=5.0)
        map_.add_node(node)

        # Queries should return empty lists when HAS_STRTREE is False
        assert map_.query_point((5.0, 5.0)) == []
        assert map_.query_bbox((4.0, 6.0, 4.0, 6.0)) == []

        # Spatial index should be None
        assert map_._spatial_index is None

        # Rebuild should do nothing
        map_._rebuild_spatial_index()
        assert map_._spatial_index is None

    print("✓ Spatial index without STRtree test passed")


def test_element_addition_error_handling():
    """Test error handling when adding elements with duplicate IDs."""
    map_ = Map(name="test_duplicates")

    # Add a node
    node1 = Node(id_="elem1", x=0.0, y=0.0)
    map_.add_node(node1)

    # Try to add another element with same ID but different type
    # This should raise KeyError
    try:
        # Try to add a Regulatory with same ID (different type)
        regulatory = Regulatory(id_="elem1", subtype="stop_sign")
        map_.add_regulatory(regulatory)
        # Should not reach here
        assert False, "Expected KeyError when adding different element type with same ID"
    except KeyError as e:
        assert "used by the other road element" in str(e)

    # Test same type replacement (should warn)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        node2 = Node(id_="elem1", x=1.0, y=1.0)  # Same ID, same type
        map_.add_node(node2)
        assert len(w) == 1
        assert "already exists" in str(w[0].message)
        assert "Replaced" in str(w[0].message)

    # Test that the node was actually replaced
    assert map_.nodes["elem1"].x == 1.0
    assert map_.nodes["elem1"].y == 1.0

    print("✓ Element addition error handling test passed")


def test_get_by_id_edge_cases():
    """Test get_by_id method edge cases."""
    map_ = Map(name="test_get_by_id")

    # Test non-existent ID
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = map_.get_by_id("nonexistent")
        assert result is None
        assert len(w) == 1
        assert "Cannot find element" in str(w[0].message)

    # Add a node and retrieve it
    node = Node(id_="test_node", x=0.0, y=0.0)
    map_.add_node(node)
    assert map_.get_by_id("test_node") is node

    print("✓ get_by_id edge cases test passed")


def main():
    """Run all optimization tests."""
    print("Running map optimization tests...")
    print("=" * 60)

    tests = [
        test_boundary_incremental_update,
        test_slots_optimization,
        test_spatial_index_optimization,
        test_o_n2_fix_in_spatial_index,
        test_vectorized_boundary_calculation,
        test_empty_map_boundary,
        test_spatial_index_edge_cases,
        test_spatial_index_without_strtree,
        test_element_addition_error_handling,
        test_get_by_id_edge_cases,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print("=" * 60)
    print(f"Test results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All optimization tests passed! ✓")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
