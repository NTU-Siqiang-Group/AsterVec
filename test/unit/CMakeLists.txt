# Unit tests — doctest-based, no external deps beyond AsterVec itself.
add_executable(astervec_unit_tests
    main.cc
    test_sanity.cc
    test_predicate_eq.cc
    test_predicate_comparison.cc
    test_predicate_set.cc
    test_predicate_exists_contains.cc
    test_predicate_logical.cc
    test_predicate_path.cc
    test_predicate_parse.cc
    test_predicate_parse_errors.cc
    test_metadata_store_basic.cc
    test_metadata_store_update.cc
    test_astervec_db_insert_metadata.cc
    test_astervec_db_delete_cascade.cc
    test_metadata_backward_compat.cc
    test_payload_crud.cc
    test_filtered_search.cc
    test_filtered_search_no_regression.cc
    test_bloom_filter.cc
    test_astervec_resolvers.cc
    test_delete_robust_red.cc
    test_delete_persistence.cc
    test_delete_stress.cc
    test_concurrent_search.cc
    test_bulk_build.cc
    # A1.7 Phase 0 torture tests (test_quiesce_backup.cc intentionally
    # SKIPPED — pgastervec-only; see the design docs §4.6).
    test_recall_baseline.cc
    test_concurrent_insert_torture.cc
    test_concurrent_upsert_race.cc
    test_insert_qps_baseline.cc
)
target_include_directories(astervec_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/doctest
    ${CMAKE_SOURCE_DIR}/third_party/json
    ${CMAKE_SOURCE_DIR}/include
    ${ROCKSDB_INCLUDE_DIR}
)
target_link_libraries(astervec_unit_tests PRIVATE astervec_static ${ROCKSDB_LIB})
if(APPLE AND JEMALLOC_LIB)
    target_link_libraries(astervec_unit_tests PRIVATE ${JEMALLOC_LIB})
endif()
