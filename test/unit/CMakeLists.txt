# Unit tests — doctest-based, no external deps beyond LSM-Vec itself.
add_executable(lsmvec_unit_tests
    main.cc
    test_sanity.cc
)
target_include_directories(lsmvec_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/doctest
    ${CMAKE_SOURCE_DIR}/third_party/json
    ${CMAKE_SOURCE_DIR}/include
    ${ROCKSDB_INCLUDE_DIR}
)
target_link_libraries(lsmvec_unit_tests PRIVATE lsmvec_static ${ROCKSDB_LIB})
if(APPLE AND JEMALLOC_LIB)
    target_link_libraries(lsmvec_unit_tests PRIVATE ${JEMALLOC_LIB})
endif()
