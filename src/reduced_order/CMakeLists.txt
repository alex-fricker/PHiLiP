configure_file(pod_neural_network_rom.py ../../tests/integration_tests_control_files/reduced_order/pod_neural_network_rom.py COPYONLY)

set(SOURCE
    rbf_interpolation.cpp
    pod_basis_online.cpp
    reduced_order_solution.cpp
    rom_test_location.cpp
    pod_basis_offline.cpp
    halton.cpp
    nearest_neighbors.cpp
    min_max_scaler.cpp
    rom_snapshots.cpp)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT PODLib POD_${dim}D)
    add_library(${PODLib} STATIC ${SOURCE})

    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${PODLib} ${ParameterLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${PODLib})
    endif()

    target_compile_definitions(${PODLib} PRIVATE PHILIP_DIM=${dim})

    # Setup target with deal.II
    unset(PODLib)

endforeach()
