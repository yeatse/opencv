# Detect Metal backend support on Apple platforms
# Note: Metal backend uses MPSGraph internally

if(APPLE)
    if(WITH_METAL)
        # Check for Metal framework
        find_library(METAL_FRAMEWORK Metal)
        find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)
        find_library(FOUNDATION_FRAMEWORK Foundation)

        if(METAL_FRAMEWORK AND MPSGRAPH_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            # Check minimum OS version
            # MPSGraph requires iOS 14+, macOS 11+, tvOS 14+, visionOS 1+

            # Test compilation
            try_compile(VALID_METAL
                "${OpenCV_BINARY_DIR}"
                SOURCES "${OpenCV_SOURCE_DIR}/cmake/checks/metal.mm"
                LINK_LIBRARIES
                    "${FOUNDATION_FRAMEWORK}"
                    "${METAL_FRAMEWORK}"
                    "${MPSGRAPH_FRAMEWORK}"
                OUTPUT_VARIABLE TRY_OUT
            )

            if(VALID_METAL)
                set(HAVE_METAL ON)
                message(STATUS "Metal Backend: YES (using MPSGraph)")
                message(STATUS "  Metal Framework: ${METAL_FRAMEWORK}")
                message(STATUS "  MPSGraph Framework: ${MPSGRAPH_FRAMEWORK}")
            else()
                message(WARNING "Metal backend compilation test failed")
                message(STATUS "${TRY_OUT}")
            endif()
        else()
            message(STATUS "Metal Backend: NO (frameworks not found)")
        endif()
    else()
        message(STATUS "Metal Backend: DISABLED (WITH_METAL=OFF)")
    endif()
else()
    message(STATUS "Metal Backend: NO (Apple platforms only)")
endif()
