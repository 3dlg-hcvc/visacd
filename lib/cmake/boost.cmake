set(BOOST_IOSTREAMS_ENABLE_ZSTD OFF CACHE BOOL "" FORCE)

set(BOOST_IOSTREAMS_ENABLE_LZMA OFF)
set(BOOST_IOSTREAMS_ENABLE_BZIP2 OFF)

include(FetchContent)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(_BOOST_OVERRIDE OVERRIDE_FIND_PACKAGE)
else()
    set(_BOOST_OVERRIDE "")
endif()

if(WIN32)
    FetchContent_Declare(
        boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.zip
        URL_HASH SHA256=ef6cc4ffc95703dfa7e9d21789b27d57c666bcbeced9f6f49162382bb2ea343f
        ${_BOOST_OVERRIDE}
    )
else()
    FetchContent_Declare(
        boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.tar.gz
        URL_HASH SHA256=121da556b718fd7bd700b5f2e734f8004f1cfa78b7d30145471c526ba75a151c
        ${_BOOST_OVERRIDE}
    )
endif()

FetchContent_MakeAvailable(boost)

if(zlib_SOURCE_DIR)
    target_include_directories(boost_iostreams PRIVATE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
endif()
