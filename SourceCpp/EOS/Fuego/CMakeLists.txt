target_sources(${pelec_exe_name}
   PRIVATE
      EOS.cpp
      EOS_FUEGO.H
   )

target_include_directories(${pelec_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
