# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yuqianjin/workspace/BrixLab/MNN/demo/exec

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/build

# Include any dependencies generated for this target.
include CMakeFiles/pictureRecognition.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pictureRecognition.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pictureRecognition.out.dir/flags.make

CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.o: CMakeFiles/pictureRecognition.out.dir/flags.make
CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.o: ../pictureRecognition.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuqianjin/workspace/BrixLab/MNN/demo/exec/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.o -c /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/pictureRecognition.cpp

CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/pictureRecognition.cpp > CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.i

CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/pictureRecognition.cpp -o CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.s

# Object files for target pictureRecognition.out
pictureRecognition_out_OBJECTS = \
"CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.o"

# External object files for target pictureRecognition.out
pictureRecognition_out_EXTERNAL_OBJECTS =

pictureRecognition.out: CMakeFiles/pictureRecognition.out.dir/pictureRecognition.cpp.o
pictureRecognition.out: CMakeFiles/pictureRecognition.out.dir/build.make
pictureRecognition.out: CMakeFiles/pictureRecognition.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuqianjin/workspace/BrixLab/MNN/demo/exec/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pictureRecognition.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pictureRecognition.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pictureRecognition.out.dir/build: pictureRecognition.out

.PHONY : CMakeFiles/pictureRecognition.out.dir/build

CMakeFiles/pictureRecognition.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pictureRecognition.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pictureRecognition.out.dir/clean

CMakeFiles/pictureRecognition.out.dir/depend:
	cd /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuqianjin/workspace/BrixLab/MNN/demo/exec /home/yuqianjin/workspace/BrixLab/MNN/demo/exec /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/build /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/build /home/yuqianjin/workspace/BrixLab/MNN/demo/exec/build/CMakeFiles/pictureRecognition.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pictureRecognition.out.dir/depend

