cmake_minimum_required(VERSION 2.8.3)
project(dynamixel_12dof_arm)

find_package(catkin REQUIRED COMPONENTS dynamixel_controllers roseus) # add roseus to gen messages


catkin_package()

#add_rostest(test/test-dxl-12dof-arm.test)
