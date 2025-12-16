setsid bash -c '
	source /opt/ros/melodic/setup.bash
	source ~/catkin_ws/devel/setup.bash
	roslaunch cs4023-major-project launch.launch >gazebo.log 2>&1 &
	sleep 15
	source /home/kars0009/catkin_ws/src/cs4023-major-project/.venv/bin/activate
	python3.7 -u  /home/kars0009/catkin_ws/src/cs4023-major-project/cs4023-major-project/src/master.py > train.log 2>&1
' < /dev/null > log.log 2>&1 &

