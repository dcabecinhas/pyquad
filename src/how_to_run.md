How to run multiple quadrotors transporting load:
=================================================

Window 1:

    source ~/setup_ros.sh ; cd aero-manager/; ./start_vehicle_comms_multiple.sh 8 log

Window 2:

    ./launch_experiment.sh


## Flight Checklist

- Setup markers on each quadrotor

- Setup markers on load

- Define vehicles on VICON Tracker

- Make sure that all quadrotors and load are not easily confused in VICON

- Double check initial quadrotor positions agains REFERENCE positions in OFFBOARD mode

- Run `source ~/setup_ros.sh ; cd aero-manager/; ./start_vehicle_comms_multiple.sh 8 log` and see that vehicles show up on QGroundControl

- Check that changing MANUAL / POSITION / OFFBOARD toggle is reflected on QGC

- For each vehicle

    - Check for defects and broken propellers
    - Power the vehicle with full battery
    - Set vehicle in MANUAL mode
    - Arm vehicle
    - Test MANUAL mode flight
    - Test POSITION mode flight
    - Test OFFBOARD mode flight (running `./launch_experiment.sh`)

- Test with 2 vehicles simultaneously (Only OFFBOARD position control)

- Test with 3 vehicles simultaneously (Only OFFBOARD position control)

- Test with 4 vehicles simultaneously (Only OFFBOARD position control)

- Test OFFBOARD LOAD controller (be prepared to revert back to OFFBOARD position control)

