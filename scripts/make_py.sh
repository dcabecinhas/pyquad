#!/usr/bin/env bash
jupyter nbconvert --to script base_rigid_body_4_vehicles.ipynb --output ../src/rigid_body_4_vehicles
sed -i '/# stop here for .py  #/q' ../src/rigid_body_4_vehicles.py
sed -i 's/SIMULATE_REAL_QUADROTOR = True/SIMULATE_REAL_QUADROTOR = False/g' ../src/rigid_body_4_vehicles.py

jupyter nbconvert --to script base_point_mass_4_vehicles.ipynb --output ../src/point_mass_4_vehicles
sed -i '/# stop here for .py  #/q' ../src/point_mass_4_vehicles.py
sed -i 's/SIMULATE_REAL_QUADROTOR = True/SIMULATE_REAL_QUADROTOR = False/g' ../src/point_mass_4_vehicles.py
