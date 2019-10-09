#!/usr/bin/env bash
jupyter nbconvert --to script base_quadrotor_dynamics.ipynb --output ../src/rigid_body_4_vehicles
sed -i '/# stop here for .py  #/q' ../src/rigid_body_4_vehicles.py
