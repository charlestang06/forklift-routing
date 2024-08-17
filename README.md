# Fleetflow: Improving DC Forklift Efficiency
By Charles Tang, Will Buttrey (TPM)
Summer 2024 Internship
BJ's Wholesale Club

## Motivation
Xdocking (Dock <-> Dock) pallet processes may be inefficient. This project aims to investigate and measure these inefficiencies and propose a model/algorithm to improve pallet Xdocking procedures.

**Problem Formulation**
- ~20 forklifts, each with capacity of up to 2 pallets
- ~100 shipping doors, 100 receiving doors, 10 PTC locations (Xdocking only)
- List of orders (start, end, demand)

## Overview
1. Excel map (Crossdock Maps/***.csv)
2. Load grid into Python (gridload.py)
3. Calculate distances or load distance map (distance_map.py)
4. Load and clean data from: Dock to Dock, Receiving to PTC/IBNP, PTC to Shipping (data_load.py)
5. Improve routing models in models.py
6. Improve dock door allocations in distance_map.py
7. Test models in test.py

## Setup
1. ```pip install -r requirements.txt```
2. Run routing or dock door allocation examples in `/examples` directory. For example:
```python examples/run_routing_model.py```

# Glossary

## Map Construction
- X = walls, 6.67 x 6.67 ft (since dock doors are about that wide and spacing is same width as dock doors)
- R = cannot go left because highway rules
- L = cannot go right because highway rules
- Numbers / Codes represent picking locations

## DCs
- C*** = cart (train staging locations)
- Dock doors ~4 -> ~200
- 6*1 = PTC in (exact number differs by DC)
- 6*2 = PTC out (exact number differs by DC)
- BRK = breakdown
- IBNP - inbound staging for storage
- MNOB - outbound staging for shipping
- ***R = receiving dock door
- ***S = shipping dock door
- ***A = candy dock door
- ***B = candy dock door
- ***CDY* = candy dock door
"# forklift-routing" 
