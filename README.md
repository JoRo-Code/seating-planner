# SeatPlanner

When you want different seatings for e.g. appetizers, main dish and dessert. 
SeatPlanner ptimizes seat arrangements according to conditions: diverse neighbours, gender, fixed seats, corners. 

Available at [SeatPlan](https://seatplan.streamlit.app/)

## Dev

### Install

Install UV as package manager (instead of pip):
```sh
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment
```sh
uv venv

```

```sh
source .venv/bin/activate
```

Install dependencies
```sh
uv pip install -r requirements.txt
```


### Run

```sh
uv run app.py
```