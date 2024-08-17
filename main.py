"""
Summary: CLI for running routing / dock door allocation models
Created by: Charles Tang
For: BJ's Wholesale Robotics Team
"""

from datetime import date, timedelta, datetime
import pandas as pd

import click
from src.data_load import load_aggregate_data, get_shipping_doors_volume
import src.models
from src.distance_map import load_distance_map


@click.command()
@click.option("--dc", type=click.INT, help="800, 820, or 840")
@click.option(
    "--routing",
    help="Choose a model for routing: HistoricalData, NearestNeighbor, RandomGroupsTSP, VRP",
)
@click.option(
    "--doors",
    help="Choose a model for shipping lane allocations: None, ClusterModel, GreedyModel, LinProgModel, ClusterSpacedModel",
)
@click.option("--forklifts", type=click.INT, help="Number of forklifts for simuation")
@click.option(
    "--date-start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(date.today()),
    help="Start date for simulation, inclusive",
)
@click.option(
    "--date-end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(date.today() + timedelta(days=1)),
    help="End date for simulation, inclusive",
)
def run_models(
    dc: int,
    routing: str,
    doors: str,
    forklifts: int,
    date_start: datetime,
    date_end: datetime,
):
    """
    Runs models in CLI interface on backtested data. Calibrated on same data.
    """
    click.echo(
        f"Running simulation with DC{dc}, {routing} Routing Model, {doors} Door Model, {forklifts} Forklifts from {date_start.date} to {date_end.date} "
    )
    # Aggregate data from date_start to date_end
    df = load_aggregate_data(date_start, dc=dc)
    for n in range(1, int((date_end - date_start).days) + 1):
        d = date_start + timedelta(days=n)
        df = pd.concat([df, load_aggregate_data(d, dc=dc)], ignore_index=False)
    click.echo(f"...Crossdocking tasks loaded (N={df.shape[0]})....")
    if doors:
        door_model = None
        model_args = {
            "tasks": df,
            "shipping_doors_counts": get_shipping_doors_volume(tasks=df, dc=dc),
            "original_distance_map": load_distance_map(dc=dc),
            "dc": dc,
        }
        match doors:
            case "ClusterModel":
                door_model = src.models.ClusterModel(**model_args)
            case "GreedyModel":
                door_model = src.models.GreedyModel(**model_args)
            case "LinProgModel":
                door_model = src.models.LinProgModel(**model_args)
            case "ClusterSpacedModel":
                door_model = src.models.ClusterSpacedModel(**model_args)
            case _:
                click.echo(f"...Using original DC{dc} distance map...")
        if door_model:
            door_model.assign_doors()
        new_dist_map = (
            door_model.calculate_distance_map()
            if door_model
            else model_args["original_distance_map"]
        )

        click.echo("...Door allocation completed...")

    model_args = {
        "distance_map": new_dist_map,
        "tasks": df,
        "num_forklifts": forklifts,
    }
    historical_model = src.models.HistoricalData(**model_args)
    routing_model = None
    match routing:
        case "HistoricalData":
            routing_model = src.models.HistoricalData(**model_args)
        case "NearestNeighbor":
            routing_model = src.models.NearestNeighbor(**model_args)
        case "RandomGroupsTSP":
            routing_model = src.models.RandomGroupsTSP(**model_args)
        case "VRP":
            routing_model = src.models.VRP(**model_args)
        case _:
            click.echo(
                "...No routing model specified, please try again with the allowed models..."
            )

    if routing_model:
        historical_dist = int(src.models.backtest_model(historical_model, dc=dc))
        dist = int(src.models.backtest_model(routing_model, dc=dc))
        click.echo("...Models backtested...")

    if historical_dist and dist:
        improvement = round(100 * (historical_dist - dist) / (historical_dist), 1)
        click.echo(
            f"Results: {improvement}% improvement in pallet travel distance on {routing} model compared to Historical Data"
        )


if __name__ == "__main__":
    run_models()
