import argparse
import os
import sqlite3
import tracemalloc
import urllib.request
from pathlib import Path
from time import perf_counter

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import psutil


DATA_DIR = Path("data")
DB_DIR = Path("benchmark_dbs")
REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "charts"
PLAN_DIR = REPORT_DIR / "explain_plans"

TRIPDATA_URLS = [
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet",
]
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
PAYMENT_TYPES = [
    (1, "Credit card"),
    (2, "Cash"),
    (3, "No charge"),
    (4, "Dispute"),
    (5, "Unknown"),
    (6, "Voided trip"),
]
QUERIES = {
    "q1_passenger_groupby": """
        SELECT
            passenger_count,
            COUNT(*) AS trip_count,
            ROUND(AVG(total_amount), 2) AS avg_total_amount
        FROM {trip_table}
        GROUP BY passenger_count
        ORDER BY trip_count DESC, passenger_count
    """,
    "q2_payment_aggregation": """
        SELECT
            p.payment_name,s
            COUNT(*) AS trip_count,
            ROUND(SUM(t.total_amount), 2) AS total_revenue
        FROM {trip_table} t
        LEFT JOIN payment_type_lookup p
            ON t.payment_type = p.payment_type
        GROUP BY p.payment_name
        ORDER BY total_revenue DESC
    """,
    "q3_monthly_revenue": """
        SELECT
            pickup_month,
            COUNT(*) AS trip_count,
            ROUND(SUM(total_amount), 2) AS total_revenue
        FROM {trip_table}
        GROUP BY pickup_month
        ORDER BY pickup_month
    """,
    "q4_top_pickup_zones": """
        SELECT
            z.zone,
            z.borough,
            COUNT(*) AS pickup_count
        FROM {trip_table} t
        JOIN zone_lookup z
            ON t.pulocationid = z.locationid
        GROUP BY z.zone, z.borough
        ORDER BY pickup_count DESC
        LIMIT 10
    """,
    "q5_borough_tip_analysis": """
        SELECT
            z.borough,
            ROUND(AVG(t.tip_amount), 2) AS avg_tip_amount,
            ROUND(AVG(t.total_amount), 2) AS avg_total_amount,
            COUNT(*) AS trip_count
        FROM {trip_table} t
        JOIN zone_lookup z
            ON t.pulocationid = z.locationid
        GROUP BY z.borough
        HAVING COUNT(*) > 1000
        ORDER BY avg_tip_amount DESC
    """,
    "q6_zone_rank_window": """
        SELECT
            borough,
            zone,
            pickup_count,
            zone_rank
        FROM (
            SELECT
                z.borough,
                z.zone,
                COUNT(*) AS pickup_count,
                RANK() OVER (
                    PARTITION BY z.borough
                    ORDER BY COUNT(*) DESC
                ) AS zone_rank
            FROM {trip_table} t
            JOIN zone_lookup z
                ON t.pulocationid = z.locationid
            GROUP BY z.borough, z.zone
        ) ranked
        WHERE zone_rank <= 3
        ORDER BY borough, zone_rank, zone
    """,
    "q7_daily_rolling_average": """
        SELECT
            pickup_day,
            daily_revenue,
            ROUND(
                AVG(daily_revenue) OVER (
                    ORDER BY pickup_day
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ),
                2
            ) AS rolling_7d_avg_revenue
        FROM (
            SELECT
                pickup_day,
                SUM(total_amount) AS daily_revenue
            FROM {trip_table}
            GROUP BY pickup_day
        ) daily
        ORDER BY pickup_day
    """,
    "q8_top_routes": """
        SELECT
            pz.zone AS pickup_zone,
            dz.zone AS dropoff_zone,
            COUNT(*) AS trip_count,
            ROUND(AVG(t.total_amount), 2) AS avg_total_amount
        FROM {trip_table} t
        JOIN zone_lookup pz
            ON t.pulocationid = pz.locationid
        JOIN zone_lookup dz
            ON t.dolocationid = dz.locationid
        GROUP BY pz.zone, dz.zone
        ORDER BY trip_count DESC
        LIMIT 10
    """,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SQLite vs DuckDB on analytical queries over NYC Taxi data."
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[1_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000],
        help="Row counts to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per query after one warm-up run.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing local files only.",
    )
    return parser.parse_args()


def ensure_directories():
    for directory in [DATA_DIR, DB_DIR, REPORT_DIR, CHART_DIR, PLAN_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_file(url, destination):
    if destination.exists():
        return
    print(f"Downloading {destination.name} ...")
    urllib.request.urlretrieve(url, destination)


def ensure_datasets(skip_download):
    parquet_files = [DATA_DIR / Path(url).name for url in TRIPDATA_URLS]
    zone_file = DATA_DIR / Path(ZONE_LOOKUP_URL).name

    if not skip_download:
        for url, file_path in zip(TRIPDATA_URLS, parquet_files):
            download_file(url, file_path)
        download_file(ZONE_LOOKUP_URL, zone_file)

    missing_files = [str(path) for path in [*parquet_files, zone_file] if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing dataset files. Either allow downloads or place these files in ./data:\n"
            + "\n".join(missing_files)
        )

    return parquet_files, zone_file


def load_zone_lookup(zone_file):
    zone_lookup = pd.read_csv(zone_file)
    zone_lookup.columns = [column.lower() for column in zone_lookup.columns]
    return zone_lookup


def build_trip_dataframe(parquet_files, max_rows):
    selected_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]
    frames = []
    rows_loaded = 0

    for parquet_file in parquet_files:
        remaining = max_rows - rows_loaded
        if remaining <= 0:
            break
        frame = pd.read_parquet(parquet_file, columns=selected_columns)
        frame = frame.head(remaining)
        frames.append(frame)
        rows_loaded += len(frame)

    if rows_loaded < max_rows:
        raise ValueError(
            f"Only {rows_loaded} rows were available, but {max_rows} rows are required."
        )

    trips = pd.concat(frames, ignore_index=True)
    trips = trips.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
            "PULocationID": "pulocationid",
            "DOLocationID": "dolocationid",
        }
    )
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"])
    trips["dropoff_datetime"] = pd.to_datetime(trips["dropoff_datetime"])
    trips["pickup_month"] = trips["pickup_datetime"].dt.strftime("%Y-%m")
    trips["pickup_day"] = trips["pickup_datetime"].dt.strftime("%Y-%m-%d")
    trips["passenger_count"] = trips["passenger_count"].fillna(0).astype("int32")
    trips["payment_type"] = trips["payment_type"].fillna(0).astype("int32")
    trips["pulocationid"] = trips["pulocationid"].fillna(0).astype("int32")
    trips["dolocationid"] = trips["dolocationid"].fillna(0).astype("int32")

    for column in ["trip_distance", "fare_amount", "tip_amount", "total_amount"]:
        trips[column] = trips[column].fillna(0.0).astype("float64")

    trips["trip_minutes"] = (
        (trips["dropoff_datetime"] - trips["pickup_datetime"]).dt.total_seconds() / 60.0
    ).fillna(0.0)
    trips["trip_minutes"] = trips["trip_minutes"].clip(lower=0.0).astype("float64")

    keep_columns = [
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_month",
        "pickup_day",
        "passenger_count",
        "trip_distance",
        "trip_minutes",
        "pulocationid",
        "dolocationid",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]
    return trips[keep_columns]


def scale_table_name(scale):
    return f"trips_{scale}"


def create_sqlite_database(trips_by_scale, zone_lookup):
    db_path = DB_DIR / "nyc_taxi_sqlite.db"
    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)
    zone_lookup.to_sql("zone_lookup", connection, if_exists="replace", index=False)
    payment_lookup = pd.DataFrame(PAYMENT_TYPES, columns=["payment_type", "payment_name"])
    payment_lookup.to_sql("payment_type_lookup", connection, if_exists="replace", index=False)

    for scale, frame in trips_by_scale.items():
        table_name = scale_table_name(scale)
        frame.to_sql(table_name, connection, if_exists="replace", index=False, chunksize=100_000)

    return connection


def create_duckdb_database(trips_by_scale, zone_lookup):
    db_path = DB_DIR / "nyc_taxi_duckdb.duckdb"
    if db_path.exists():
        db_path.unlink()

    connection = duckdb.connect(str(db_path))
    connection.register("zone_lookup_df", zone_lookup)
    connection.execute("CREATE OR REPLACE TABLE zone_lookup AS SELECT * FROM zone_lookup_df")

    payment_lookup = pd.DataFrame(PAYMENT_TYPES, columns=["payment_type", "payment_name"])
    connection.register("payment_lookup_df", payment_lookup)
    connection.execute(
        "CREATE OR REPLACE TABLE payment_type_lookup AS SELECT * FROM payment_lookup_df"
    )

    for scale, frame in trips_by_scale.items():
        table_name = scale_table_name(scale)
        connection.register("trip_scale_df", frame)
        connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM trip_scale_df")
        connection.unregister("trip_scale_df")

    return connection


def run_sqlite_query(connection, sql):
    cursor = connection.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    return rows


def run_duckdb_query(connection, sql):
    return connection.execute(sql).fetchall()


def fetch_sqlite_explain(connection, sql):
    cursor = connection.cursor()
    cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
    rows = cursor.fetchall()
    return "\n".join(" | ".join(str(value) for value in row) for row in rows)


def fetch_duckdb_explain(connection, sql):
    rows = connection.execute(f"EXPLAIN {sql}").fetchall()
    return "\n".join(str(row[0]) for row in rows)


def measure_query(engine_name, connection, sql):
    process = psutil.Process(os.getpid())
    io_before = process.io_counters()
    tracemalloc.start()
    start = perf_counter()

    if engine_name == "sqlite":
        rows = run_sqlite_query(connection, sql)
    else:
        rows = run_duckdb_query(connection, sql)

    elapsed_ms = (perf_counter() - start) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    io_after = process.io_counters()

    return {
        "latency_ms": round(elapsed_ms, 2),
        "peak_python_memory_mb": round(peak_bytes / (1024 * 1024), 3),
        "read_bytes": max(io_after.read_bytes - io_before.read_bytes, 0),
        "write_bytes": max(io_after.write_bytes - io_before.write_bytes, 0),
        "rows_returned": len(rows),
    }


def benchmark_engine(engine_name, connection, scales, runs):
    records = []

    for scale in scales:
        table_name = scale_table_name(scale)
        for query_name, template in QUERIES.items():
            sql = template.format(trip_table=table_name)

            _ = measure_query(engine_name, connection, sql)

            for run_id in range(1, runs + 1):
                metrics = measure_query(engine_name, connection, sql)
                records.append(
                    {
                        "engine": engine_name,
                        "scale_rows": scale,
                        "query_name": query_name,
                        "run_id": run_id,
                        **metrics,
                    }
                )

    return pd.DataFrame(records)


def save_explain_plans(sqlite_connection, duckdb_connection, largest_scale):
    for query_name, template in QUERIES.items():
        sql = template.format(trip_table=scale_table_name(largest_scale))
        sqlite_plan = fetch_sqlite_explain(sqlite_connection, sql)
        duckdb_plan = fetch_duckdb_explain(duckdb_connection, sql)

        (PLAN_DIR / f"{query_name}_sqlite_explain.txt").write_text(sqlite_plan, encoding="utf-8")
        (PLAN_DIR / f"{query_name}_duckdb_explain.txt").write_text(duckdb_plan, encoding="utf-8")


def summarize_results(results):
    grouped = (
        results.groupby(["engine", "scale_rows", "query_name"], as_index=False)
        .agg(
            median_latency_ms=("latency_ms", "median"),
            avg_latency_ms=("latency_ms", "mean"),
            max_latency_ms=("latency_ms", "max"),
            median_peak_python_memory_mb=("peak_python_memory_mb", "median"),
            median_read_bytes=("read_bytes", "median"),
            median_write_bytes=("write_bytes", "median"),
            rows_returned=("rows_returned", "max"),
        )
        .sort_values(["scale_rows", "query_name", "engine"])
    )
    grouped["avg_latency_ms"] = grouped["avg_latency_ms"].round(2)
    return grouped


def plot_latency_bar(summary, largest_scale):
    figure_data = summary[summary["scale_rows"] == largest_scale].copy()
    pivot = figure_data.pivot(index="query_name", columns="engine", values="median_latency_ms")

    plt.figure(figsize=(12, 7))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title(f"Median Query Latency at {largest_scale:,} Rows")
    plt.xlabel("Query")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Engine")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "latency_bar_by_query.png")
    plt.close()


def plot_scaling_line(summary):
    figure_data = (
        summary.groupby(["engine", "scale_rows"], as_index=False)["median_latency_ms"]
        .median()
        .sort_values(["engine", "scale_rows"])
    )

    plt.figure(figsize=(10, 6))
    for engine_name, engine_data in figure_data.groupby("engine"):
        plt.plot(
            engine_data["scale_rows"],
            engine_data["median_latency_ms"],
            marker="o",
            label=engine_name,
        )
    plt.title("Median Analytical Query Latency vs Dataset Size")
    plt.xlabel("Dataset Size (rows)")
    plt.ylabel("Latency (ms)")
    plt.legend(title="Engine")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "latency_vs_scale.png")
    plt.close()


def plot_memory_bar(summary, largest_scale):
    figure_data = summary[summary["scale_rows"] == largest_scale].copy()
    pivot = figure_data.pivot(
        index="query_name",
        columns="engine",
        values="median_peak_python_memory_mb",
    )

    plt.figure(figsize=(12, 7))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title(f"Median Peak Python Memory at {largest_scale:,} Rows")
    plt.xlabel("Query")
    plt.ylabel("Peak Python Memory (MB)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Engine")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "memory_bar_by_query.png")
    plt.close()


def dataframe_to_markdown_table(dataframe):
    headers = list(dataframe.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in headers) + " |"
        for _, row in dataframe.iterrows()
    ]
    return "\n".join([header_row, separator_row, *rows])


def build_findings(summary, largest_scale):
    largest = summary[summary["scale_rows"] == largest_scale].copy()
    latency_by_engine = (
        largest.groupby("engine", as_index=False)["median_latency_ms"].median().sort_values("median_latency_ms")
    )
    memory_by_engine = (
        largest.groupby("engine", as_index=False)["median_peak_python_memory_mb"]
        .median()
        .sort_values("median_peak_python_memory_mb")
    )
    slowest_query = largest.loc[largest["median_latency_ms"].idxmax()]
    fastest_query = largest.loc[largest["median_latency_ms"].idxmin()]
    highest_io_query = largest.loc[largest["median_read_bytes"].idxmax()]

    lines = [
        (
            f"1. Table 1 and Figure 1 show that `{latency_by_engine.iloc[0]['engine']}` delivered the lower median "
            f"latency at {largest_scale:,} rows, with a workload-median of "
            f"{latency_by_engine.iloc[0]['median_latency_ms']:.2f} ms versus "
            f"{latency_by_engine.iloc[1]['median_latency_ms']:.2f} ms for `{latency_by_engine.iloc[1]['engine']}`."
        ),
        (
            f"2. Table 1 and Figure 3 show that `{memory_by_engine.iloc[0]['engine']}` used the lower median Python-side "
            f"peak memory at {largest_scale:,} rows, at {memory_by_engine.iloc[0]['median_peak_python_memory_mb']:.3f} MB versus "
            f"{memory_by_engine.iloc[1]['median_peak_python_memory_mb']:.3f} MB."
        ),
        (
            f"3. The slowest measured case at the largest scale was `{slowest_query['query_name']}` on `{slowest_query['engine']}` "
            f"at {slowest_query['median_latency_ms']:.2f} ms, while the fastest was `{fastest_query['query_name']}` on "
            f"`{fastest_query['engine']}` at {fastest_query['median_latency_ms']:.2f} ms."
        ),
        (
            f"4. A visible anomaly is `{highest_io_query['query_name']}` on `{highest_io_query['engine']}`, which had the largest "
            f"median read volume at {int(highest_io_query['median_read_bytes'])} bytes. That suggests a heavier scan path than the "
            "other measured workloads."
        ),
        (
            "5. `tracemalloc` measures Python heap allocations, not total database-engine memory. The memory values are therefore "
            "useful for consistent Python-side comparison, but they understate native engine memory usage. This limitation is part "
            "of the benchmark and is stated explicitly rather than hidden."
        ),
    ]
    return lines


def write_analysis_report(summary, scales, largest_scale):
    min_scale = min(scales)
    max_scale = max(scales)
    table_one = (
        summary[summary["scale_rows"] == largest_scale][
            [
                "engine",
                "query_name",
                "median_latency_ms",
                "median_peak_python_memory_mb",
                "median_read_bytes",
                "rows_returned",
            ]
        ]
        .sort_values(["query_name", "engine"])
        .reset_index(drop=True)
    )
    findings = build_findings(summary, largest_scale)
    lines = [
        "# SQLite vs DuckDB for Analytical Queries",
        "",
        "## Summary Table",
        "",
        (
            f"Table 1: Median latency, peak Python memory, read I/O, and result cardinality for each query at "
            f"{largest_scale:,} rows."
        ),
        "",
        dataframe_to_markdown_table(table_one),
        "",
        "## Figures",
        "",
        (
            f"Figure 1: Median query latency (ms) across benchmark queries at {largest_scale:,} rows for SQLite and DuckDB."
        ),
        "",
        "![Figure 1: Median query latency by query](charts/latency_bar_by_query.png)",
        "",
        (
            f"Figure 2: Median analytical query latency (ms) as dataset size scales from "
            f"{min_scale:,} to {max_scale:,} rows."
        ),
        "",
        "![Figure 2: Median latency vs scale](charts/latency_vs_scale.png)",
        "",
        (
            f"Figure 3: Median peak Python memory (MB) across benchmark queries at {largest_scale:,} rows for SQLite and DuckDB."
        ),
        "",
        "![Figure 3: Median peak Python memory by query](charts/memory_bar_by_query.png)",
        "",
        "## Findings",
        "",
        *findings,
        "",
        "## Method Notes",
        "",
        "- Dataset: NYC Yellow Taxi trip data with taxi zone lookup dimension.",
        "- Engines compared: SQLite row-store versus DuckDB columnar engine.",
        "- Measurement tools: `time.perf_counter`, `tracemalloc`, and `psutil` I/O counters.",
        "- All figures were recreated in Python using `matplotlib`.",
        "- No raw screenshots were used.",
    ]
    (REPORT_DIR / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    ensure_directories()
    parquet_files, zone_file = ensure_datasets(args.skip_download)
    zone_lookup = load_zone_lookup(zone_file)
    max_scale = max(args.scales)
    trips = build_trip_dataframe(parquet_files, max_scale)
    trips_by_scale = {scale: trips.head(scale).copy() for scale in sorted(args.scales)}

    print("Creating SQLite and DuckDB databases ...")
    sqlite_connection = create_sqlite_database(trips_by_scale, zone_lookup)
    duckdb_connection = create_duckdb_database(trips_by_scale, zone_lookup)

    print("Running SQLite benchmarks ...")
    sqlite_results = benchmark_engine("sqlite", sqlite_connection, sorted(args.scales), args.runs)
    print("Running DuckDB benchmarks ...")
    duckdb_results = benchmark_engine("duckdb", duckdb_connection, sorted(args.scales), args.runs)

    combined_results = pd.concat([sqlite_results, duckdb_results], ignore_index=True)
    combined_results.to_csv(REPORT_DIR / "benchmark_runs.csv", index=False)

    summary = summarize_results(combined_results)
    summary.to_csv(REPORT_DIR / "benchmark_summary.csv", index=False)

    save_explain_plans(sqlite_connection, duckdb_connection, max_scale)
    plot_latency_bar(summary, max_scale)
    plot_scaling_line(summary)
    plot_memory_bar(summary, max_scale)
    write_analysis_report(summary, sorted(args.scales), max_scale)

    sqlite_connection.close()
    duckdb_connection.close()

    print(f"Benchmark complete. Reports written to: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
